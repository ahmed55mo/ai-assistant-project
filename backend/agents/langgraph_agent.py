from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from backend.agents.confirmation import (
    PendingConfirmation,
    confirmation_prompt,
    is_confirmation,
    is_rejection,
)
from backend.agents.errors import AgentConfigurationError
from backend.agents.nodes import (
    make_model_node,
    make_tool_node,
    route_after_model,
    route_after_tools,
)
from backend.agents.state import AgentState
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ConfirmationRequired, ToolError, ToolRegistry
from backend.tools.langchain_tools import build_langchain_tools
from backend.rag.tool import reset_trusted_scope, set_trusted_scope



class ConversationMemoryAdapter:
    """Translate Phase 3 memory records to and from LangChain messages."""

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    def load(self, conversation_id: str) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for item in self.memory.get_llm_history(conversation_id):
            role = item.get("role")
            if role == "user":
                messages.append(HumanMessage(content=item.get("content", "")))
            elif role == "assistant":
                tool_calls = [
                    {
                        "name": call["function"]["name"],
                        "args": json.loads(call["function"].get("arguments", "{}")),
                        "id": call["id"],
                        "type": "tool_call",
                    }
                    for call in item.get("tool_calls", [])
                ]
                messages.append(
                    AIMessage(content=item.get("content", ""), tool_calls=tool_calls)
                )
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=item.get("content", ""),
                        tool_call_id=item["tool_call_id"],
                        name=item.get("name"),
                    )
                )
        return messages

    def append_new(self, conversation_id: str, messages: list[BaseMessage]) -> None:
        for message in messages:
            if isinstance(message, HumanMessage):
                self.memory.add_message(conversation_id, "user", str(message.content))
            elif isinstance(message, AIMessage):
                calls = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                    for call in message.tool_calls
                ]
                self.memory.add_message(
                    conversation_id,
                    "assistant",
                    str(message.content),
                    **({"tool_calls": calls} if calls else {}),
                )
            elif isinstance(message, ToolMessage):
                self.memory.add_message(
                    conversation_id,
                    "tool",
                    str(message.content),
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )


def build_agent_graph(model: Any, tools: list[Any]):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", make_model_node(model, tools))
    workflow.add_node("tools", make_tool_node(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_model,
        {"tools": "tools", "end": END},
    )
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {"agent": "agent", "end": END},
    )
    return workflow.compile()


class LangGraphAgent:
    def __init__(
        self,
        memory: ConversationMemory,
        registry: ToolRegistry,
        model: Any | None = None,
    ) -> None:
        self.memory = ConversationMemoryAdapter(memory)
        self.registry = registry
        self.tools = build_langchain_tools(registry)
        self.model = model or _build_chatgroq()
        self.graph = build_agent_graph(self.model, self.tools)
        self.pending: dict[str, PendingConfirmation] = {}
        self.user_id = "anonymous"

    def set_user_context(self, user_id: str) -> None:
        self.user_id = user_id.strip() or "anonymous"

    def respond(self, conversation_id: str, user_message: str) -> str:
        return self.respond_with_sources(conversation_id, user_message)["response"]

    def respond_with_sources(
        self, conversation_id: str, user_message: str
    ) -> dict[str, Any]:
        pending = self.pending.get(conversation_id)
        if pending is not None:
            response = self._continue_pending(conversation_id, user_message, pending)
            return {"response": response, "sources": []}

        history = self.memory.load(conversation_id)
        initial_count = len(history)
        history.append(HumanMessage(content=user_message))
        scope_token = set_trusted_scope(self.user_id, conversation_id)
        try:
            result = self.graph.invoke(
                {
                    "conversation_id": conversation_id,
                    "messages": history,
                    "pending_confirmation": None,
                    "last_error": None,
                }
            )
        finally:
            reset_trusted_scope(scope_token)
        generated = result["messages"][initial_count:]
        pending_result = result.get("pending_confirmation")
        if pending_result:
            self.pending[conversation_id] = pending_result
            generated = [
                message
                for message in generated
                if not (
                    isinstance(message, ToolMessage)
                    and message.tool_call_id == pending_result["tool_call_id"]
                )
            ]
        self.memory.append_new(conversation_id, generated)
        if result.get("last_error"):
            raise AgentConfigurationError(result["last_error"])
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                return {
                    "response": str(message.content).strip(),
                    "sources": _extract_sources(generated),
                }
        raise AgentConfigurationError("The LangGraph agent returned an empty response.")

    def _continue_pending(
        self,
        conversation_id: str,
        user_message: str,
        pending: PendingConfirmation,
    ) -> str:
        if is_rejection(user_message):
            self.pending.pop(conversation_id, None)
            self.memory.memory.add_message(
                conversation_id, "user", user_message, llm_visible=False
            )
            response = "I did not make the requested change."
            self.memory.memory.add_message(conversation_id, "assistant", response)
            return response

        if not is_confirmation(user_message):
            return confirmation_prompt(pending["tool_name"], pending["arguments"])

        # Remove the pending entry before execution so a repeated request cannot
        # execute the same operation twice, even if the provider raises.
        self.pending.pop(conversation_id, None)
        self.memory.memory.add_message(
            conversation_id, "user", user_message, llm_visible=False
        )
        try:
            result = self.registry.execute(
                pending["tool_name"], pending["arguments"], confirmed=True
            )
        except (ConfirmationRequired, ToolError) as exc:
            result = {"error": str(exc)}

        history = self.memory.load(conversation_id)
        history = [
            message
            for message in history
            if not (
                isinstance(message, ToolMessage)
                and message.tool_call_id == pending["tool_call_id"]
            )
        ]
        tool_message = ToolMessage(
            content=json.dumps(result, default=str),
            tool_call_id=pending["tool_call_id"],
            name=pending["tool_name"],
        )
        history.append(tool_message)
        base_count = len(history)
        scope_token = set_trusted_scope(self.user_id, conversation_id)
        try:
            graph_result = self.graph.invoke(
                {
                    "conversation_id": conversation_id,
                    "messages": history,
                    "pending_confirmation": None,
                    "last_error": None,
                }
            )
        finally:
            reset_trusted_scope(scope_token)
        self.memory.memory.add_message(
            conversation_id,
            "tool",
            tool_message.content,
            tool_call_id=tool_message.tool_call_id,
            name=tool_message.name,
        )
        self.memory.append_new(conversation_id, graph_result["messages"][base_count:])
        if graph_result.get("last_error"):
            raise AgentConfigurationError(graph_result["last_error"])
        for message in reversed(graph_result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                return str(message.content).strip()
        raise AgentConfigurationError("The LangGraph agent returned an empty response.")


def _extract_sources(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "search_knowledge":
            continue
        try:
            payload = json.loads(str(message.content))
        except (TypeError, ValueError):
            continue
        for source in payload.get("sources", []):
            document_id = source.get("document_id")
            filename = source.get("filename")
            chunk_id = source.get("chunk_id")
            page_number = source.get("page_number")
            if not all(isinstance(value, str) and value for value in (document_id, filename, chunk_id)):
                continue
            if page_number is not None and (
                isinstance(page_number, bool) or not isinstance(page_number, int)
            ):
                continue
            key = (document_id, page_number)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                }
            )
    return sources


def _build_chatgroq() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise AgentConfigurationError("GROQ_API_KEY is not configured.")
    return ChatGroq(
        api_key=api_key,
        model=os.getenv("MODEL_NAME", "openai/gpt-oss-120b"),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )


__all__ = ["ConversationMemoryAdapter", "LangGraphAgent", "build_agent_graph"]
