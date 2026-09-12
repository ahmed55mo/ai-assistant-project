from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agents.langgraph_agent import ConversationMemoryAdapter, build_agent_graph
from backend.agents.state import AgentState
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ToolDefinition, ToolRegistry
from backend.tools.langchain_tools import build_langchain_tools
from backend.tools.registry import build_registry


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                "calculator",
                "Calculate.",
                {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                    "additionalProperties": False,
                },
                lambda args: {"result": 4},
            ),
            ToolDefinition(
                "send_email",
                "Send email.",
                {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                    "additionalProperties": False,
                },
                lambda args: args,
                read_only=False,
                requires_confirmation=True,
            ),
            ToolDefinition(
                "send_telegram_message",
                "Send Telegram.",
                {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                    "additionalProperties": False,
                },
                lambda args: args,
                read_only=False,
                requires_confirmation=True,
            ),
        ]
    )


def test_adapters_expose_tools_without_provider_secrets() -> None:
    tools = {tool.name: tool for tool in build_langchain_tools(_registry())}
    assert {"calculator", "send_email", "send_telegram_message"} <= tools.keys()
    assert set(tools["send_telegram_message"].args_schema.model_fields) == {"message"}
    assert set(tools["send_email"].args_schema.model_fields) == {
        "recipient",
        "subject",
        "body",
    }
    schemas = json.dumps(
        [tool.args_schema.model_json_schema() for tool in tools.values()]
    )
    for secret in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "RESEND_API_KEY"):
        assert secret not in schemas


def test_all_phase3_tool_categories_are_available() -> None:
    names = {tool.name for tool in build_langchain_tools(build_registry())}
    assert {"calculator", "search_emails", "list_events", "send_email", "send_telegram_message"} <= names


def test_agent_state_and_memory_adapter_preserve_message_metadata() -> None:
    memory = ConversationMemory()
    conversation_id = memory.create_conversation()
    memory.add_message(conversation_id, "user", "calculate 2+2")
    memory.add_message(
        conversation_id,
        "assistant",
        "",
        tool_calls=[
            {
                "id": "call-123",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression":"2+2"}'},
            }
        ],
    )
    memory.add_message(
        conversation_id,
        "tool",
        '{"result":4}',
        tool_call_id="call-123",
        name="calculator",
    )
    messages = ConversationMemoryAdapter(memory).load(conversation_id)
    state: AgentState = {
        "conversation_id": conversation_id,
        "messages": messages,
        "pending_confirmation": None,
        "last_error": None,
    }
    assert isinstance(state["messages"][0], HumanMessage)
    assert isinstance(state["messages"][1], AIMessage)
    assert state["messages"][1].tool_calls[0]["id"] == "call-123"
    assert isinstance(state["messages"][2], ToolMessage)
    assert state["messages"][2].tool_call_id == "call-123"


class _FakeModel:
    def bind_tools(self, tools: list[Any]) -> "_FakeModel":
        self.tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content="mocked response")


def test_graph_constructs_and_invokes_without_external_api() -> None:
    graph = build_agent_graph(_FakeModel(), build_langchain_tools(_registry()))
    result = graph.invoke(
        {
            "conversation_id": "conversation",
            "messages": [HumanMessage(content="hello")],
            "pending_confirmation": None,
            "last_error": None,
        }
    )
    assert result["messages"][-1].content == "mocked response"
