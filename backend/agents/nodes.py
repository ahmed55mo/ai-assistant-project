from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from backend.agents.confirmation import PendingConfirmation, confirmation_prompt
from backend.agents.state import AgentState
from backend.tools.base import ConfirmationRequired, ToolError


def make_model_node(model: Any, tools: list[Any]):
    try:
        bound_model = model.bind_tools(tools)
    except NotImplementedError:
        # Lightweight fake models used for graph tests may not implement binding.
        bound_model = model

    def agent_model(state: AgentState) -> dict[str, Any]:
        try:
            response = bound_model.invoke(state["messages"])
        except Exception as exc:
            return {"last_error": str(exc)}
        return {"messages": [response], "last_error": None}

    return agent_model


def make_tool_node(tools: list[Any]):
    tool_map = {tool.name: tool for tool in tools}

    def execute_tools(state: AgentState) -> dict[str, Any]:
        assistant = state["messages"][-1]
        if not isinstance(assistant, AIMessage):
            return {"last_error": "Tool execution requires an assistant tool call."}

        results: list[ToolMessage] = []
        for call in assistant.tool_calls:
            tool_name = call["name"]
            call_id = call["id"]
            arguments = dict(call.get("args", {}))
            tool = tool_map.get(tool_name)
            if tool is None:
                results.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool: {tool_name}"}),
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )
                continue
            try:
                if tool.metadata and tool.metadata.get("requires_confirmation"):
                    raise ConfirmationRequired(f"Confirmation required before running {tool_name}.")
                result = tool.invoke(arguments)
            except ConfirmationRequired:
                pending: PendingConfirmation = {
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
                return {
                    "messages": [
                        ToolMessage(
                            content=json.dumps({"error": "Confirmation required."}),
                            tool_call_id=call_id,
                            name=tool_name,
                        ),
                        AIMessage(content=confirmation_prompt(tool_name, arguments)),
                    ],
                    "pending_confirmation": pending,
                    "last_error": None,
                }
            except (ToolError, ValueError, TypeError) as exc:
                result = {"error": str(exc)}
            results.append(
                ToolMessage(
                    content=json.dumps(result, default=str),
                    tool_call_id=call_id,
                    name=tool_name,
                )
            )
        return {"messages": results, "pending_confirmation": None, "last_error": None}

    return execute_tools


def route_after_model(state: AgentState) -> str:
    if state.get("last_error"):
        return "end"
    if state.get("pending_confirmation"):
        return "end"
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"


def route_after_tools(state: AgentState) -> str:
    return "end" if state.get("pending_confirmation") else "agent"
