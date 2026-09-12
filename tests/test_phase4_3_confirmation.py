from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend import main
from backend.agents.langgraph_agent import LangGraphAgent
from backend.agents.runtime import AgentRuntimeFactory
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ToolDefinition, ToolRegistry
from backend.tools.langchain_tools import build_langchain_tools


def _registry(execute: Any, *, read_only: bool = False) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                "send_email",
                "Send an email.",
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
                execute,
                read_only=read_only,
                requires_confirmation=not read_only,
            )
        ]
    )


class ScriptedModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        self.tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        return self.responses.pop(0)


def _tool_call(call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "send_email",
                "args": {
                    "recipient": "recipient@example.com",
                    "subject": "Subject",
                    "body": "Body",
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _agent(execute: Any, responses: list[AIMessage]) -> tuple[LangGraphAgent, ConversationMemory]:
    memory = ConversationMemory()
    agent = LangGraphAgent(
        memory,
        _registry(execute),
        model=ScriptedModel(responses),
    )
    return agent, memory


def test_side_effect_is_blocked_and_pending_state_is_created() -> None:
    calls: list[dict[str, Any]] = []
    agent, memory = _agent(
        lambda arguments: calls.append(arguments) or {"sent": True},
        [_tool_call()],
    )
    conversation_id = memory.create_conversation()

    response = agent.respond(conversation_id, "send the email")

    assert "Should I send it?" in response
    assert calls == []
    assert agent.pending[conversation_id]["tool_call_id"] == "call-1"
    assert agent.pending[conversation_id]["tool_name"] == "send_email"


def test_confirmation_resumes_once_and_clears_pending() -> None:
    calls: list[dict[str, Any]] = []
    agent, memory = _agent(
        lambda arguments: calls.append(arguments) or {"sent": True},
        [
            _tool_call("call-2"),
            AIMessage(content="Email sent."),
            AIMessage(content="No pending operation."),
        ],
    )
    conversation_id = memory.create_conversation()
    agent.respond(conversation_id, "send it")

    response = agent.respond(conversation_id, "yes")
    repeated = agent.respond(conversation_id, "yes")

    assert response == "Email sent."
    assert repeated == "No pending operation."
    assert len(calls) == 1
    assert conversation_id not in agent.pending
    tool_messages = [
        item
        for item in memory.get_history(conversation_id)
        if item.get("role") == "tool"
    ]
    assert tool_messages[-1]["tool_call_id"] == "call-2"


def test_rejection_clears_pending_without_execution() -> None:
    calls: list[dict[str, Any]] = []
    agent, memory = _agent(
        lambda arguments: calls.append(arguments) or {"sent": True},
        [_tool_call()],
    )
    conversation_id = memory.create_conversation()
    agent.respond(conversation_id, "send it")

    response = agent.respond(conversation_id, "no")

    assert response == "I did not make the requested change."
    assert calls == []
    assert conversation_id not in agent.pending


def test_ambiguous_response_keeps_pending_and_does_not_execute() -> None:
    calls: list[dict[str, Any]] = []
    agent, memory = _agent(
        lambda arguments: calls.append(arguments) or {"sent": True},
        [_tool_call()],
    )
    conversation_id = memory.create_conversation()
    prompt = agent.respond(conversation_id, "send it")

    response = agent.respond(conversation_id, "I will decide later")

    assert response == prompt
    assert calls == []
    assert conversation_id in agent.pending


def test_pending_confirmation_is_conversation_scoped() -> None:
    calls: list[dict[str, Any]] = []
    agent, memory = _agent(
        lambda arguments: calls.append(arguments) or {"sent": True},
        [_tool_call("call-a"), AIMessage(content="No operation requested.")],
    )
    conversation_a = memory.create_conversation()
    conversation_b = memory.create_conversation()
    agent.respond(conversation_a, "send it")

    response_b = agent.respond(conversation_b, "yes")

    assert response_b == "No operation requested."
    assert calls == []
    assert conversation_a in agent.pending


def test_read_only_tool_does_not_require_confirmation() -> None:
    definition = ToolDefinition(
        "calculator",
        "Calculate.",
        {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        lambda arguments: {"result": 4},
    )
    registry = ToolRegistry([definition])
    assert registry.get("calculator").requires_confirmation is False
    assert registry.execute("calculator", {"expression": "2+2"}) == {"result": 4}


def test_api_confirmation_flow_uses_langgraph_without_external_calls(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    memory = ConversationMemory()
    registry = _registry(lambda arguments: calls.append(arguments) or {"sent": True})
    model = ScriptedModel([_tool_call("api-call"), AIMessage(content="Completed.")])
    agent = LangGraphAgent(memory, registry, model=model)
    monkeypatch.setenv("AGENT_ENGINE", "langgraph")
    monkeypatch.setattr(main, "conversation_memory", memory)
    monkeypatch.setattr(main, "conversation_service", None)
    monkeypatch.setattr(
        main,
        "runtime_factory",
        AgentRuntimeFactory(langgraph_factory=lambda runtime_memory, _: agent),
    )

    client = TestClient(main.app)
    first = client.post("/api/chat", json={"message": "send email"})
    conversation_id = first.json()["conversation_id"]
    second = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "yes"},
    )

    assert first.status_code == 200
    assert "Should I send it?" in first.json()["response"]
    assert second.status_code == 200
    assert second.json() == {
        "conversation_id": conversation_id,
        "response": "Completed.",
    }
    assert len(calls) == 1
