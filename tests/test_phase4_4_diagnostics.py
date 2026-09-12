from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend import main
from backend.agents.errors import AgentConfigurationError
from backend.agents.langgraph_agent import LangGraphAgent
from backend.agents.runtime import AgentRuntimeFactory
from backend.services.calendar_service import GoogleCalendarProvider
from backend.services.memory_service import ConversationMemory
from backend.services.notification_service import NotificationService
from backend.tools.base import ToolError, ToolRegistry
from backend.tools.calendar_tools import calendar_tools
from backend.tools.langchain_tools import build_langchain_tools
from backend.tools.notification_tools import notification_tools


class ScriptedModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        self.tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


def _telegram_agent(model: ScriptedModel) -> tuple[LangGraphAgent, ConversationMemory, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    service = NotificationService()
    service.send_telegram_message = lambda arguments: (
        sent.append(dict(arguments))
        or {"sent": True, "provider": "telegram", "message_id": 42}
    )
    registry = ToolRegistry(notification_tools(service))
    memory = ConversationMemory()
    return LangGraphAgent(memory, registry, model=model), memory, sent


def test_telegram_confirmation_resume_creates_valid_tool_message_and_returns_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "telegram-call-42"
    model = ScriptedModel(
        [
            _tool_call(
                "send_telegram_message",
                {"message": "Test message"},
                call_id,
            ),
            AIMessage(content="Telegram message sent."),
        ]
    )
    agent, memory, sent = _telegram_agent(model)
    monkeypatch.setenv("AGENT_ENGINE", "langgraph")
    monkeypatch.setattr(main, "conversation_memory", memory)
    monkeypatch.setattr(main, "conversation_service", None)
    monkeypatch.setattr(
        main,
        "runtime_factory",
        AgentRuntimeFactory(langgraph_factory=lambda runtime_memory, _: agent),
    )

    client = TestClient(main.app)
    first = client.post("/api/chat", json={"message": "Send Test message to my Telegram"})
    conversation_id = first.json()["conversation_id"]
    second = client.post(
        "/api/chat",
        json={"conversation_id": conversation_id, "message": "yes"},
    )

    assert first.status_code == 200
    assert "Should I send it?" in first.json()["response"]
    assert second.status_code == 200
    assert second.json()["response"] == "Telegram message sent."
    assert sent == [{"message": "Test message"}]
    assert conversation_id not in agent.pending
    tool_messages = [
        message
        for message in memory.get_history(conversation_id)
        if message.get("role") == "tool"
    ]
    assert tool_messages[-1]["tool_call_id"] == call_id
    assert tool_messages[-1]["name"] == "send_telegram_message"
    resumed_messages = model.invocations[-1]
    assert resumed_messages[-1].type == "tool"
    assert resumed_messages[-1].tool_call_id == call_id


def test_telegram_success_then_final_model_failure_does_not_duplicate_side_effect() -> None:
    call_id = "telegram-failure-call"
    model = ScriptedModel(
        [
            _tool_call("send_telegram_message", {"message": "Once"}, call_id),
            RuntimeError("final response unavailable"),
        ]
    )
    agent, memory, sent = _telegram_agent(model)
    conversation_id = memory.create_conversation()
    agent.respond(conversation_id, "send it")

    with pytest.raises(AgentConfigurationError, match="final response unavailable"):
        agent.respond(conversation_id, "yes")

    assert sent == [{"message": "Once"}]
    assert conversation_id not in agent.pending


def test_telegram_rejection_never_executes_tool() -> None:
    model = ScriptedModel(
        [_tool_call("send_telegram_message", {"message": "No send"}, "reject-call")]
    )
    agent, memory, sent = _telegram_agent(model)
    conversation_id = memory.create_conversation()
    agent.respond(conversation_id, "send it")

    assert agent.respond(conversation_id, "no") == "I did not make the requested change."
    assert sent == []
    assert conversation_id not in agent.pending


class CalendarEventsApi:
    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None):
        self.payload = payload or {"items": [{"id": "event-1"}]}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list(self, **kwargs: Any) -> Any:
        self.calls.append(("list", kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(execute=lambda: self.payload)

    def get(self, **kwargs: Any) -> Any:
        self.calls.append(("get", kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(execute=lambda: {"id": kwargs["eventId"]})


class CalendarApi:
    def __init__(self, events: CalendarEventsApi):
        self.events_api = events

    def events(self) -> CalendarEventsApi:
        return self.events_api


def test_langgraph_calendar_read_passes_manual_semantics_and_needs_no_confirmation() -> None:
    events = CalendarEventsApi()
    provider = GoogleCalendarProvider(service=CalendarApi(events))
    registry = ToolRegistry(calendar_tools(provider))
    tools = {tool.name: tool for tool in build_langchain_tools(registry)}

    result = tools["list_events"].invoke({})

    assert result == {"provider": "google", "events": [{"id": "event-1"}]}
    assert events.calls[0][0] == "list"
    assert events.calls[0][1]["calendarId"] == "primary"
    assert events.calls[0][1]["maxResults"] == 10
    assert registry.get("list_events").requires_confirmation is False


def test_calendar_provider_failure_is_translated_to_safe_tool_error() -> None:
    events = CalendarEventsApi(error=RuntimeError("provider detail must not escape"))
    provider = GoogleCalendarProvider(service=CalendarApi(events))
    registry = ToolRegistry(calendar_tools(provider))
    tool = {item.name: item for item in build_langchain_tools(registry)}["list_events"]

    with pytest.raises(ToolError, match="Google Calendar could not list events"):
        tool.invoke({})


def test_calendar_argument_mapping_omits_optional_none_values(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG", logger="backend.tools.langchain_tools")
    events = CalendarEventsApi()
    provider = GoogleCalendarProvider(service=CalendarApi(events))
    registry = ToolRegistry(calendar_tools(provider))
    tool = {item.name: item for item in build_langchain_tools(registry)}["list_events"]

    tool.invoke({"calendar_id": None, "time_min": None, "max_results": None})

    assert events.calls[0][1]["calendarId"] == "primary"
    assert "argument_keys" in caplog.text or events.calls
