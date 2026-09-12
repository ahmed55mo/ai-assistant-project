from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.agents.langgraph_agent import LangGraphAgent
from backend.agents.runtime import (
    AgentRuntimeFactory,
    RuntimeConfigurationError,
    configured_engine,
)
from backend.services.conversation_service import ConversationService
from backend.services.memory_service import ConversationMemory


class FakeRuntime:
    def __init__(self, memory: ConversationMemory, response: str = "ok") -> None:
        self.memory = memory
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def respond(self, conversation_id: str, user_message: str) -> str:
        self.calls.append((conversation_id, user_message))
        self.memory.add_message(conversation_id, "user", user_message)
        self.memory.add_message(conversation_id, "assistant", self.response)
        return self.response


def test_missing_engine_defaults_to_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_ENGINE", raising=False)
    assert configured_engine() == "manual"


def test_factory_selects_manual_and_caches_per_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ENGINE", "manual")
    manual_factory = Mock(side_effect=lambda memory, llm, registry: FakeRuntime(memory))
    factory = AgentRuntimeFactory(manual_factory=manual_factory)
    memory = ConversationMemory()

    first = factory.get(memory)
    second = factory.get(memory)

    assert isinstance(first, FakeRuntime)
    assert first is second
    manual_factory.assert_called_once()


def test_factory_selects_langgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ENGINE", "langgraph")
    langgraph_factory = Mock(side_effect=lambda memory, registry: FakeRuntime(memory))
    factory = AgentRuntimeFactory(langgraph_factory=langgraph_factory)

    runtime = factory.get(ConversationMemory())

    assert isinstance(runtime, FakeRuntime)
    langgraph_factory.assert_called_once()


def test_invalid_engine_raises_safe_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_ENGINE", "unsupported")
    with pytest.raises(RuntimeConfigurationError, match="Unsupported AGENT_ENGINE"):
        configured_engine()


def test_api_contract_is_shared_by_manual_and_langgraph_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for engine in ("manual", "langgraph"):
        memory = ConversationMemory()
        fake_runtime = FakeRuntime(memory, response=f"{engine} response")
        factory = AgentRuntimeFactory(
            manual_factory=lambda memory, llm, registry: fake_runtime,
            langgraph_factory=lambda memory, registry: fake_runtime,
        )
        monkeypatch.setenv("AGENT_ENGINE", engine)
        monkeypatch.setattr(main, "conversation_memory", memory)
        monkeypatch.setattr(main, "conversation_service", None)
        monkeypatch.setattr(main, "runtime_factory", factory)

        response = TestClient(main.app).post(
            "/api/chat", json={"message": "hello"}
        )

        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"conversation_id", "response"}
        assert payload["response"] == f"{engine} response"
        assert fake_runtime.calls == [(payload["conversation_id"], "hello")]


def test_langgraph_selection_can_construct_the_real_runtime_without_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_ENGINE", "langgraph")
    fake_agent = Mock(spec=LangGraphAgent)
    factory = AgentRuntimeFactory(langgraph_factory=lambda memory, registry: fake_agent)

    assert factory.get(ConversationMemory()) is fake_agent

