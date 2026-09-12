from __future__ import annotations

import os
from typing import Callable

from backend.agents.errors import AgentConfigurationError
from backend.agents.interfaces import AgentRuntime
from backend.agents.langgraph_agent import LangGraphAgent
from backend.services.conversation_service import ConversationService
from backend.services.llm_service import LLMService
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ToolRegistry
from backend.tools.registry import build_registry
from backend.rag.retrieval import RetrievalService


SUPPORTED_ENGINES = frozenset({"manual", "langgraph"})


class RuntimeConfigurationError(AgentConfigurationError):
    """Raised when AGENT_ENGINE is missing or unsupported."""


def configured_engine() -> str:
    engine = os.getenv("AGENT_ENGINE", "manual").strip().lower()
    if engine not in SUPPORTED_ENGINES:
        supported = ", ".join(sorted(SUPPORTED_ENGINES))
        raise RuntimeConfigurationError(
            f"Unsupported AGENT_ENGINE '{engine}'. Expected one of: {supported}."
        )
    return engine


class AgentRuntimeFactory:
    """Create one runtime per engine and memory instance."""

    def __init__(
        self,
        manual_factory: Callable[..., AgentRuntime] = ConversationService,
        langgraph_factory: Callable[..., AgentRuntime] = LangGraphAgent,
        retrieval_service: RetrievalService | None = None,
    ) -> None:
        self._manual_factory = manual_factory
        self._langgraph_factory = langgraph_factory
        self._retrieval_service = retrieval_service
        self._engine: str | None = None
        self._memory: ConversationMemory | None = None
        self._runtime: AgentRuntime | None = None

    def get(
        self,
        memory: ConversationMemory,
        *,
        legacy_manual: AgentRuntime | None = None,
    ) -> AgentRuntime:
        # Preserve the existing application's explicit manual-service injection
        # used by callers and tests. Normal application startup leaves this None,
        # so AGENT_ENGINE remains authoritative for runtime selection.
        if legacy_manual is not None:
            return legacy_manual
        engine = configured_engine()
        if self._runtime is not None and self._engine == engine and self._memory is memory:
            return self._runtime

        registry = build_registry(self._retrieval_service)
        if engine == "manual":
            runtime = self._manual_factory(memory, LLMService(), registry)
        else:
            runtime = self._langgraph_factory(memory, registry)
        self._engine = engine
        self._memory = memory
        self._runtime = runtime
        return runtime

    def reset(self) -> None:
        self._engine = None
        self._memory = None
        self._runtime = None


__all__ = [
    "AgentRuntimeFactory",
    "RuntimeConfigurationError",
    "SUPPORTED_ENGINES",
    "configured_engine",
]
