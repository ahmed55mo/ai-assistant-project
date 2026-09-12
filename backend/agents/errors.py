from __future__ import annotations


class AgentError(RuntimeError):
    """Base error for the LangGraph foundation."""


class AgentConfigurationError(AgentError):
    """Raised when the graph agent cannot be configured."""
