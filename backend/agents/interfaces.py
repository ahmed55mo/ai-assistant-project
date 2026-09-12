from __future__ import annotations

from typing import Protocol


class AgentRuntime(Protocol):
    """Runtime abstraction shared by manual and graph-backed agents."""

    def respond(self, conversation_id: str, user_message: str) -> str:
        ...
