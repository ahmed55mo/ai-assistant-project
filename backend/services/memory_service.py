from __future__ import annotations

from threading import Lock
from uuid import uuid4
from typing import Any


Message = dict[str, Any]


class ConversationNotFoundError(KeyError):
    """Raised when a conversation ID does not exist in memory."""


class ConversationMemory:
    """Simple process-local conversation storage for the current application run."""

    def __init__(self, max_messages: int = 20) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        self.max_messages = max_messages
        self._conversations: dict[str, list[Message]] = {}
        self._lock = Lock()

    def create_conversation(self) -> str:
        conversation_id = str(uuid4())
        with self._lock:
            self._conversations[conversation_id] = []
        return conversation_id

    def has_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            return conversation_id in self._conversations

    def get_history(self, conversation_id: str) -> list[Message]:
        with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(conversation_id)
            return [message.copy() for message in self._conversations[conversation_id]]

    def get_llm_history(self, conversation_id: str) -> list[Message]:
        return [
            message
            for message in self.get_history(conversation_id)
            if message.get("llm_visible", True)
        ]

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str | None = None,
        **metadata: Any,
    ) -> None:
        with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(conversation_id)
            message: Message = {"role": role}
            if content is not None:
                message["content"] = content
            message.update(metadata)
            self._conversations[conversation_id].append(message)
            self._conversations[conversation_id] = self._conversations[conversation_id][
                -self.max_messages :
            ]

    def clear_conversation(self, conversation_id: str) -> None:
        with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(conversation_id)
            del self._conversations[conversation_id]
