from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from backend.agents.confirmation import PendingConfirmation


class AgentState(TypedDict):
    conversation_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    pending_confirmation: PendingConfirmation | None
    last_error: str | None
