from __future__ import annotations

from typing import Any, TypedDict
import re


class PendingConfirmation(TypedDict):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


def confirmation_prompt(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "send_email":
        return (
            "I’m ready to send this email:\n"
            f"To: {arguments.get('recipient', arguments.get('to', ''))}\n"
            f"Subject: {arguments.get('subject', '')}\n"
            f"Body: {arguments.get('body', '')}\n"
            "Should I send it?"
        )
    if tool_name == "send_telegram_message":
        return (
            "I’m ready to send this Telegram message:\n"
            f"Message: {arguments.get('message', '')}\n"
            "Should I send it?"
        )
    return f"I can run `{tool_name}` with the requested details. Should I proceed?"


def is_confirmation(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    return normalized in {
        "yes",
        "y",
        "confirm",
        "confirmed",
        "approved",
        "approve",
        "go ahead",
        "send it",
        "do it",
        "proceed",
    }


def is_rejection(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    return normalized in {
        "no",
        "n",
        "cancel",
        "cancel it",
        "reject",
        "rejected",
        "don't do it",
        "do not",
        "do not do it",
        "stop",
    }
