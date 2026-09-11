from __future__ import annotations

import json
import os
import re
from typing import Any

from backend.services.llm_service import LLMService
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ConfirmationRequired, ToolError, ToolRegistry


class ConversationService:
    def __init__(
        self,
        memory: ConversationMemory,
        llm: LLMService,
        registry: ToolRegistry,
        max_iterations: int | None = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.registry = registry
        self.max_iterations = max_iterations or int(os.getenv("MAX_TOOL_ITERATIONS", "5"))
        self.pending: dict[str, tuple[str, str, dict[str, Any]]] = {}

    def respond(self, conversation_id: str, user_message: str) -> str:
        pending = self.pending.get(conversation_id)
        if pending and _is_confirmation(user_message):
            call_id, name, arguments = pending
            del self.pending[conversation_id]
            self.memory.add_message(
                conversation_id, "user", user_message, llm_visible=False
            )
            try:
                result = self.registry.execute(name, arguments, confirmed=True)
                self._store_tool_result(conversation_id, call_id, name, result)
                response = _confirmation_success_message(name, result)
            except ToolError as exc:
                result = {"error": str(exc)}
                self._store_tool_result(conversation_id, call_id, name, result)
                response = f"I could not complete {name}: {exc}"
            self.memory.add_message(conversation_id, "assistant", response)
            return response

        if pending and _is_rejection(user_message):
            call_id, name, _ = self.pending.pop(conversation_id)
            self.memory.add_message(
                conversation_id, "user", user_message, llm_visible=False
            )
            self._store_tool_result(
                conversation_id,
                call_id,
                name,
                {"error": "Operation cancelled by the user."},
            )
            response = "I did not make the requested change."
            self.memory.add_message(conversation_id, "assistant", response)
            return response

        self.memory.add_message(conversation_id, "user", user_message)
        messages = self.memory.get_llm_history(conversation_id)
        for _ in range(self.max_iterations):
            completion = self.llm.complete(messages, self.registry.definitions())
            assistant = completion.choices[0].message
            tool_calls = getattr(assistant, "tool_calls", None) or []
            if not tool_calls:
                response = (assistant.content or "").strip()
                if not response:
                    raise ToolError("The LLM returned an empty response.")
                self.memory.add_message(conversation_id, "assistant", response)
                return response

            assistant_message = _assistant_tool_call_message(assistant.content, tool_calls)
            messages.append(assistant_message)
            self.memory.add_message(
                conversation_id,
                "assistant",
                assistant_message["content"],
                tool_calls=assistant_message["tool_calls"],
            )
            for call in tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ToolError("Tool arguments must be an object.")
                    result = self.registry.execute(call.function.name, arguments)
                except ConfirmationRequired:
                    self.pending[conversation_id] = (
                        call.id,
                        call.function.name,
                        arguments,
                    )
                    prompt = _confirmation_prompt(call.function.name, arguments)
                    self.memory.add_message(
                        conversation_id, "assistant", prompt, llm_visible=False
                    )
                    return prompt
                except (json.JSONDecodeError, ToolError) as exc:
                    result = {"error": str(exc)}
                tool_message = _tool_result_message(call.id, call.function.name, result)
                messages.append(tool_message)
                self._store_tool_result(conversation_id, call.id, call.function.name, result)
        raise ToolError("The maximum number of tool iterations was reached.")

    def _store_tool_result(
        self, conversation_id: str, call_id: str, name: str, result: dict[str, Any]
    ) -> None:
        self.memory.add_message(
            conversation_id,
            "tool",
            json.dumps(_memory_safe_result(result)),
            tool_call_id=call_id,
            name=name,
        )


def _is_confirmation(message: str) -> bool:
    normalized = message.strip().lower()
    return normalized in {"yes", "y", "confirm", "proceed", "send it", "do it"} or bool(
        re.fullmatch(r"(yes|y)(?:[,\s!]+(?:send it|proceed|do it))?[.!]?", normalized)
    )


def _assistant_tool_call_message(content: str | None, tool_calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in tool_calls
        ],
    }


def _tool_result_message(call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(result),
    }


def _is_rejection(message: str) -> bool:
    return message.strip().lower() in {"no", "n", "cancel", "stop", "don't", "do not"}


def _confirmation_success_message(name: str, result: dict[str, Any]) -> str:
    if name == "create_event":
        summary = result.get("summary", "calendar event")
        link = result.get("html_link")
        return (
            f"Confirmed. I created the calendar event '{summary}'."
            + (f" Open it here: {link}" if link else "")
        )
    return f"Confirmed. The {name} operation completed successfully."


def _confirmation_prompt(name: str, arguments: dict[str, Any]) -> str:
    if name == "send_email":
        return (
            "I’m ready to send this email:\n"
            f"To: {arguments.get('to', '')}\n"
            f"Subject: {arguments.get('subject', '')}\n"
            f"Body: {arguments.get('body', '')}\n"
            "Should I send it?"
        )
    if name == "send_telegram_message":
        return (
            "I’m ready to send this Telegram message:\n"
            f"Message: {arguments.get('message', '')}\n"
            "Should I send it?"
        )
    return f"I can run `{name}` with the requested details. Should I proceed?"


def _memory_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    safe = dict(result)
    if "emails" in safe and isinstance(safe["emails"], list):
        safe["emails"] = [
            (
                {key: value for key, value in item.items() if key != "body"}
                if isinstance(item, dict)
                else item
            )
            for item in safe["emails"]
        ]
    if "body" in safe:
        safe["body"] = "[email body omitted from conversation memory]"
    return safe
