import os
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from groq import APIError, Groq

load_dotenv()

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
SYSTEM_PROMPT = """You are a professional AI assistant.
Be helpful, concise, and informative. Maintain a professional tone and use
conversation context when it is provided. Clearly state when you do not know
something. Never claim to have performed an action that you did not perform."""
logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM service is not configured correctly."""


class LLMService:
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is not configured. Add it to the environment before using /api/chat."
            )

        self.model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()
        if not self.model_name:
            self.model_name = DEFAULT_MODEL_NAME
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        self.client = Groq(api_key=api_key, timeout=timeout)

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> Any:
        conversation_messages = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n"
                    f"Current local date and time: {datetime.now().astimezone().isoformat()}.\n"
                    "Use this date when interpreting relative dates such as tomorrow."
                ),
            },
            *messages,
        ]
        options: dict[str, Any] = {
            "model": self.model_name,
            "messages": conversation_messages,
        }
        if tools:
            options["tools"] = list(tools)
            options["tool_choice"] = "auto"
        try:
            return self.client.chat.completions.create(**options)
        except APIError as exc:
            _log_groq_error(exc, self.model_name)
            raise

    def generate_response(self, messages: Sequence[dict[str, Any]]) -> str:
        completion = self.complete(messages)
        response = completion.choices[0].message.content
        if not response:
            raise RuntimeError("The Groq API returned an empty response.")
        return response.strip()


def _log_groq_error(error: APIError, model_name: str) -> None:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    body = getattr(error, "body", None)
    provider_message = _safe_provider_message(body)
    error_type = type(error).__name__
    category = _error_category(error)
    logger.error(
        "Groq request failed: category=%s exception_type=%s status_code=%s "
        "model=%s provider_message=%s",
        category,
        error_type,
        status_code if status_code is not None else "unavailable",
        model_name,
        provider_message,
    )


def _error_category(error: APIError) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code == 400:
        return "bad_request"
    if status_code in {401, 403}:
        return "authentication_or_permission"
    if status_code == 429:
        return "rate_limit"
    if status_code is not None and status_code >= 500:
        return "provider_server_error"
    if "timeout" in type(error).__name__.lower():
        return "timeout"
    if "connection" in type(error).__name__.lower():
        return "connection"
    return "api_error"


def _safe_provider_message(body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            value = error.get("message") or error.get("type") or error.get("code")
            if value:
                return _truncate(str(value))
        if isinstance(error, str):
            return _truncate(error)
        message = body.get("message")
        if message:
            return _truncate(str(message))
    if body is not None and not isinstance(body, (bytes, bytearray)):
        return _truncate(str(body))
    return "no provider error body"


def _truncate(value: str, limit: int = 500) -> str:
    value = " ".join(value.split())
    return value[:limit] + ("..." if len(value) > limit else "")
