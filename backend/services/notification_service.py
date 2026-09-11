from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import resend
from dotenv import load_dotenv
from resend.exceptions import ResendError

from backend.tools.base import ToolError

load_dotenv()


class NotificationService:
    def send_email(self, arguments: dict[str, Any]) -> dict[str, Any]:
        recipient = arguments.get("to")
        subject = arguments.get("subject")
        body = arguments.get("body")
        if not all(isinstance(value, str) and value.strip() for value in (recipient, subject, body)):
            raise ToolError("Email requires a recipient, subject, and body.")
        if not _is_email(recipient):
            raise ToolError("Email recipient is invalid.")

        api_key = os.getenv("RESEND_API_KEY", "").strip()
        sender = os.getenv("RESEND_FROM_EMAIL", "").strip()
        if not api_key or not sender:
            raise ToolError(
                "Email notifications are not configured. Set RESEND_API_KEY and "
                "RESEND_FROM_EMAIL; no email was sent."
            )
        if not _is_email(sender):
            raise ToolError("RESEND_FROM_EMAIL is invalid; no email was sent.")

        resend.api_key = api_key
        try:
            response = resend.Emails.send(
                {
                    "from": sender,
                    "to": [recipient.strip()],
                    "subject": subject.strip(),
                    "text": body,
                }
            )
        except ResendError as exc:
            raise ToolError(_resend_error_message(exc)) from exc

        email_id = _response_value(response, "id")
        if not email_id:
            raise ToolError("Resend did not confirm the email.")
        return {"sent": True, "provider": "resend", "email_id": email_id, "to": recipient.strip()}

    def send_telegram_message(self, arguments: dict[str, Any]) -> dict[str, Any]:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        message = arguments.get("message")
        if not token or not chat_id:
            raise ToolError(
                "Telegram notifications are not configured. Set TELEGRAM_BOT_TOKEN "
                "and TELEGRAM_CHAT_ID; no message was sent."
            )
        if not isinstance(message, str) or not message.strip() or len(message) > 4096:
            raise ToolError("Telegram message must be between 1 and 4096 characters.")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = _post_json(
            url,
            {"chat_id": chat_id, "text": message.strip()},
            {},
            "Telegram",
        )
        if not payload.get("ok"):
            raise ToolError("Telegram did not confirm the message.")
        message_id = payload.get("result", {}).get("message_id")
        return {"sent": True, "provider": "telegram", "chat_id": str(chat_id), "message_id": message_id}


def _is_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _resend_error_message(error: Exception) -> str:
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        status_code = None

    if status_code in {401, 403}:
        return "Resend authentication failed; no email was sent."
    if status_code == 429:
        return "Resend is rate-limiting requests; no email was sent."
    if (
        isinstance(error, (URLError, TimeoutError, OSError))
        or "timeout" in type(error).__name__.lower()
        or getattr(error, "error_type", "") == "HttpClientError"
    ):
        return "Resend could not be reached; no email was sent."
    return "Resend rejected the email request; no email was sent."


def _post_json(
    url: str,
    payload: dict[str, Any],
    extra_headers: dict[str, str],
    provider_name: str,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **extra_headers},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ToolError(f"{provider_name} rejected the notification request.") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"{provider_name} could not be reached.") from exc
    if not isinstance(data, dict):
        raise ToolError(f"{provider_name} returned an invalid response.")
    return data
