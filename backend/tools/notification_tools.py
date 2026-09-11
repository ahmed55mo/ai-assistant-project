from __future__ import annotations

from backend.services.notification_service import NotificationService
from backend.tools.base import ToolDefinition


def notification_tools(service: NotificationService | None = None) -> list[ToolDefinition]:
    service = service or NotificationService()
    return [
        ToolDefinition(
            "send_email",
            "Send an email through Resend after explicit user confirmation.",
            {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
            service.send_email,
            False,
            True,
        ),
        ToolDefinition(
            "send_telegram_message",
            "Send a Telegram text message after explicit user confirmation.",
            {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "minLength": 1, "maxLength": 4096},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
            service.send_telegram_message,
            False,
            True,
        ),
    ]
