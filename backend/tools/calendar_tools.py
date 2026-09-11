from __future__ import annotations

from typing import Any

from backend.services.calendar_service import CalendarProvider, normalize_create_event
from backend.tools.base import ToolDefinition


def calendar_tools(provider: CalendarProvider) -> list[ToolDefinition]:
    return [
        ToolDefinition("list_events", "List upcoming calendar events.", {"type": "object", "properties": {"calendar_id": {"type": "string"}, "time_min": {"type": "string"}, "max_results": {"type": "integer"}}, "additionalProperties": False}, provider.list_events),
        ToolDefinition("get_event", "Read one calendar event.", {"type": "object", "properties": {"calendar_id": {"type": "string"}, "event_id": {"type": "string"}}, "required": ["event_id"], "additionalProperties": False}, provider.get_event),
        ToolDefinition(
            "create_event",
            "Create a Google Calendar event after explicit user confirmation. "
            "Use an ISO-8601 start, optional ISO-8601 end, and an IANA timezone. "
            "If no duration is specified, omit end and use the default one-hour duration.",
            {
                "type": "object",
                "properties": {
                    "calendar_id": {"type": "string", "description": "Calendar ID; use primary unless specified."},
                    "summary": {"type": "string", "description": "Event title."},
                    "start": {"type": "string", "description": "ISO-8601 start date/time."},
                    "end": {"type": "string", "description": "Optional ISO-8601 end date/time; defaults to one hour after start."},
                    "timezone": {"type": "string", "description": "Optional IANA timezone such as Africa/Cairo or UTC. Defaults to APP_TIMEZONE."},
                    "description": {"type": "string"},
                },
                "required": ["summary", "start"],
                "additionalProperties": False,
            },
            provider.create_event,
            False,
            True,
            normalize_create_event,
        ),
        ToolDefinition("update_event", "Update a calendar event after user confirmation.", {"type": "object", "properties": {"calendar_id": {"type": "string"}, "event_id": {"type": "string"}, "event": {"type": "object"}}, "required": ["event_id", "event"], "additionalProperties": False}, provider.update_event, False, True),
        ToolDefinition("delete_event", "Delete a calendar event after user confirmation.", {"type": "object", "properties": {"calendar_id": {"type": "string"}, "event_id": {"type": "string"}}, "required": ["event_id"], "additionalProperties": False}, provider.delete_event, False, True),
    ]
