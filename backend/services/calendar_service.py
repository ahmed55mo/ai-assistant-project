from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from googleapiclient.discovery import build

from backend.auth.google_oauth import GoogleOAuth
from backend.tools.base import ToolError


class CalendarEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calendar_id: str = "primary"
    summary: str = Field(min_length=1, max_length=200)
    start: datetime
    end: datetime | None = None
    timezone: str = Field(default_factory=lambda: os.getenv("APP_TIMEZONE", "UTC"))
    description: str | None = Field(default=None, max_length=5000)

    @field_validator("summary", "timezone")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value

    def normalized(self) -> dict[str, Any]:
        timezone_info = ZoneInfo(self.timezone)
        start = self.start if self.start.tzinfo else self.start.replace(tzinfo=timezone_info)
        end = self.end
        if end is None:
            end = start + timedelta(hours=1)
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone_info)
        if end <= start:
            raise ValueError("end must be after start")
        return {
            "calendar_id": self.calendar_id,
            "event": {
                "summary": self.summary,
                "description": self.description or "",
                "start": {"dateTime": start.isoformat(), "timeZone": self.timezone},
                "end": {"dateTime": end.isoformat(), "timeZone": self.timezone},
            },
        }


def normalize_create_event(arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        return CalendarEventInput.model_validate(arguments).normalized()
    except (ValidationError, ValueError, TypeError) as exc:
        raise ToolError(
            "Invalid calendar event. Provide summary, an ISO-8601 start, "
            "optional ISO-8601 end, and an IANA timezone. "
            "If end is omitted, a one-hour duration is used."
        ) from exc


class CalendarProvider:
    def list_events(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def create_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def update_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def delete_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class MockCalendarProvider(CalendarProvider):
    def list_events(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"provider": "mock", "events": []}

    def get_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"provider": "mock", "event": None}

    def create_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolError("Calendar is using the mock provider; no event was created.")

    def update_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolError("Calendar is using the mock provider; no event was updated.")

    def delete_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ToolError("Calendar is using the mock provider; no event was deleted.")


class GoogleCalendarProvider(CalendarProvider):
    def __init__(self, oauth: GoogleOAuth | None = None, service: Any | None = None) -> None:
        self.oauth = oauth or GoogleOAuth()
        self._service = service

    @property
    def service(self) -> Any:
        if self._service is None:
            self._service = build("calendar", "v3", credentials=self.oauth.credentials(), cache_discovery=False)
        return self._service

    def list_events(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = self.service.events().list(
                calendarId=arguments.get("calendar_id", "primary"),
                timeMin=arguments.get("time_min", now),
                maxResults=min(int(arguments.get("max_results", 10)), 20),
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            return {"provider": "google", "events": result.get("items", [])}
        except Exception as exc:
            raise ToolError("Google Calendar could not list events.") from exc

    def get_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            event = self.service.events().get(
                calendarId=arguments.get("calendar_id", "primary"), eventId=arguments["event_id"]
            ).execute()
            return {"provider": "google", "event": event}
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("Calendar event_id is required.") from exc
        except Exception as exc:
            raise ToolError("Google Calendar could not retrieve the event.") from exc

    def create_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            event = self.service.events().insert(
                calendarId=arguments["calendar_id"], body=arguments["event"]
            ).execute()
            return {
                "provider": "google",
                "confirmed": True,
                "event_id": event.get("id"),
                "html_link": event.get("htmlLink"),
                "summary": event.get("summary", arguments["event"]["summary"]),
                "start": event.get("start", arguments["event"]["start"]),
                "end": event.get("end", arguments["event"]["end"]),
            }
        except Exception as exc:
            raise ToolError(
                "Google Calendar could not create the event. Check authentication, "
                "calendar permissions, and the event details."
            ) from exc

    def update_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            event = self.service.events().update(
                calendarId=arguments.get("calendar_id", "primary"),
                eventId=arguments["event_id"], body=arguments["event"],
            ).execute()
            return {"provider": "google", "event": event, "confirmed": True}
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("Calendar update requires event_id and event.") from exc
        except Exception as exc:
            raise ToolError("Google Calendar could not update the event.") from exc

    def delete_event(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            self.service.events().delete(
                calendarId=arguments.get("calendar_id", "primary"), eventId=arguments["event_id"]
            ).execute()
            return {"provider": "google", "event_id": arguments["event_id"], "confirmed": True}
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("Calendar deletion requires event_id.") from exc
        except Exception as exc:
            raise ToolError("Google Calendar could not delete the event.") from exc


def calendar_provider() -> CalendarProvider:
    return GoogleCalendarProvider() if os.getenv("CALENDAR_PROVIDER", "mock") == "google" else MockCalendarProvider()
