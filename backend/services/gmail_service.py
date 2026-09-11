from __future__ import annotations

import base64
import re
from email.utils import parseaddr
from typing import Any

from googleapiclient.discovery import build

from backend.auth.google_oauth import GoogleOAuth, GoogleOAuthError
from backend.tools.base import ToolError


class GmailService:
    def __init__(self, oauth: GoogleOAuth | None = None, service: Any | None = None) -> None:
        self.oauth = oauth or GoogleOAuth()
        self._service = service

    @property
    def service(self) -> Any:
        if self._service is None:
            try:
                self._service = build("gmail", "v1", credentials=self.oauth.credentials(), cache_discovery=False)
            except GoogleOAuthError:
                raise
            except Exception as exc:
                raise ToolError("Gmail could not be initialized.") from exc
        return self._service

    def search(self, query: str, max_results: int = 10) -> dict[str, Any]:
        if not isinstance(query, str) or len(query) > 200:
            raise ToolError("Gmail search query is invalid.")
        max_results = max(1, min(int(max_results), 20))
        try:
            result = self.service.users().messages().list(
                userId="me", q=query.strip(), maxResults=max_results
            ).execute()
            messages = result.get("messages", [])
            items = [self._get_metadata(item["id"]) for item in messages]
            return {"query": query, "count": len(items), "emails": items}
        except GoogleOAuthError:
            raise
        except Exception as exc:
            raise ToolError("Gmail search failed.") from exc

    def recent(self, max_results: int = 10) -> dict[str, Any]:
        return self.search("", max_results)

    def get(self, message_id: str) -> dict[str, Any]:
        if not isinstance(message_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", message_id):
            raise ToolError("Invalid Gmail message ID.")
        try:
            message = self.service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            return self._parse_message(message)
        except GoogleOAuthError:
            raise
        except Exception as exc:
            raise ToolError("Gmail message could not be retrieved.") from exc

    def _get_metadata(self, message_id: str) -> dict[str, Any]:
        message = self.service.users().messages().get(
            userId="me", id=message_id, format="metadata",
            metadataHeaders=["From", "To", "Subject", "Date"],
        ).execute()
        headers = {
            item["name"].lower(): item.get("value", "")
            for item in message.get("payload", {}).get("headers", [])
        }
        return {
            "id": message_id,
            "thread_id": message.get("threadId"),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": message.get("snippet", "")[:500],
            "labels": message.get("labelIds", []),
        }

    def _parse_message(self, message: dict[str, Any]) -> dict[str, Any]:
        metadata = self._get_metadata(message["id"])
        body = _extract_text(message.get("payload", {}))[:4000]
        sender_name, sender_email = parseaddr(metadata["from"])
        metadata.update({"sender_name": sender_name, "sender_email": sender_email, "body": body})
        return metadata


def _extract_text(payload: dict[str, Any]) -> str:
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = _extract_text(part)
        if text:
            return text
    return ""
