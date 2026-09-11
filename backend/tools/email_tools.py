from __future__ import annotations

from typing import Any

from backend.services.gmail_service import GmailService
from backend.tools.base import ToolDefinition


def gmail_tools(service: GmailService | None = None) -> list[ToolDefinition]:
    gmail = service or GmailService()

    def search(arguments: dict[str, Any]) -> dict[str, Any]:
        return gmail.search(arguments.get("query", ""), arguments.get("max_results", 10))

    def recent(arguments: dict[str, Any]) -> dict[str, Any]:
        return gmail.recent(arguments.get("max_results", 10))

    def read(arguments: dict[str, Any]) -> dict[str, Any]:
        return gmail.get(arguments.get("message_id", ""))

    return [
        ToolDefinition(
            "search_emails", "Search Gmail using safe Gmail query operators.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"], "additionalProperties": False},
            search,
        ),
        ToolDefinition(
            "get_recent_emails", "Get a small set of recent Gmail messages.",
            {"type": "object", "properties": {"max_results": {"type": "integer", "minimum": 1, "maximum": 20}}, "additionalProperties": False},
            recent,
        ),
        ToolDefinition(
            "get_email", "Read one Gmail message by ID.",
            {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"], "additionalProperties": False},
            read,
        ),
        ToolDefinition(
            "summarize_emails", "Retrieve matching email metadata and snippets for summarization.",
            {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"], "additionalProperties": False},
            search,
        ),
        ToolDefinition(
            "identify_important_emails", "Find unread or important Gmail messages.",
            {"type": "object", "properties": {"max_results": {"type": "integer", "minimum": 1, "maximum": 20}}, "additionalProperties": False},
            lambda args: gmail.search("is:important OR is:unread", args.get("max_results", 10)),
        ),
    ]
