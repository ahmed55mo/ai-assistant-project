import unittest

from backend.services.gmail_service import GmailService
from backend.tools.base import ToolError
from backend.tools.email_tools import gmail_tools


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeMessages:
    def list(self, **kwargs):
        return FakeRequest({"messages": [{"id": "abc"}]})

    def get(self, **kwargs):
        return FakeRequest(
            {
                "id": "abc",
                "threadId": "thread",
                "snippet": "A short snippet",
                "labelIds": ["IMPORTANT"],
                "payload": {
                    "headers": [
                        {"name": "From", "value": "Ahmed <ahmed@example.com>"},
                        {"name": "Subject", "value": "Hello"},
                    ]
                },
            }
        )


class FakeUsers:
    def messages(self):
        return FakeMessages()


class FakeGmailApi:
    def users(self):
        return FakeUsers()


class GmailServiceTests(unittest.TestCase):
    def test_search_and_read_use_metadata_without_credentials(self):
        service = GmailService(service=FakeGmailApi())
        result = service.search("from:ahmed@example.com", 5)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["emails"][0]["subject"], "Hello")
        email = service.get("abc")
        self.assertEqual(email["sender_email"], "ahmed@example.com")

    def test_gmail_search_and_summary_tools_use_structured_results(self):
        tools = {tool.name: tool for tool in gmail_tools(GmailService(service=FakeGmailApi()))}
        summary = tools["summarize_emails"].execute(
            {"query": "is:important", "max_results": 5}
        )
        important = tools["identify_important_emails"].execute({"max_results": 5})
        self.assertEqual(summary["count"], 1)
        self.assertEqual(important["emails"][0]["subject"], "Hello")

    def test_invalid_message_and_api_failure_are_safe_errors(self):
        service = GmailService(service=FakeGmailApi())
        with self.assertRaises(ToolError):
            service.get("../invalid")
        failing = GmailService(service=type("BrokenApi", (), {
            "users": lambda self: (_ for _ in ()).throw(RuntimeError("provider failure"))
        })())
        with self.assertRaises(ToolError):
            failing.search("is:unread")


if __name__ == "__main__":
    unittest.main()
