import json
import os
import unittest
from unittest.mock import patch

from resend.exceptions import ResendError

from backend.services.notification_service import NotificationService
from backend.tools.base import ToolError
from backend.tools.notification_tools import notification_tools


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class NotificationServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = NotificationService()
        self.environment = {
            "RESEND_API_KEY": os.environ.get("RESEND_API_KEY"),
            "RESEND_FROM_EMAIL": os.environ.get("RESEND_FROM_EMAIL"),
            "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN"),
            "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID"),
        }

    def tearDown(self):
        for key, value in self.environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_resend_success(self):
        os.environ["RESEND_API_KEY"] = "test-key"
        os.environ["RESEND_FROM_EMAIL"] = "assistant@example.com"
        with patch(
            "backend.services.notification_service.resend.Emails.send",
            return_value={"id": "email-123"},
        ) as request:
            result = self.service.send_email(
                {"to": "user@example.com", "subject": "Hello", "body": "Test body"}
            )
        self.assertEqual(result["email_id"], "email-123")
        self.assertTrue(result["sent"])
        request.assert_called_once_with(
            {
                "from": "assistant@example.com",
                "to": ["user@example.com"],
                "subject": "Hello",
                "text": "Test body",
            }
        )

    def test_resend_missing_api_key_sender_and_invalid_recipient(self):
        os.environ.pop("RESEND_API_KEY", None)
        os.environ["RESEND_FROM_EMAIL"] = "assistant@example.com"
        with self.assertRaises(ToolError):
            self.service.send_email(
                {"to": "user@example.com", "subject": "Hello", "body": "Test"}
            )
        os.environ["RESEND_API_KEY"] = "test-key"
        os.environ.pop("RESEND_FROM_EMAIL", None)
        with self.assertRaises(ToolError):
            self.service.send_email(
                {"to": "user@example.com", "subject": "Hello", "body": "Test"}
            )
        os.environ["RESEND_FROM_EMAIL"] = "assistant@example.com"
        with self.assertRaises(ToolError):
            self.service.send_email(
                {"to": "not-an-email", "subject": "Hello", "body": "Test"}
            )
        with self.assertRaises(ToolError):
            self.service.send_email(
                {"to": "user@example.com", "subject": "", "body": "Test"}
            )

    def test_resend_api_network_auth_and_rate_limit_failures(self):
        os.environ["RESEND_API_KEY"] = "test-key"
        os.environ["RESEND_FROM_EMAIL"] = "assistant@example.com"
        with patch(
            "backend.services.notification_service.resend.Emails.send",
            side_effect=ResendError(400, "validation_error", "provider rejected", "check request"),
        ):
            with self.assertRaises(ToolError):
                self.service.send_email(
                    {"to": "user@example.com", "subject": "Hello", "body": "Test"}
                )
        with patch(
            "backend.services.notification_service.resend.Emails.send",
            side_effect=ResendError(500, "HttpClientError", "network failure", "retry"),
        ):
            with self.assertRaises(ToolError):
                self.service.send_email(
                    {"to": "user@example.com", "subject": "Hello", "body": "Test"}
                )
        for status_code, expected in (
            (401, "authentication"),
            (429, "rate-limiting"),
        ):
            error = ResendError(status_code, "provider_error", "secret provider details", "retry")
            with patch(
                "backend.services.notification_service.resend.Emails.send",
                side_effect=error,
            ):
                with self.assertRaisesRegex(ToolError, expected):
                    self.service.send_email(
                        {"to": "user@example.com", "subject": "Hello", "body": "Test"}
                    )

    def test_telegram_success(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
        os.environ["TELEGRAM_CHAT_ID"] = "123"
        with patch(
            "backend.services.notification_service.urlopen",
            return_value=FakeResponse({"ok": True, "result": {"message_id": 42}}),
        ) as request:
            result = self.service.send_telegram_message({"message": "Assistant is working"})
        self.assertEqual(result["message_id"], 42)
        self.assertTrue(result["sent"])
        self.assertEqual(request.call_args.args[0].full_url, "https://api.telegram.org/bottest-token/sendMessage")
        self.assertEqual(json.loads(request.call_args.args[0].data), {"chat_id": "123", "text": "Assistant is working"})

    def test_telegram_missing_configuration_api_failure_and_malformed_message(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ["TELEGRAM_CHAT_ID"] = "123"
        with self.assertRaises(ToolError):
            self.service.send_telegram_message({"message": "Hello"})
        os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        with self.assertRaises(ToolError):
            self.service.send_telegram_message({"message": "Hello"})
        os.environ["TELEGRAM_CHAT_ID"] = "123"
        with self.assertRaises(ToolError):
            self.service.send_telegram_message({"message": ""})
        with patch(
            "backend.services.notification_service.urlopen",
            return_value=FakeResponse({"ok": False}),
        ):
            with self.assertRaises(ToolError):
                self.service.send_telegram_message({"message": "Hello"})

    def test_telegram_repeated_requests_use_configured_chat_id(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "test-token"
        os.environ["TELEGRAM_CHAT_ID"] = "123"
        with patch(
            "backend.services.notification_service.urlopen",
            return_value=FakeResponse({"ok": True, "result": {"message_id": 42}}),
        ) as request:
            self.service.send_telegram_message({"message": "First"})
            self.service.send_telegram_message({"message": "Second"})
        self.assertEqual(request.call_count, 2)
        self.assertEqual(json.loads(request.call_args_list[0].args[0].data)["chat_id"], "123")
        self.assertEqual(json.loads(request.call_args_list[1].args[0].data)["chat_id"], "123")

    def test_notification_registry_names_and_confirmation(self):
        tools = {tool.name: tool for tool in notification_tools(self.service)}
        self.assertEqual(set(tools), {"send_email", "send_telegram_message"})
        self.assertTrue(all(tool.requires_confirmation for tool in tools.values()))
        telegram_schema = tools["send_telegram_message"].parameters
        self.assertEqual(telegram_schema["required"], ["message"])
        self.assertEqual(set(telegram_schema["properties"]), {"message"})


if __name__ == "__main__":
    unittest.main()
