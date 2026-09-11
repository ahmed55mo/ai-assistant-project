import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from backend.services.calendar_service import (
    GoogleCalendarProvider,
    MockCalendarProvider,
    normalize_create_event,
)
from backend.services.conversation_service import ConversationService
from backend.services.gmail_service import GmailService
from backend.services.memory_service import ConversationMemory
from backend.services.notification_service import NotificationService
from backend.tools.base import ConfirmationRequired, ToolDefinition, ToolError, ToolRegistry
from backend.tools.calculator import CALCULATOR_TOOL, calculate
from backend.tools.calendar_tools import calendar_tools
from backend.tools.email_tools import gmail_tools
from backend.tools.notification_tools import notification_tools


class ToolCall:
    def __init__(self, name, arguments, call_id):
        self.id = call_id
        self.function = type(
            "Function", (), {"name": name, "arguments": arguments}
        )()


class FakeMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeCompletion:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class ScriptedLLM:
    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append(deepcopy(messages))
        return FakeCompletion(self.messages.pop(0))


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeGmailMessages:
    def __init__(self, messages=None, failure=None):
        self._messages = messages or []
        self.failure = failure

    def list(self, **kwargs):
        if self.failure:
            raise self.failure
        return FakeRequest({"messages": [{"id": item["id"]} for item in self._messages]})

    def get(self, **kwargs):
        if self.failure:
            raise self.failure
        message = next(item for item in self._messages if item["id"] == kwargs["id"])
        return FakeRequest(message)


class FakeGmailApi:
    def __init__(self, messages=None, failure=None):
        self.messages_api = FakeGmailMessages(messages, failure)

    def users(self):
        return type("Users", (), {"messages": lambda _: self.messages_api})()


class RecordingCalendarProvider(MockCalendarProvider):
    def __init__(self):
        self.calls = []

    def list_events(self, arguments):
        self.calls.append(("list", arguments))
        return {"provider": "mock", "events": [{"id": "event-1"}]}

    def get_event(self, arguments):
        self.calls.append(("get", arguments))
        return {"provider": "mock", "event": {"id": arguments["event_id"]}}

    def create_event(self, arguments):
        self.calls.append(("create", arguments))
        return {"provider": "mock", "confirmed": True, "summary": "Created"}

    def update_event(self, arguments):
        self.calls.append(("update", arguments))
        return {"provider": "mock", "confirmed": True}

    def delete_event(self, arguments):
        self.calls.append(("delete", arguments))
        return {"provider": "mock", "confirmed": True}


def tool_messages(messages):
    return [message for message in messages if message["role"] == "tool"]


class Phase3VerificationTests(unittest.TestCase):
    def test_calculator_validation_execution_and_result_return_to_llm(self):
        self.assertEqual(calculate({"expression": "(8 + 2) * 4"})["result"], 40)
        with self.assertRaises(ToolError):
            calculate({"expression": "__import__('os').getcwd()"})

        call = ToolCall("calculator", json.dumps({"expression": "6 * 7"}), "calc-1")
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("42")])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, ToolRegistry([CALCULATOR_TOOL]))
        self.assertEqual(service.respond(memory.create_conversation(), "calculate"), "42")
        result = tool_messages(llm.calls[1])[0]
        self.assertEqual(result["tool_call_id"], "calc-1")
        self.assertEqual(json.loads(result["content"])["result"], 42)

    def test_gmail_search_get_empty_failure_and_tool_protocol(self):
        message = {
            "id": "abc",
            "threadId": "thread",
            "snippet": "snippet",
            "labelIds": ["IMPORTANT"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "Ahmed <ahmed@example.com>"},
                    {"name": "Subject", "value": "Hello"},
                ]
            },
        }
        service = GmailService(service=FakeGmailApi([message]))
        self.assertEqual(service.search("is:important", 5)["count"], 1)
        self.assertEqual(service.get("abc")["subject"], "Hello")
        self.assertEqual(GmailService(service=FakeGmailApi([])).search("none")["count"], 0)
        with self.assertRaises(ToolError):
            GmailService(service=FakeGmailApi(failure=RuntimeError("provider failure"))).search("x")

        call = ToolCall("search_emails", json.dumps({"query": "latest"}), "gmail-1")
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Mail checked")])
        memory = ConversationMemory()
        registry = ToolRegistry(gmail_tools(service))
        self.assertEqual(
            ConversationService(memory, llm, registry).respond(
                memory.create_conversation(), "check mail"
            ),
            "Mail checked",
        )
        self.assertEqual(tool_messages(llm.calls[1])[0]["tool_call_id"], "gmail-1")

    def test_google_calendar_read_mutations_confirmation_and_failure(self):
        provider = RecordingCalendarProvider()
        registry = ToolRegistry(calendar_tools(provider))
        self.assertFalse(registry.get("list_events").requires_confirmation)
        self.assertFalse(registry.get("get_event").requires_confirmation)
        for name in ("create_event", "update_event", "delete_event"):
            self.assertTrue(registry.get(name).requires_confirmation)

        self.assertEqual(registry.execute("list_events", {})["events"][0]["id"], "event-1")
        self.assertEqual(registry.execute("get_event", {"event_id": "event-1"})["event"]["id"], "event-1")
        event_arguments = {
            "summary": "Planning",
            "start": "2026-09-11T15:00:00",
            "timezone": "UTC",
        }
        with self.assertRaises(ConfirmationRequired):
            registry.execute("create_event", event_arguments)
        self.assertTrue(
            registry.execute("create_event", event_arguments, confirmed=True)["confirmed"]
        )
        self.assertTrue(
            registry.execute(
                "update_event", {"event_id": "event-1", "event": {"summary": "Updated"}}, confirmed=True
            )["confirmed"]
        )
        self.assertTrue(registry.execute("delete_event", {"event_id": "event-1"}, confirmed=True)["confirmed"])

        class BrokenEvents:
            def list(self, **kwargs):
                raise RuntimeError("calendar unavailable")

        class BrokenService:
            def events(self):
                return BrokenEvents()

        with self.assertRaises(ToolError):
            GoogleCalendarProvider(service=BrokenService()).list_events({})

    def test_telegram_configuration_schema_confirmation_and_repeated_send(self):
        environment = {
            "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN"),
            "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID"),
        }
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "bot-secret"
            os.environ["TELEGRAM_CHAT_ID"] = "configured-chat"
            service = NotificationService()
            with patch(
                "backend.services.notification_service.urlopen",
                side_effect=[
                    type("Response", (), {
                        "__enter__": lambda self: self,
                        "__exit__": lambda self, *args: False,
                        "read": lambda self: b'{"ok": true, "result": {"message_id": 1}}',
                    })(),
                    type("Response", (), {
                        "__enter__": lambda self: self,
                        "__exit__": lambda self, *args: False,
                        "read": lambda self: b'{"ok": true, "result": {"message_id": 2}}',
                    })(),
                    type("Response", (), {
                        "__enter__": lambda self: self,
                        "__exit__": lambda self, *args: False,
                        "read": lambda self: b'{"ok": true, "result": {"message_id": 3}}',
                    })(),
                ],
            ) as request:
                self.assertTrue(service.send_telegram_message({"message": "one"})["sent"])
                self.assertTrue(service.send_telegram_message({"message": "two"})["sent"])
            self.assertEqual(request.call_count, 2)
            for call in request.call_args_list:
                self.assertIn(b"configured-chat", call.args[0].data)
                self.assertNotIn(b"bot-secret", call.args[0].data)

            tool = next(tool for tool in notification_tools(service) if tool.name == "send_telegram_message")
            self.assertEqual(tool.parameters["required"], ["message"])
            self.assertEqual(set(tool.parameters["properties"]), {"message"})
            serialized = json.dumps(tool.as_groq_tool())
            self.assertNotIn("bot-secret", serialized)
            self.assertNotIn("configured-chat", serialized)

            call = ToolCall("send_telegram_message", json.dumps({"message": "confirm"}), "telegram-1")
            llm = ScriptedLLM([FakeMessage(tool_calls=[call])])
            memory = ConversationMemory()
            conversation_id = memory.create_conversation()
            conversation = ConversationService(memory, llm, ToolRegistry(notification_tools(service)))
            with patch(
                "backend.services.notification_service.urlopen",
                return_value=type("Response", (), {
                    "__enter__": lambda self: self,
                    "__exit__": lambda self, *args: False,
                    "read": lambda self: b'{"ok": true, "result": {"message_id": 3}}',
                })(),
            ):
                self.assertIn(
                    "Should I send it?",
                    conversation.respond(conversation_id, "send telegram"),
                )
                self.assertIn(
                    "completed successfully",
                    conversation.respond(conversation_id, "yes"),
                )
        finally:
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_telegram_missing_configuration_and_api_failure(self):
        environment = {
            "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN"),
            "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID"),
        }
        try:
            os.environ["TELEGRAM_BOT_TOKEN"] = "token"
            os.environ.pop("TELEGRAM_CHAT_ID", None)
            with self.assertRaisesRegex(ToolError, "not configured"):
                NotificationService().send_telegram_message({"message": "hello"})
            os.environ["TELEGRAM_CHAT_ID"] = "chat"
            with patch(
                "backend.services.notification_service.urlopen",
                return_value=type("Response", (), {
                    "__enter__": lambda self: self,
                    "__exit__": lambda self, *args: False,
                    "read": lambda self: b'{"ok": false}',
                })(),
            ):
                with self.assertRaises(ToolError):
                    NotificationService().send_telegram_message({"message": "hello"})
        finally:
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_resend_success_configuration_failure_and_confirmation(self):
        environment = {
            "RESEND_API_KEY": os.environ.get("RESEND_API_KEY"),
            "RESEND_FROM_EMAIL": os.environ.get("RESEND_FROM_EMAIL"),
        }
        try:
            os.environ["RESEND_API_KEY"] = "resend-secret"
            os.environ["RESEND_FROM_EMAIL"] = "assistant@example.com"
            service = NotificationService()
            with patch(
                "backend.services.notification_service.resend.Emails.send",
                return_value={"id": "email-1"},
            ) as send:
                self.assertTrue(
                    service.send_email(
                        {"to": "user@example.com", "subject": "Hi", "body": "Body"}
                    )["sent"]
                )
            self.assertNotIn("resend-secret", json.dumps(notification_tools(service)[0].as_groq_tool()))
            send.assert_called_once()

            os.environ.pop("RESEND_API_KEY")
            with self.assertRaises(ToolError):
                service.send_email({"to": "user@example.com", "subject": "Hi", "body": "Body"})
            os.environ["RESEND_API_KEY"] = "resend-secret"
            os.environ.pop("RESEND_FROM_EMAIL")
            with self.assertRaises(ToolError):
                service.send_email({"to": "user@example.com", "subject": "Hi", "body": "Body"})
            os.environ["RESEND_FROM_EMAIL"] = "assistant@example.com"
            with self.assertRaises(ToolError):
                service.send_email({"to": "invalid", "subject": "Hi", "body": "Body"})

            call = ToolCall(
                "send_email",
                json.dumps({"to": "user@example.com", "subject": "Hi", "body": "Body"}),
                "resend-1",
            )
            llm = ScriptedLLM([FakeMessage(tool_calls=[call])])
            memory = ConversationMemory()
            conversation_id = memory.create_conversation()
            conversation = ConversationService(memory, llm, ToolRegistry(notification_tools(service)))
            with patch(
                "backend.services.notification_service.resend.Emails.send",
                return_value={"id": "email-2"},
            ):
                self.assertIn("Should I send it?", conversation.respond(conversation_id, "send email"))
                self.assertIn("completed successfully", conversation.respond(conversation_id, "yes"))
        finally:
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_protocol_multiple_tools_cross_domain_and_failures(self):
        gmail = ToolDefinition(
            "gmail_read", "read", {"type": "object"}, lambda args: {"emails": ["one"]}
        )
        calendar = ToolDefinition(
            "calendar_read", "read", {"type": "object"}, lambda args: {"events": ["one"]}
        )
        telegram = ToolDefinition(
            "telegram_send", "send", {"type": "object"},
            lambda args: {"sent": True, "provider": "telegram"},
        )
        resend = ToolDefinition(
            "resend_send", "send", {"type": "object"},
            lambda args: {"sent": True, "provider": "resend"},
        )
        broken = ToolDefinition(
            "broken", "broken", {"type": "object"},
            lambda args: (_ for _ in ()).throw(ToolError("provider failed")),
        )
        calls = [
            ToolCall("gmail_read", "{}", "gmail-1"),
            ToolCall("calendar_read", "{}", "calendar-1"),
            ToolCall("telegram_send", "{}", "telegram-1"),
            ToolCall("resend_send", "{}", "resend-1"),
            ToolCall("broken", "{}", "broken-1"),
        ]
        llm = ScriptedLLM(
            [
                FakeMessage(tool_calls=calls[:2]),
                FakeMessage(tool_calls=calls[2:4]),
                FakeMessage(tool_calls=[calls[4]]),
                FakeMessage("Complete"),
            ]
        )
        memory = ConversationMemory()
        registry = ToolRegistry([gmail, calendar, telegram, resend, broken])
        service = ConversationService(memory, llm, registry)
        self.assertEqual(service.respond(memory.create_conversation(), "run workflow"), "Complete")
        self.assertEqual(
            [item["tool_call_id"] for item in tool_messages(llm.calls[1])],
            ["gmail-1", "calendar-1"],
        )
        self.assertEqual(
            [item["tool_call_id"] for item in tool_messages(llm.calls[2])],
            ["gmail-1", "calendar-1", "telegram-1", "resend-1"],
        )
        final_results = tool_messages(llm.calls[3])
        self.assertEqual(final_results[-1]["tool_call_id"], "broken-1")
        self.assertIn("provider failed", final_results[-1]["content"])

    def test_actual_gmail_calendar_telegram_and_resend_sequences(self):
        gmail_message = {
            "id": "abc",
            "threadId": "thread",
            "snippet": "snippet",
            "labelIds": [],
            "payload": {"headers": [{"name": "Subject", "value": "Hello"}]},
        }
        gmail = GmailService(service=FakeGmailApi([gmail_message]))
        calendar = RecordingCalendarProvider()
        sent = []

        class FakeNotificationService:
            def send_telegram_message(self, arguments):
                sent.append(("telegram", arguments))
                return {"sent": True, "provider": "telegram"}

            def send_email(self, arguments):
                sent.append(("resend", arguments))
                return {"sent": True, "provider": "resend"}

        registry = ToolRegistry(
            gmail_tools(gmail)
            + calendar_tools(calendar)
            + notification_tools(FakeNotificationService())
        )
        gmail_call = ToolCall("search_emails", json.dumps({"query": "latest"}), "gmail-x")
        calendar_call = ToolCall("list_events", "{}", "calendar-x")
        telegram_call = ToolCall(
            "send_telegram_message", json.dumps({"message": "Hello"}), "telegram-x"
        )
        resend_call = ToolCall(
            "send_email",
            json.dumps(
                {"to": "user@example.com", "subject": "Hi", "body": "Body"}
            ),
            "resend-x",
        )
        llm = ScriptedLLM(
            [
                FakeMessage(tool_calls=[gmail_call, calendar_call]),
                FakeMessage(tool_calls=[telegram_call]),
                FakeMessage(tool_calls=[resend_call]),
            ]
        )
        memory = ConversationMemory()
        conversation_id = memory.create_conversation()
        conversation = ConversationService(memory, llm, registry)

        self.assertIn(
            "Should I send it?",
            conversation.respond(conversation_id, "check mail and then send Telegram"),
        )
        self.assertEqual(
            [item["tool_call_id"] for item in tool_messages(llm.calls[1])],
            ["gmail-x", "calendar-x"],
        )
        self.assertIn("completed successfully", conversation.respond(conversation_id, "yes"))
        self.assertIn("Should I send it?", conversation.respond(conversation_id, "send an email"))
        self.assertIn("completed successfully", conversation.respond(conversation_id, "yes"))
        self.assertEqual([provider for provider, _ in sent], ["telegram", "resend"])

    def test_invalid_tool_arguments_become_safe_tool_results(self):
        call = ToolCall("calculator", "{not-json", "invalid-1")
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Handled safely")])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, ToolRegistry([CALCULATOR_TOOL]))
        self.assertEqual(
            service.respond(memory.create_conversation(), "calculate"),
            "Handled safely",
        )
        result = tool_messages(llm.calls[1])[0]
        self.assertEqual(result["tool_call_id"], "invalid-1")
        self.assertIn("Expecting", result["content"])

    def test_confirmation_accept_reject_pending_and_no_duplicate_execution(self):
        executions = []
        side_effect = ToolDefinition(
            "side_effect", "side effect", {"type": "object"},
            lambda args: executions.append(args) or {"ok": True},
            read_only=False, requires_confirmation=True,
        )
        call = ToolCall("side_effect", "{}", "side-1")
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Next")])
        memory = ConversationMemory()
        conversation_id = memory.create_conversation()
        service = ConversationService(memory, llm, ToolRegistry([side_effect]))
        self.assertIn("Should I proceed", service.respond(conversation_id, "do it"))
        self.assertEqual(service.respond(conversation_id, "no"), "I did not make the requested change.")
        self.assertEqual(executions, [])
        self.assertEqual(service.respond(conversation_id, "next"), "Next")
        self.assertEqual(len(llm.calls), 2)

        call2 = ToolCall("side_effect", "{}", "side-2")
        llm2 = ScriptedLLM([FakeMessage(tool_calls=[call2])])
        memory2 = ConversationMemory()
        cid2 = memory2.create_conversation()
        service2 = ConversationService(memory2, llm2, ToolRegistry([side_effect]))
        service2.respond(cid2, "do it")
        self.assertIn("completed successfully", service2.respond(cid2, "yes"))
        self.assertEqual(len(executions), 1)
        cancellation = next(
            message
            for message in memory.get_history(conversation_id)
            if message.get("role") == "tool"
        )
        self.assertIn("Operation cancelled", cancellation["content"])

    def test_conversation_memory_isolation_valid_history_and_clear(self):
        memory = ConversationMemory()
        first = memory.create_conversation()
        second = memory.create_conversation()
        memory.add_message(first, "user", "one")
        memory.add_message(first, "assistant", "", tool_calls=[{"id": "call-1"}])
        memory.add_message(first, "tool", "{}", tool_call_id="call-1", name="calculator")
        memory.add_message(second, "user", "two")
        self.assertEqual(memory.get_history(first)[-1]["tool_call_id"], "call-1")
        self.assertEqual(memory.get_history(first)[-2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(memory.get_history(second), [{"role": "user", "content": "two"}])
        memory.clear_conversation(first)
        self.assertFalse(memory.has_conversation(first))
        self.assertEqual(memory.get_history(second), [{"role": "user", "content": "two"}])


if __name__ == "__main__":
    unittest.main()
