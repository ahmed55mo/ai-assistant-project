import json
import unittest
from copy import deepcopy

from backend.services.calendar_service import (
    GoogleCalendarProvider,
    MockCalendarProvider,
    normalize_create_event,
)
from backend.services.conversation_service import ConversationService
from backend.services.memory_service import ConversationMemory
from backend.tools.base import ConfirmationRequired, ToolDefinition, ToolError, ToolRegistry
from backend.tools.calculator import CALCULATOR_TOOL, calculate
from backend.tools.calendar_tools import calendar_tools
from backend.tools.notification_tools import notification_tools


class ToolCall:
    def __init__(self, name, arguments, call_id="call-1"):
        self.id = call_id
        self.function = type("Function", (), {"name": name, "arguments": arguments})()


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


class ToolTests(unittest.TestCase):
    def test_calculator_and_invalid_expression(self):
        self.assertEqual(calculate({"expression": "(15 + 5) * 3"})["result"], 60)
        self.assertEqual(calculate({"expression": "20 / 4"})["result"], 5)
        self.assertEqual(calculate({"expression": "20% of 450"})["result"], 90)
        with self.assertRaises(ToolError):
            calculate({"expression": "__import__('os').getcwd()"})

    def test_registry_and_multiple_calculator_calls(self):
        registry = ToolRegistry([CALCULATOR_TOOL])
        first = ToolCall("calculator", json.dumps({"expression": "25 * 48"}), "one")
        second = ToolCall("calculator", json.dumps({"expression": "125 + 75"}), "two")
        llm = ScriptedLLM([
            FakeMessage(tool_calls=[first, second]),
            FakeMessage("Both calculations are complete."),
        ])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, registry)
        response = service.respond(memory.create_conversation(), "calculate both")
        self.assertEqual(response, "Both calculations are complete.")
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(len([m for m in memory.get_history(next(iter(memory._conversations))) if m["role"] == "tool"]), 2)
        tool_messages = [message for message in llm.calls[1] if message["role"] == "tool"]
        self.assertEqual(
            [(message["tool_call_id"], message["name"]) for message in tool_messages],
            [("one", "calculator"), ("two", "calculator")],
        )
        assistant_message = next(
            message for message in llm.calls[1] if message["role"] == "assistant"
        )
        self.assertEqual(
            [call["id"] for call in assistant_message["tool_calls"]],
            ["one", "two"],
        )

    def test_single_tool_result_preserves_call_id_for_next_llm_request(self):
        call = ToolCall("calculator", json.dumps({"expression": "2 + 2"}), "calc-42")
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Done.")])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, ToolRegistry([CALCULATOR_TOOL]))

        self.assertEqual(service.respond(memory.create_conversation(), "calculate"), "Done.")
        tool_message = next(message for message in llm.calls[1] if message["role"] == "tool")
        self.assertEqual(tool_message["tool_call_id"], "calc-42")

    def test_tool_failure_preserves_call_id_for_next_llm_request(self):
        failing = ToolDefinition(
            "broken",
            "broken",
            {"type": "object"},
            lambda arguments: (_ for _ in ()).throw(ToolError("provider failed")),
        )
        call = ToolCall("broken", "{}", "failure-7")
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Recovered.")])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, ToolRegistry([failing]))

        self.assertEqual(service.respond(memory.create_conversation(), "try"), "Recovered.")
        tool_message = next(message for message in llm.calls[1] if message["role"] == "tool")
        self.assertEqual(tool_message["tool_call_id"], "failure-7")
        self.assertIn("provider failed", tool_message["content"])

    def test_confirmation_tool_result_preserves_call_id_in_history(self):
        sent = []

        class FakeNotificationService:
            def send_telegram_message(self, arguments):
                sent.append(arguments)
                return {"sent": True, "provider": "telegram", "message_id": 1}

            def send_email(self, arguments):
                sent.append(arguments)
                return {"sent": True, "provider": "resend", "email_id": "email-1"}

        call = ToolCall(
            "send_telegram_message",
            json.dumps({"message": "hello"}),
            "telegram-9",
        )
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Next answer.")])
        memory = ConversationMemory()
        conversation_id = memory.create_conversation()
        service = ConversationService(
            memory, llm, ToolRegistry(notification_tools(FakeNotificationService()))
        )

        service.respond(conversation_id, "send telegram")
        service.respond(conversation_id, "yes")
        history = memory.get_history(conversation_id)
        tool_message = next(message for message in history if message["role"] == "tool")
        self.assertEqual(tool_message["tool_call_id"], "telegram-9")
        self.assertEqual(tool_message["name"], "send_telegram_message")
        self.assertEqual(sent, [{"message": "hello"}])

    def test_resend_confirmation_is_valid_for_a_later_llm_request(self):
        sent = []

        class FakeNotificationService:
            def send_email(self, arguments):
                sent.append(arguments)
                return {"sent": True, "provider": "resend", "email_id": "email-1"}

            def send_telegram_message(self, arguments):
                raise AssertionError("unexpected tool")

        call = ToolCall(
            "send_email",
            json.dumps(
                {
                    "to": "user@example.com",
                    "subject": "Meeting",
                    "body": "Confirmed.",
                }
            ),
            "resend-4",
        )
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Follow-up.")])
        memory = ConversationMemory()
        conversation_id = memory.create_conversation()
        service = ConversationService(
            memory, llm, ToolRegistry(notification_tools(FakeNotificationService()))
        )

        service.respond(conversation_id, "send email")
        service.respond(conversation_id, "yes")
        self.assertEqual(service.respond(conversation_id, "what next?"), "Follow-up.")
        follow_up_messages = llm.calls[1]
        tool_message = next(
            message for message in follow_up_messages if message["role"] == "tool"
        )
        self.assertEqual(tool_message["tool_call_id"], "resend-4")
        self.assertEqual(sent[0]["to"], "user@example.com")

    def test_multi_step_tool_calling_preserves_each_assistant_and_result_pair(self):
        first = ToolCall("calculator", json.dumps({"expression": "3 * 3"}), "step-1")
        second = ToolCall("calculator", json.dumps({"expression": "9 + 1"}), "step-2")
        llm = ScriptedLLM(
            [
                FakeMessage(tool_calls=[first]),
                FakeMessage(tool_calls=[second]),
                FakeMessage("All steps complete."),
            ]
        )
        memory = ConversationMemory()
        service = ConversationService(memory, llm, ToolRegistry([CALCULATOR_TOOL]))

        self.assertEqual(
            service.respond(memory.create_conversation(), "do two calculations"),
            "All steps complete.",
        )
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in llm.calls[1]
                if message["role"] == "tool"
            ],
            ["step-1"],
        )
        self.assertEqual(
            [
                message["tool_call_id"]
                for message in llm.calls[2]
                if message["role"] == "tool"
            ],
            ["step-1", "step-2"],
        )

    def test_unknown_tool_and_max_iterations(self):
        registry = ToolRegistry([CALCULATOR_TOOL])
        with self.assertRaises(ToolError):
            registry.execute("missing", {})
        loop = ToolCall("calculator", json.dumps({"expression": "1 + 1"}))
        llm = ScriptedLLM([FakeMessage(tool_calls=[loop]), FakeMessage(tool_calls=[loop])])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, registry, max_iterations=2)
        with self.assertRaises(ToolError):
            service.respond(memory.create_conversation(), "loop")

    def test_calendar_mock_and_notification_definitions_require_confirmation(self):
        registry = ToolRegistry(calendar_tools(MockCalendarProvider()) + notification_tools())
        self.assertTrue(registry.get("create_event").requires_confirmation)
        with self.assertRaises(Exception):
            registry.execute("create_event", {"event": {"summary": "Test"}})
        self.assertTrue(registry.get("send_email").requires_confirmation)

    def test_calendar_event_normalization_and_default_duration(self):
        arguments = normalize_create_event(
            {
                "summary": "Grad project meeting",
                "start": "2026-09-11T15:00:00",
                "timezone": "Africa/Cairo",
            }
        )
        self.assertEqual(arguments["event"]["summary"], "Grad project meeting")
        self.assertEqual(arguments["event"]["start"]["timeZone"], "Africa/Cairo")
        self.assertIn("16:00:00", arguments["event"]["end"]["dateTime"])

    def test_calendar_creation_asks_for_confirmation_before_execution(self):
        class RecordingProvider(MockCalendarProvider):
            def __init__(self):
                self.created = False

            def create_event(self, arguments):
                self.created = True
                return {"provider": "mock", "confirmed": True, "summary": arguments["event"]["summary"]}

        provider = RecordingProvider()
        registry = ToolRegistry(calendar_tools(provider))
        tool_call = ToolCall(
            "create_event",
            json.dumps(
                {
                    "summary": "Grad project meeting",
                    "start": "2026-09-11T15:00:00",
                    "timezone": "Africa/Cairo",
                }
            ),
        )
        llm = ScriptedLLM([FakeMessage(tool_calls=[tool_call])])
        memory = ConversationMemory()
        conversation_id = memory.create_conversation()
        response = ConversationService(memory, llm, registry).respond(
            conversation_id, "Schedule a grad project meeting tomorrow at 3 PM"
        )
        self.assertIn("Should I proceed", response)
        self.assertFalse(provider.created)

    def test_confirmed_calendar_creation_executes_provider(self):
        class RecordingProvider(MockCalendarProvider):
            def __init__(self):
                self.created = False

            def create_event(self, arguments):
                self.created = True
                return {
                    "provider": "google",
                    "confirmed": True,
                    "summary": arguments["event"]["summary"],
                    "event_id": "event-123",
                }

        provider = RecordingProvider()
        registry = ToolRegistry(calendar_tools(provider))
        tool_call = ToolCall(
            "create_event",
            json.dumps(
                {
                    "summary": "Grad project meeting",
                    "start": "2026-09-11T15:00:00",
                    "timezone": "Africa/Cairo",
                }
            ),
        )
        llm = ScriptedLLM([FakeMessage(tool_calls=[tool_call])])
        memory = ConversationMemory()
        conversation_id = memory.create_conversation()
        service = ConversationService(memory, llm, registry)
        service.respond(conversation_id, "Schedule the meeting")
        response = service.respond(conversation_id, "yes")
        self.assertTrue(provider.created)
        self.assertIn("created the calendar event", response)

    def test_notification_confirmation_and_rejection_are_conversation_scoped(self):
        sent = []

        class FakeNotificationService:
            def send_telegram_message(self, arguments):
                sent.append(arguments)
                return {"sent": True, "provider": "telegram", "message_id": 1}

            def send_email(self, arguments):
                sent.append(arguments)
                return {"sent": True, "provider": "resend", "email_id": "email-1"}

        registry = ToolRegistry(notification_tools(FakeNotificationService()))
        call = ToolCall(
            "send_telegram_message",
            json.dumps({"message": "Assistant is working"}),
        )
        llm = ScriptedLLM([FakeMessage(tool_calls=[call]), FakeMessage("Unrelated answer")])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, registry)
        conversation_id = memory.create_conversation()

        prompt = service.respond(conversation_id, "Send a Telegram message")
        self.assertIn("Should I send it?", prompt)
        self.assertEqual(sent, [])

        cancelled = service.respond(conversation_id, "No")
        self.assertIn("did not make", cancelled)
        self.assertEqual(sent, [])

        unrelated = service.respond(conversation_id, "Yes")
        self.assertEqual(unrelated, "Unrelated answer")
        self.assertEqual(sent, [])

    def test_notification_confirmation_executes_only_after_yes(self):
        sent = []

        class FakeNotificationService:
            def send_email(self, arguments):
                sent.append(arguments)
                return {"sent": True, "provider": "resend", "email_id": "email-1"}

            def send_telegram_message(self, arguments):
                raise AssertionError("unexpected tool")

        registry = ToolRegistry(notification_tools(FakeNotificationService()))
        call = ToolCall(
            "send_email",
            json.dumps(
                {
                    "to": "user@example.com",
                    "subject": "Meeting",
                    "body": "The project meeting is confirmed.",
                }
            ),
        )
        llm = ScriptedLLM([FakeMessage(tool_calls=[call])])
        memory = ConversationMemory()
        service = ConversationService(memory, llm, registry)
        conversation_id = memory.create_conversation()
        service.respond(conversation_id, "Send the email")
        response = service.respond(conversation_id, "Yes, send it.")
        self.assertIn("send_email operation completed successfully", response)
        self.assertEqual(len(sent), 1)

    def test_multiple_tool_domains_and_tool_failure_do_not_crash_conversation(self):
        class FakeGmail:
            def search(self, arguments):
                return {"emails": [{"subject": "Meeting"}]}

        class FakeCalendar:
            def list_events(self, arguments):
                return {"events": [{"summary": "Grad project meeting"}]}

        registry = ToolRegistry(
            [
                ToolDefinition(
                    "search_emails", "search", {"type": "object"}, FakeGmail().search
                ),
                ToolDefinition(
                    "list_events", "list", {"type": "object"}, FakeCalendar().list_events
                ),
            ]
        )
        email_call = ToolCall(
            "search_emails", json.dumps({"query": "latest"}), "email-call"
        )
        calendar_call = ToolCall("list_events", json.dumps({}), "calendar-call")
        llm = ScriptedLLM(
            [
                FakeMessage(tool_calls=[email_call, calendar_call]),
                FakeMessage("You have a meeting tomorrow."),
            ]
        )
        memory = ConversationMemory()
        service = ConversationService(memory, llm, registry)
        response = service.respond(memory.create_conversation(), "Check email and calendar")
        self.assertEqual(response, "You have a meeting tomorrow.")
        self.assertEqual(len(llm.calls), 2)

        failing_registry = ToolRegistry(
            [
                ToolDefinition(
                    "broken_tool",
                    "broken",
                    {"type": "object"},
                    lambda arguments: (_ for _ in ()).throw(ToolError("provider failed")),
                )
            ]
        )
        failing_call = ToolCall("broken_tool", json.dumps({}))
        failing_llm = ScriptedLLM(
            [FakeMessage(tool_calls=[failing_call]), FakeMessage("I could not access it.")]
        )
        failing_memory = ConversationMemory()
        failing_service = ConversationService(failing_memory, failing_llm, failing_registry)
        self.assertEqual(
            failing_service.respond(failing_memory.create_conversation(), "Try it"),
            "I could not access it.",
        )

    def test_malformed_calendar_event_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            normalize_create_event(
                {"summary": "Missing start", "timezone": "Africa/Cairo"}
            )

    def test_google_calendar_create_event_uses_confirmed_api_response(self):
        class FakeRequest:
            def execute(self):
                return {
                    "id": "event-123",
                    "htmlLink": "https://calendar.google.com/event-123",
                    "summary": "Grad project meeting",
                    "start": {"dateTime": "2026-09-11T15:00:00+03:00"},
                    "end": {"dateTime": "2026-09-11T16:00:00+03:00"},
                }

        class FakeEvents:
            def insert(self, **kwargs):
                self.arguments = kwargs
                return FakeRequest()

        class FakeService:
            def __init__(self):
                self.events_api = FakeEvents()

            def events(self):
                return self.events_api

        provider = GoogleCalendarProvider(service=FakeService())
        result = provider.create_event(
            normalize_create_event(
                {
                    "summary": "Grad project meeting",
                    "start": "2026-09-11T15:00:00+03:00",
                    "end": "2026-09-11T16:00:00+03:00",
                    "timezone": "Africa/Cairo",
                }
            )
        )
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["event_id"], "event-123")

    def test_google_calendar_read_and_mutation_methods(self):
        class FakeRequest:
            def __init__(self, value=None):
                self.value = value

            def execute(self):
                return self.value

        class FakeEvents:
            def list(self, **kwargs):
                return FakeRequest({"items": [{"id": "event-1"}]})

            def get(self, **kwargs):
                return FakeRequest({"id": kwargs["eventId"]})

            def update(self, **kwargs):
                return FakeRequest({"id": kwargs["eventId"], "summary": "Updated"})

            def delete(self, **kwargs):
                return FakeRequest(None)

        class FakeService:
            def events(self):
                return FakeEvents()

        provider = GoogleCalendarProvider(service=FakeService())
        self.assertEqual(len(provider.list_events({})["events"]), 1)
        self.assertEqual(provider.get_event({"event_id": "event-1"})["event"]["id"], "event-1")
        self.assertTrue(
            provider.update_event(
                {"event_id": "event-1", "event": {"summary": "Updated"}}
            )["confirmed"]
        )
        self.assertTrue(provider.delete_event({"event_id": "event-1"})["confirmed"])

    def test_google_calendar_api_failure_is_a_tool_error(self):
        class BrokenEvents:
            def list(self, **kwargs):
                raise RuntimeError("calendar unavailable")

        class BrokenService:
            def events(self):
                return BrokenEvents()

        with self.assertRaises(ToolError):
            GoogleCalendarProvider(service=BrokenService()).list_events({})

    def test_confirmation_gate_requires_explicit_approval(self):
        sent = []
        tool = ToolDefinition(
            "side_effect",
            "test side effect",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda arguments: sent.append(arguments) or {"sent": True},
            read_only=False,
            requires_confirmation=True,
        )
        registry = ToolRegistry([tool])
        with self.assertRaises(ConfirmationRequired):
            registry.execute("side_effect", {})
        self.assertEqual(sent, [])
        self.assertEqual(registry.execute("side_effect", {}, confirmed=True), {"sent": True})
        self.assertEqual(sent, [{}])


if __name__ == "__main__":
    unittest.main()
