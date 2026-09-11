import unittest

from fastapi.testclient import TestClient

from backend import main
from backend.services.memory_service import ConversationMemory


class FakeMessage:
    def __init__(self, content: str = "normal response", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeCompletion:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeLLM:
    def __init__(self, messages=None):
        self.calls = []
        self.messages = messages or [FakeMessage()]

    def complete(self, messages, tools=None):
        self.calls.append((messages, tools))
        return FakeCompletion(self.messages.pop(0))


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        main.conversation_memory = ConversationMemory(max_messages=20)
        main.conversation_service = None
        self.client = TestClient(main.app)

    def test_health_and_normal_response(self):
        fake = FakeLLM([FakeMessage("Hello")])
        main.conversation_service = main.ConversationService(
            main.conversation_memory, fake, main.build_registry()
        )
        response = self.client.post("/api/chat", json={"message": "Hi"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Hello")

    def test_two_independent_conversations(self):
        fake = FakeLLM([FakeMessage("one"), FakeMessage("two")])
        main.conversation_service = main.ConversationService(
            main.conversation_memory, fake, main.build_registry()
        )
        first = self.client.post("/api/chat", json={"message": "first"}).json()
        second = self.client.post("/api/chat", json={"message": "second"}).json()
        self.assertNotEqual(first["conversation_id"], second["conversation_id"])
        self.assertEqual(main.conversation_memory.get_history(first["conversation_id"])[0]["content"], "first")
        self.assertEqual(main.conversation_memory.get_history(second["conversation_id"])[0]["content"], "second")

    def test_create_and_clear_conversation(self):
        created = self.client.post("/api/conversations")
        conversation_id = created.json()["conversation_id"]
        deleted = self.client.delete(f"/api/conversations/{conversation_id}")
        self.assertEqual(deleted.status_code, 204)
        missing = self.client.post(
            "/api/chat", json={"conversation_id": conversation_id, "message": "hello"}
        )
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
