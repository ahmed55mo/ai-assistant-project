from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.services.llm_service import LLMService


def test_direct_groq_client_construction_and_completion_shape(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    fake_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))]
    )
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = fake_completion

    with patch("backend.services.llm_service.Groq", return_value=fake_client) as groq:
        service = LLMService()
        result = service.complete(
            [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "calculator"}}],
        )

    groq.assert_called_once_with(api_key="test-key", timeout=60.0)
    assert result is fake_completion
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"][-1] == {"role": "user", "content": "hello"}
    assert kwargs["tools"][0]["function"]["name"] == "calculator"
    assert kwargs["tool_choice"] == "auto"
