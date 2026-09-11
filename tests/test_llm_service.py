import unittest
from unittest.mock import patch

import httpx
from groq import APITimeoutError, AuthenticationError, BadRequestError, RateLimitError

from backend.services.llm_service import (
    LLMService,
    _error_category,
    _safe_provider_message,
)


class LLMServiceTests(unittest.TestCase):
    def _error(self, error_type, body, status_code):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(status_code, request=request)
        return error_type("provider request failed", response=response, body=body)

    def test_safe_provider_message_extracts_only_error_text(self):
        message = _safe_provider_message(
            {"error": {"message": "Model is temporarily unavailable", "request_id": "secret-id"}}
        )
        self.assertEqual(message, "Model is temporarily unavailable")

    def test_error_categories(self):
        self.assertEqual(
            _error_category(self._error(BadRequestError, {"error": {"message": "invalid"}}, 400)),
            "bad_request",
        )
        self.assertEqual(
            _error_category(self._error(RateLimitError, {"error": {"message": "slow down"}}, 429)),
            "rate_limit",
        )
        self.assertEqual(
            _error_category(
                self._error(AuthenticationError, {"error": {"message": "invalid key"}}, 401)
            ),
            "authentication_or_permission",
        )
        self.assertEqual(
            _error_category(
                APITimeoutError(
                    request=httpx.Request(
                        "POST", "https://api.groq.com/openai/v1/chat/completions"
                    ),
                )
            ),
            "timeout",
        )

    def test_complete_logs_diagnostic_details_and_reraises(self):
        service = object.__new__(LLMService)
        service.model_name = "test-model"
        service.client = type(
            "FakeClient",
            (),
            {
                "chat": type(
                    "FakeChat",
                    (),
                    {
                        "completions": type(
                            "FakeCompletions",
                            (),
                            {
                                "create": lambda self, **kwargs: (_ for _ in ()).throw(
                                    RateLimitError(
                                        "rate limited",
                                        response=httpx.Response(
                                            429,
                                            request=httpx.Request(
                                                "POST",
                                                "https://api.groq.com/openai/v1/chat/completions",
                                            ),
                                        ),
                                        body={"error": {"message": "Too many requests"}},
                                    )
                                )
                            },
                        )()
                    },
                )()
            },
        )()
        with patch("backend.services.llm_service.logger") as logger:
            with self.assertRaises(RateLimitError):
                service.complete([{"role": "user", "content": "hello"}])
        logged = " ".join(str(call) for call in logger.error.call_args.args)
        self.assertIn("rate_limit", logged)
        self.assertIn("429", logged)
        self.assertIn("Too many requests", logged)
        self.assertNotIn("Authorization", logged)
        self.assertEqual(logger.error.call_args.args[-1], "Too many requests")


if __name__ == "__main__":
    unittest.main()
