import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.auth.google_oauth import GoogleOAuth, GoogleOAuthError


class FakeCredentials:
    def to_json(self):
        return '{"authorized": true}'


class FakeFlow:
    def __init__(self, state="generated-state"):
        self.state = state
        self.code_verifier = "original-code-verifier"
        self.credentials = FakeCredentials()
        self.fetch_token_calls = []

    def authorization_url(self, **kwargs):
        return "https://accounts.google.com/o/oauth2/auth?state=generated-state", self.state

    def fetch_token(self, **kwargs):
        self.fetch_token_calls.append(kwargs)


class GoogleOAuthTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.client_file = Path(self.temp_dir.name) / "credentials.json"
        self.token_file = Path(self.temp_dir.name) / "token.json"
        self.client_file.write_text("{}", encoding="utf-8")
        self.oauth = GoogleOAuth()
        self.oauth.client_secret_file = self.client_file
        self.oauth.token_file = self.token_file
        self.oauth.redirect_uri = "http://127.0.0.1:8000/auth/google/callback"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_authorization_url_stores_original_flow_and_verifier(self):
        flow = FakeFlow()
        with patch.object(self.oauth, "_flow", return_value=flow):
            url, state = self.oauth.authorization_url()

        self.assertIn("accounts.google.com", url)
        self.assertEqual(state, "generated-state")
        self.assertIs(self.oauth._pending_flows[state], flow)
        self.assertEqual(self.oauth._pending_flows[state].code_verifier, "original-code-verifier")

    def test_callback_uses_original_flow_for_token_exchange(self):
        flow = FakeFlow()
        with patch.object(self.oauth, "_flow", return_value=flow):
            self.oauth.authorization_url()

        self.oauth.complete(
            "http://127.0.0.1:8000/auth/google/callback?state=generated-state&code=redacted",
            "generated-state",
        )

        self.assertEqual(
            flow.fetch_token_calls,
            [{"authorization_response": "http://127.0.0.1:8000/auth/google/callback?state=generated-state&code=redacted"}],
        )
        self.assertEqual(self.token_file.read_text(encoding="utf-8"), '{"authorized": true}')
        self.assertNotIn("generated-state", self.oauth._pending_flows)

    def test_missing_or_invalid_state_is_rejected(self):
        with self.assertRaises(GoogleOAuthError):
            self.oauth.complete("http://127.0.0.1:8000/auth/google/callback?code=redacted")
        with self.assertRaises(GoogleOAuthError):
            self.oauth.complete(
                "http://127.0.0.1:8000/auth/google/callback?state=unknown&code=redacted",
                "unknown",
            )

    def test_local_transport_environment_is_restored(self):
        os.environ.pop("OAUTHLIB_INSECURE_TRANSPORT", None)
        with self.oauth._local_insecure_transport():
            self.assertEqual(os.environ["OAUTHLIB_INSECURE_TRANSPORT"], "1")
        self.assertNotIn("OAUTHLIB_INSECURE_TRANSPORT", os.environ)


if __name__ == "__main__":
    unittest.main()
