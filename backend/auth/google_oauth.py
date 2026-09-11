from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials


GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",
]


class GoogleOAuthError(RuntimeError):
    """Raised when Google OAuth is not configured or cannot complete."""


class GoogleOAuth:
    _transport_lock = Lock()

    def __init__(self) -> None:
        self.client_secret_file = Path(os.getenv("GOOGLE_CLIENT_SECRET_FILE", "credentials.json"))
        self.token_file = Path(os.getenv("GOOGLE_TOKEN_FILE", "token.json"))
        self.redirect_uri = os.getenv(
            "GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/auth/google/callback"
        )
        self.allow_insecure_local_oauth = (
            os.getenv("GOOGLE_ALLOW_INSECURE_LOCAL_OAUTH", "true").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._pending_flows: dict[str, Flow] = {}
        self._pending_flows_lock = Lock()

    def _flow(self, state: str | None = None) -> Flow:
        if not self.client_secret_file.exists():
            raise GoogleOAuthError(
                f"Google OAuth client file not found: {self.client_secret_file}"
            )
        flow = Flow.from_client_secrets_file(
            str(self.client_secret_file), scopes=GOOGLE_SCOPES, state=state
        )
        flow.redirect_uri = self.redirect_uri
        return flow

    def authorization_url(self) -> tuple[str, str]:
        flow = self._flow()
        url, state = flow.authorization_url(
            access_type="offline", include_granted_scopes="true", prompt="consent"
        )
        with self._pending_flows_lock:
            self._pending_flows[state] = flow
        return url, state

    def complete(self, authorization_response: str, state: str | None = None) -> None:
        if not state:
            raise GoogleOAuthError("OAuth state is missing.")
        with self._pending_flows_lock:
            flow = self._pending_flows.pop(state, None)
        if flow is None:
            raise GoogleOAuthError("OAuth state is invalid or has expired.")
        with self._local_insecure_transport():
            flow.fetch_token(authorization_response=authorization_response)
        self.token_file.write_text(flow.credentials.to_json(), encoding="utf-8")

    @contextmanager
    def _local_insecure_transport(self):
        parsed_uri = urlparse(self.redirect_uri)
        is_local_http = parsed_uri.scheme == "http" and parsed_uri.hostname in {
            "127.0.0.1",
            "localhost",
        }
        if not (self.allow_insecure_local_oauth and is_local_http):
            yield
            return

        # OAuthlib permits HTTP only for this local loopback callback. Production
        # deployments must use HTTPS and must not enable this development flag.
        with self._transport_lock:
            previous_value = os.environ.get("OAUTHLIB_INSECURE_TRANSPORT")
            os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
            try:
                yield
            finally:
                if previous_value is None:
                    os.environ.pop("OAUTHLIB_INSECURE_TRANSPORT", None)
                else:
                    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = previous_value

    def credentials(self) -> Credentials:
        if not self.token_file.exists():
            raise GoogleOAuthError("Google account is not authenticated. Open /auth/google first.")
        try:
            info: dict[str, Any] = json.loads(self.token_file.read_text(encoding="utf-8"))
            credentials = Credentials.from_authorized_user_info(info, GOOGLE_SCOPES)
        except (OSError, ValueError) as exc:
            raise GoogleOAuthError("Stored Google credentials are invalid.") from exc
        if credentials.expired and credentials.refresh_token:
            try:
                from google.auth.transport.requests import Request

                credentials.refresh(Request())
                self.token_file.write_text(credentials.to_json(), encoding="utf-8")
            except Exception as exc:
                raise GoogleOAuthError("Stored Google credentials could not be refreshed.") from exc
        if not credentials.valid:
            raise GoogleOAuthError("Stored Google credentials are no longer valid.")
        return credentials
