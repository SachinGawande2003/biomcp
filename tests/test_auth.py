from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

import biomcp.auth as auth_module


def _pkce_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def test_register_oauth_client_and_exchange_authorization_code(
    monkeypatch: pytest.MonkeyPatch,
):
    temp_store = Path(".codex_test_tmp") / "oauth-test-clients.json"
    temp_store.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("BIOMCP_AUTH_ENABLED", "1")
    monkeypatch.setenv("BIOMCP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("BIOMCP_AUTH_STORE_FILE", str(temp_store))
    monkeypatch.setattr(auth_module, "_OAUTH_CLIENTS", None)
    monkeypatch.setattr(auth_module, "_AUTH_CODES", {})
    monkeypatch.setattr(auth_module, "_ACCESS_TOKENS", {})
    monkeypatch.setattr(auth_module, "_REFRESH_TOKENS", {})

    client = auth_module.register_oauth_client(
        {"client_name": "Claude Connector", "redirect_uris": ["https://claude.ai/callback"]}
    )
    verifier = "pkce-verifier-1234567890"
    code = auth_module.issue_authorization_code(
        client_id=client["client_id"],
        redirect_uri="https://claude.ai/callback",
        code_challenge=_pkce_s256(verifier),
        code_challenge_method="S256",
        scope="mcp:tools mcp:resources",
        subject="test-user",
    )

    token_payload = auth_module.exchange_authorization_code(
        code=code,
        client_id=client["client_id"],
        redirect_uri="https://claude.ai/callback",
        code_verifier=verifier,
    )

    assert token_payload["token_type"] == "Bearer"
    assert token_payload["scope"] == "mcp:tools mcp:resources"
    validated = auth_module.validate_access_token(token_payload["access_token"])
    assert validated is not None
    assert validated["subject"] == "test-user"


def test_validate_api_key_uses_configured_limits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIOMCP_AUTH_ENABLED", "1")
    monkeypatch.setenv("BIOMCP_API_KEYS", "primary:test-secret")
    monkeypatch.setenv("BIOMCP_API_KEY_RATE_LIMIT_REQUESTS", "55")
    monkeypatch.setenv("BIOMCP_API_KEY_RATE_LIMIT_WINDOW_SECONDS", "30")

    payload = auth_module.validate_api_key("test-secret")

    assert payload is not None
    assert payload["key_id"] == "primary"
    assert payload["rate_limit_requests"] == 55
    assert payload["rate_limit_window_seconds"] == 30
