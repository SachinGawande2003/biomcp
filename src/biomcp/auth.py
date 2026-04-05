from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from biomcp.observability import record_auth_event

_AUTH_STATE_PATH = ".biomcp_oauth_clients.json"
_OAUTH_CLIENTS: dict[str, dict[str, Any]] | None = None
_AUTH_CODES: dict[str, dict[str, Any]] = {}
_ACCESS_TOKENS: dict[str, dict[str, Any]] = {}
_REFRESH_TOKENS: dict[str, dict[str, Any]] = {}


def _auth_store_path() -> Path:
    configured = os.getenv("BIOMCP_AUTH_STORE_FILE", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / _AUTH_STATE_PATH


def auth_enabled() -> bool:
    flag = os.getenv("BIOMCP_AUTH_ENABLED", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(os.getenv("BIOMCP_API_KEYS", "").strip())


def oauth_enabled() -> bool:
    return auth_enabled() and os.getenv("BIOMCP_OAUTH_ENABLED", "1").strip().lower() not in {"0", "false", "off"}


def api_key_auth_enabled() -> bool:
    return auth_enabled() and bool(_configured_api_keys())


def issuer_url(default_url: str) -> str:
    return os.getenv("BIOMCP_OAUTH_ISSUER_URL", default_url).rstrip("/")


def _load_oauth_clients() -> dict[str, dict[str, Any]]:
    global _OAUTH_CLIENTS
    if _OAUTH_CLIENTS is not None:
        return _OAUTH_CLIENTS

    path = _auth_store_path()
    if path.exists():
        try:
            _OAUTH_CLIENTS = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _OAUTH_CLIENTS = {}
    else:
        _OAUTH_CLIENTS = {}
    return _OAUTH_CLIENTS


def _persist_oauth_clients() -> None:
    path = _auth_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_load_oauth_clients(), indent=2, sort_keys=True), encoding="utf-8")


def _configured_api_keys() -> dict[str, dict[str, Any]]:
    raw = os.getenv("BIOMCP_API_KEYS", "").strip()
    if not raw:
        return {}

    default_requests = int(os.getenv("BIOMCP_API_KEY_RATE_LIMIT_REQUESTS", "600"))
    default_window = int(os.getenv("BIOMCP_API_KEY_RATE_LIMIT_WINDOW_SECONDS", "60"))
    parsed: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(raw.split(","), start=1):
        token = chunk.strip()
        if not token:
            continue
        if ":" in token:
            key_id, secret = token.split(":", 1)
        else:
            key_id, secret = f"key-{index}", token
        parsed[secret.strip()] = {
            "key_id": key_id.strip(),
            "secret": secret.strip(),
            "rate_limit_requests": default_requests,
            "rate_limit_window_seconds": default_window,
            "scopes": ["mcp:tools", "mcp:resources"],
        }
    return parsed


def _code_challenge_s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _validate_redirect_uri(client_id: str, redirect_uri: str) -> None:
    client = _load_oauth_clients().get(client_id)
    if client is None:
        raise ValueError("Unknown OAuth client_id.")
    if redirect_uri not in client.get("redirect_uris", []):
        raise ValueError("redirect_uri is not registered for this client.")


def register_oauth_client(payload: dict[str, Any]) -> dict[str, Any]:
    if not oauth_enabled():
        raise ValueError("OAuth is not enabled.")

    redirect_uris = payload.get("redirect_uris") or []
    if not isinstance(redirect_uris, list) or not redirect_uris:
        raise ValueError("redirect_uris must be a non-empty array.")

    client_name = str(payload.get("client_name", "") or "BioMCP client").strip()
    client_id = f"biomcp-{secrets.token_urlsafe(12)}"
    client = {
        "client_id": client_id,
        "client_name": client_name,
        "redirect_uris": [str(uri).strip() for uri in redirect_uris if str(uri).strip()],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "created_at": int(time.time()),
    }
    clients = _load_oauth_clients()
    clients[client_id] = client
    _persist_oauth_clients()
    record_auth_event("client_registered")
    return client


def build_authorization_metadata(base_url: str) -> dict[str, Any]:
    base = issuer_url(base_url)
    metadata = {
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": ["mcp:tools", "mcp:resources"],
    }
    return metadata


def build_consent_page(params: dict[str, str], *, server_name: str) -> str:
    hidden = "\n".join(
        f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(value)}" />'
        for key, value in params.items()
    )
    client = _load_oauth_clients().get(params["client_id"], {})
    client_name = html.escape(str(client.get("client_name", params["client_id"])))
    scope = html.escape(params.get("scope", "mcp:tools mcp:resources"))
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{html.escape(server_name)} Authorization</title></head>
<body style="font-family: sans-serif; max-width: 560px; margin: 3rem auto; line-height: 1.5;">
  <h1>{html.escape(server_name)} access request</h1>
  <p><strong>{client_name}</strong> wants to connect to your hosted BioMCP server.</p>
  <p>Requested scopes: <code>{scope}</code></p>
  <form method="post" action="/oauth/authorize">
    {hidden}
    <button type="submit" name="decision" value="approve">Approve</button>
    <button type="submit" name="decision" value="deny">Deny</button>
  </form>
</body></html>"""


def issue_authorization_code(
    *,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    scope: str,
    subject: str,
) -> str:
    _validate_redirect_uri(client_id, redirect_uri)
    if code_challenge_method != "S256":
        raise ValueError("Only PKCE S256 is supported.")
    code = secrets.token_urlsafe(32)
    _AUTH_CODES[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "scope": scope or "mcp:tools mcp:resources",
        "subject": subject,
        "expires_at": time.time() + 300,
    }
    record_auth_event("authorization_code_issued")
    return code


def build_redirect_uri(redirect_uri: str, *, code: str | None = None, state: str = "", error: str = "") -> str:
    params: dict[str, str] = {}
    if code:
        params["code"] = code
    if state:
        params["state"] = state
    if error:
        params["error"] = error
    separator = "&" if "?" in redirect_uri else "?"
    return f"{redirect_uri}{separator}{urlencode(params)}"


def exchange_authorization_code(
    *,
    code: str,
    client_id: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    payload = _AUTH_CODES.pop(code, None)
    if not payload:
        raise ValueError("Invalid or expired authorization code.")
    if payload["expires_at"] < time.time():
        raise ValueError("Authorization code has expired.")
    if payload["client_id"] != client_id or payload["redirect_uri"] != redirect_uri:
        raise ValueError("Authorization code does not match the client or redirect URI.")
    if _code_challenge_s256(code_verifier) != payload["code_challenge"]:
        raise ValueError("PKCE verification failed.")

    access_token = secrets.token_urlsafe(32)
    refresh_token = secrets.token_urlsafe(32)
    expires_in = int(os.getenv("BIOMCP_OAUTH_TOKEN_TTL_SECONDS", "3600"))
    token_payload = {
        "client_id": client_id,
        "subject": payload["subject"],
        "scope": payload["scope"],
        "expires_at": time.time() + expires_in,
    }
    _ACCESS_TOKENS[access_token] = token_payload
    _REFRESH_TOKENS[refresh_token] = {
        "client_id": client_id,
        "subject": payload["subject"],
        "scope": payload["scope"],
    }
    record_auth_event("token_issued")
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "refresh_token": refresh_token,
        "scope": payload["scope"],
    }


def refresh_access_token(*, refresh_token: str, client_id: str) -> dict[str, Any]:
    payload = _REFRESH_TOKENS.get(refresh_token)
    if not payload or payload["client_id"] != client_id:
        raise ValueError("Invalid refresh token.")
    access_token = secrets.token_urlsafe(32)
    expires_in = int(os.getenv("BIOMCP_OAUTH_TOKEN_TTL_SECONDS", "3600"))
    _ACCESS_TOKENS[access_token] = {
        "client_id": client_id,
        "subject": payload["subject"],
        "scope": payload["scope"],
        "expires_at": time.time() + expires_in,
    }
    record_auth_event("token_refreshed")
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "scope": payload["scope"],
    }


def validate_access_token(token: str) -> dict[str, Any] | None:
    payload = _ACCESS_TOKENS.get(token)
    if not payload:
        return None
    if payload["expires_at"] < time.time():
        _ACCESS_TOKENS.pop(token, None)
        return None
    return payload


def validate_api_key(secret: str) -> dict[str, Any] | None:
    return _configured_api_keys().get(secret)


def default_auth_subject() -> str:
    return os.getenv("BIOMCP_OAUTH_DEFAULT_SUBJECT", "heuris-biomcp-user").strip()


__all__ = [
    "api_key_auth_enabled",
    "auth_enabled",
    "build_authorization_metadata",
    "build_consent_page",
    "build_redirect_uri",
    "default_auth_subject",
    "exchange_authorization_code",
    "issuer_url",
    "oauth_enabled",
    "refresh_access_token",
    "register_oauth_client",
    "issue_authorization_code",
    "validate_access_token",
    "validate_api_key",
]
