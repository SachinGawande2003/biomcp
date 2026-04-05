from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from biomcp.tools.ncbi import search_pubmed
from biomcp.utils import get_http_client


def _watch_store_path() -> Path:
    configured = Path(
        os.environ.get("BIOMCP_SESSION_STORE_DIR")
        or (Path(__file__).resolve().parents[2] / ".biomcp_sessions")
    )
    configured.mkdir(parents=True, exist_ok=True)
    return configured / "watches.json"


def _load_watches() -> dict[str, dict[str, Any]]:
    path = _watch_store_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_watches(payload: dict[str, dict[str, Any]]) -> None:
    _watch_store_path().write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def list_watches() -> list[dict[str, Any]]:
    return sorted(_load_watches().values(), key=lambda item: item.get("created_at", ""), reverse=True)


def upsert_watch(topic: str, *, label: str = "") -> dict[str, Any]:
    normalized = topic.strip()
    if not normalized:
        raise ValueError("topic is required for watch registration.")

    watches = _load_watches()
    key = normalized.lower()
    existing = watches.get(key)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = existing or {
        "watch_id": f"watch-{int(time.time())}-{abs(hash(key)) % 100000}",
        "topic": normalized,
        "created_at": now,
        "last_checked_at": "",
        "last_seen": {"pubmed": [], "biorxiv": []},
    }
    payload["label"] = label.strip()
    payload["updated_at"] = now
    watches[key] = payload
    _save_watches(watches)
    return payload


def remove_watch(topic: str) -> bool:
    watches = _load_watches()
    removed = watches.pop(topic.strip().lower(), None)
    if removed is not None:
        _save_watches(watches)
    return removed is not None


async def _query_biorxiv(topic: str, since_date: str) -> list[dict[str, Any]]:
    client = await get_http_client()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    resp = await client.get(f"https://api.biorxiv.org/details/biorxiv/{since_date}/{today}/0/json")
    resp.raise_for_status()
    payload = resp.json()
    tokens = {token for token in topic.lower().split() if len(token) > 2}
    matches: list[dict[str, Any]] = []
    for item in payload.get("collection", [])[:200]:
        text = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("abstract", "")),
                str(item.get("category", "")),
            ]
        ).lower()
        if tokens and not any(token in text for token in tokens):
            continue
        matches.append(
            {
                "title": item.get("title", ""),
                "doi": item.get("doi", ""),
                "date": item.get("date", ""),
                "server": item.get("server", "bioRxiv"),
                "url": f"https://www.biorxiv.org/content/{item.get('doi', '')}v1",
            }
        )
    return matches


async def check_watch(topic: str) -> dict[str, Any]:
    watches = _load_watches()
    key = topic.strip().lower()
    watch = watches.get(key)
    if watch is None:
        raise LookupError(f"No watch found for '{topic}'.")

    last_checked_at = watch.get("last_checked_at") or watch.get("created_at", "")
    since_date = (last_checked_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())).split("T", 1)[0]
    pubmed_query = f'({watch["topic"]}) AND ("{since_date}"[Date - Publication] : "3000"[Date - Publication])'
    pubmed = await search_pubmed(pubmed_query, max_results=20, sort="pub_date")
    biorxiv = await _query_biorxiv(watch["topic"], since_date)

    seen_pubmed = set(watch.get("last_seen", {}).get("pubmed", []))
    seen_biorxiv = set(watch.get("last_seen", {}).get("biorxiv", []))
    new_pubmed = [article for article in pubmed.get("articles", []) if article.get("pmid") not in seen_pubmed]
    new_biorxiv = [item for item in biorxiv if item.get("doi") not in seen_biorxiv]

    watch["last_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    watch["last_seen"] = {
        "pubmed": [article.get("pmid", "") for article in pubmed.get("articles", []) if article.get("pmid")],
        "biorxiv": [item.get("doi", "") for item in biorxiv if item.get("doi")],
    }
    watches[key] = watch
    _save_watches(watches)

    return {
        "watch": watch,
        "new_items": {
            "pubmed": new_pubmed,
            "biorxiv": new_biorxiv,
        },
        "counts": {
            "pubmed_total": pubmed.get("total_found", 0),
            "pubmed_new": len(new_pubmed),
            "biorxiv_new": len(new_biorxiv),
        },
    }


def resource_uri_for_watch(topic: str) -> str:
    return f"biomcp://watch/{quote(topic.strip())}"


__all__ = [
    "check_watch",
    "list_watches",
    "remove_watch",
    "resource_uri_for_watch",
    "upsert_watch",
]
