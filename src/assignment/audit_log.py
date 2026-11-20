"""
Assignment 11 — Request audit log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a request; use distinct IDs for concurrent requests by one user."""
        key = (user_id, request_id)
        if key in self._open:
            raise ValueError("A request with this user_id/request_id is already open")
        self._open[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": content_filter(text)["redacted"],
            "timestamp": utc_now_iso(),
            "started": time.perf_counter(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a request, retaining useful evidence without logging secrets."""
        key = (user_id, request_id)
        if key not in self._open:
            raise ValueError("No matching input was recorded for this request")
        entry = self._open.pop(key)
        started = entry.pop("started")
        entry.update(
            output=content_filter(text)["redacted"],
            blocked=blocked,
            layer=layer,
            completed_at=utc_now_iso(),
            latency_ms=round(max(0.0, time.perf_counter() - started) * 1000, 3),
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
