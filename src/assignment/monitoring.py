"""
Assignment 11 — Monitoring counters and threshold alerts.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    redacted_responses: int = 0
    _active_alerts: set[str] = field(default_factory=set, init=False, repr=False)

    def record_request(
        self, *, blocked: bool = False, layer: str | None = None,
        judge_checked: bool = False, judge_failed: bool = False,
        redacted: bool = False,
    ) -> list[Alert]:
        """Record one completed request, including a request blocked at input."""
        if judge_failed and not judge_checked:
            raise ValueError("A judge failure requires a judge check")
        self.total_requests += 1
        self.blocked_requests += int(blocked)
        self.rate_limit_hits += int(blocked and layer == "rate_limiter")
        self.judge_checks += int(judge_checked)
        self.judge_fails += int(judge_failed)
        self.redacted_responses += int(redacted)
        return self.check_metrics()

    def check_metrics(self) -> list[Alert]:
        """Emit an alert once per threshold crossing; retain alert history."""
        snapshot = self.snapshot()
        thresholds = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        active = set()
        new_alerts = []
        for metric, threshold in thresholds.items():
            value = snapshot[metric]
            if value > threshold:
                active.add(metric)
                if metric not in self._active_alerts:
                    alert = Alert(metric, value, threshold, f"{metric} exceeded {threshold}: {value:.3f}")
                    self.alerts.append(alert)
                    new_alerts.append(alert)
        self._active_alerts = active
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default.
        Use ``filepath or default_metrics_path()`` so running from ``src/`` does not
        create ``src/outputs/``.
        """
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "redacted_responses": self.redacted_responses,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
