"""Probe result + pass/fail gating shared by all tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_RUN = "NOT_RUN"


@dataclass
class ProbeResult:
    probe: str
    status: str
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"status": self.status, "metrics": self.metrics}
        if self.error:
            out["error"] = self.error
        return out
