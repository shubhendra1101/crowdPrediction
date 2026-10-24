"""Alerts with hysteresis (T7.4; architecture.md §2.6).

A zone's alert is raised once its risk level has stayed at or above `alert_level` for `raise_after_s`
(default 10 s), and cleared once it has stayed below for `clear_after_s` (default 30 s). Short spikes
therefore do not alert, and an alert does not flicker off during a brief dip. Every alert keeps its
zone, highest level, time-to-critical and the reasons that triggered it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from crowdsafe.risk.rules import LEVELS, ZoneRisk


@dataclass
class Alert:
    zone_id: str
    level: str
    since: float                         # time the alert was raised
    first_seen: float                    # time the level was first reached (raise = first_seen + raise_after)
    time_to_critical_s: float | None
    reasons: list[str] = field(default_factory=list)
    cleared: float | None = None

    def to_message(self) -> dict:
        """Alert entry of the WebSocket message (architecture.md §2.7)."""
        return {"zone_id": self.zone_id, "level": self.level, "time_to_critical_s": self.time_to_critical_s,
                "reason": "; ".join(self.reasons), "since": self.since}


@dataclass
class _ZoneState:
    above_since: float | None = None
    below_since: float | None = None
    peak: int = 0
    active: Alert | None = None


class AlertManager:
    """Feed it the ZoneRisk list of each update; it returns the active alerts and keeps a history."""

    def __init__(self, raise_after_s: float = 10.0, clear_after_s: float = 30.0, alert_level: str = "High"):
        self.raise_after = raise_after_s
        self.clear_after = clear_after_s
        self.alert_idx = LEVELS.index(alert_level)
        self.state: dict[str, _ZoneState] = {}
        self.history: list[Alert] = []

    def update(self, t: float, risks: list[ZoneRisk]) -> list[Alert]:
        for r in risks:
            s = self.state.setdefault(r.zone_id, _ZoneState())
            if r.level_idx >= self.alert_idx:
                s.below_since = None
                if s.above_since is None:
                    s.above_since, s.peak = t, r.level_idx
                s.peak = max(s.peak, r.level_idx)
                if s.active is None and t - s.above_since >= self.raise_after:
                    s.active = Alert(r.zone_id, LEVELS[s.peak], t, s.above_since, r.time_to_critical_s, list(r.reasons))
                    self.history.append(s.active)
                elif s.active is not None:          # keep the alert current
                    if s.peak > LEVELS.index(s.active.level):
                        s.active.level = LEVELS[s.peak]
                    s.active.time_to_critical_s = r.time_to_critical_s
                    s.active.reasons = list(dict.fromkeys(s.active.reasons + r.reasons))[-6:]
            else:
                if s.active is None:
                    s.above_since = None
                else:
                    if s.below_since is None:
                        s.below_since = t
                    if t - s.below_since >= self.clear_after:
                        s.active.cleared = t
                        s.active, s.above_since, s.below_since, s.peak = None, None, None, 0
        return [s.active for s in self.state.values() if s.active is not None]
