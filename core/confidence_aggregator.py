"""
Confidence Aggregator
=====================
Combines outputs from Scenario A and B detectors into final flag events.
Applies the global sustained-confidence rule:
  → Emit flag only if confidence > 0.75 for > 1.5 seconds

Also deduplicates overlapping events and assigns severity levels.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

CONFIDENCE_THRESHOLD  = 0.75
MIN_DURATION_S        = 1.5
SEVERITY_HIGH         = 0.90
SEVERITY_MEDIUM       = 0.75


@dataclass
class FlagEvent:
    event_type:   str
    scenario:     str     # "A", "B", or "AB"
    start_s:      float
    end_s:        float
    duration_s:   float
    confidence:   float
    severity:     str     # "HIGH", "MEDIUM"
    details:      dict


class ConfidenceAggregator:

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        min_duration_s:       float = MIN_DURATION_S,
    ):
        self.threshold    = confidence_threshold
        self.min_duration = min_duration_s

    def aggregate(
        self,
        scenario_a_events: list[dict],
        scenario_b_events: list[dict],
    ) -> list[FlagEvent]:
        """
        Merge A + B events, apply duration filter, deduplicate overlaps.
        """
        all_events: list[FlagEvent] = []

        for ev in scenario_a_events:
            if ev["duration_s"] >= self.min_duration and ev["confidence_max"] >= self.threshold:
                all_events.append(FlagEvent(
                    event_type  = ev["type"],
                    scenario    = "A",
                    start_s     = ev["start_s"],
                    end_s       = ev["end_s"],
                    duration_s  = ev["duration_s"],
                    confidence  = ev["confidence_max"],
                    severity    = "HIGH" if ev["confidence_max"] >= SEVERITY_HIGH else "MEDIUM",
                    details     = ev,
                ))

        for ev in scenario_b_events:
            if ev["duration_s"] >= self.min_duration and ev["confidence"] >= self.threshold:
                all_events.append(FlagEvent(
                    event_type  = ev["type"],
                    scenario    = "B",
                    start_s     = ev["start_s"],
                    end_s       = ev["end_s"],
                    duration_s  = ev["duration_s"],
                    confidence  = ev["confidence"],
                    severity    = "HIGH" if ev["confidence"] >= SEVERITY_HIGH else "MEDIUM",
                    details     = ev,
                ))

        all_events.sort(key=lambda e: e.start_s)
        return self._merge_overlapping(all_events)

    @staticmethod
    def _merge_overlapping(events: list[FlagEvent]) -> list[FlagEvent]:
        """Merge events that overlap in time into a single higher-severity event."""
        if not events:
            return []

        merged: list[FlagEvent] = [events[0]]
        for ev in events[1:]:
            last = merged[-1]
            if ev.start_s <= last.end_s + 0.5:   # 500ms tolerance
                # Merge: extend window, take max confidence, mark as AB
                new_end  = max(last.end_s, ev.end_s)
                new_conf = max(last.confidence, ev.confidence)
                scenario = "AB" if last.scenario != ev.scenario else last.scenario
                merged[-1] = FlagEvent(
                    event_type  = "MULTIPLE_SPEAKERS_DETECTED",
                    scenario    = scenario,
                    start_s     = last.start_s,
                    end_s       = new_end,
                    duration_s  = new_end - last.start_s,
                    confidence  = new_conf,
                    severity    = "HIGH" if new_conf >= SEVERITY_HIGH else "MEDIUM",
                    details     = {**last.details, "merged_with": ev.details},
                )
            else:
                merged.append(ev)

        return merged

    @staticmethod
    def summarise(events: list[FlagEvent]) -> dict:
        if not events:
            return {
                "total_flags":     0,
                "high_severity":   0,
                "medium_severity": 0,
                "scenario_a":      0,
                "scenario_b":      0,
                "scenario_ab":     0,
                "total_flagged_s": 0.0,
            }

        return {
            "total_flags":     len(events),
            "high_severity":   sum(1 for e in events if e.severity == "HIGH"),
            "medium_severity": sum(1 for e in events if e.severity == "MEDIUM"),
            "scenario_a":      sum(1 for e in events if e.scenario == "A"),
            "scenario_b":      sum(1 for e in events if e.scenario == "B"),
            "scenario_ab":     sum(1 for e in events if e.scenario == "AB"),
            "total_flagged_s": round(sum(e.duration_s for e in events), 2),
        }
