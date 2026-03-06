"""
Output Builder
==============
Generates report.txt in the same style as Stage 1.
"""

from __future__ import annotations

from datetime import datetime

from .confidence_aggregator import FlagEvent


def build_report(
    filename:    str,
    duration_s:  float,
    flag_events: list[FlagEvent],
    summary:     dict,
    noise_stats: dict,
    report_path: str,
) -> None:

    flagged_s = sum(e.duration_s for e in flag_events)

    lines = [
        "=" * 72,
        "STAGE 2 — MULTI-SPEAKER DETECTION REPORT",
        f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        f"  File           : {filename}",
        f"  Duration       : {duration_s:.2f}s",
        "",
        "── Noise Classifier ─────────────────────────────────────────────",
        f"  Total frames   : {noise_stats.get('total_frames', 0)}",
        f"  Voice frames   : {noise_stats.get('voice_frames', 0)}"
        f"  ({noise_stats.get('voice_pct', 0.0):.1f}%)",
        f"  Noise frames   : {noise_stats.get('noise_frames', 0)}"
        f"  ({noise_stats.get('noise_pct', 0.0):.1f}%)",
        "",
        "── Detection Summary ────────────────────────────────────────────",
        f"  Total flags    : {summary['total_flags']}",
        f"  HIGH severity  : {summary['high_severity']}",
        f"  MEDIUM sev.    : {summary['medium_severity']}",
        f"  Scenario A     : {summary['scenario_a']}  (simultaneous speech)",
        f"  Scenario B     : {summary['scenario_b']}  (voice change, turn-based)",
        f"  Combined AB    : {summary['scenario_ab']}",
        f"  Total flagged  : {flagged_s:.2f}s",
        "",
    ]

    if flag_events:
        lines += [
            "── Flag Events ──────────────────────────────────────────────────",
            f"  {'#':<4} {'Scenario':<10} {'Start':>8} {'End':>8} {'Dur':>6}"
            f"  {'Conf':>6}  {'Sev':<8}  Type",
            "  " + "─" * 68,
        ]
        for i, ev in enumerate(flag_events, 1):
            lines.append(
                f"  {i:<4} {ev.scenario:<10}"
                f" {ev.start_s:>7.2f}s {ev.end_s:>7.2f}s {ev.duration_s:>5.2f}s"
                f"  {ev.confidence:>5.3f}  {ev.severity:<8}  {ev.event_type}"
            )
            if ev.scenario in ("B", "AB") and "reference_f0_hz" in ev.details:
                lines.append(
                    f"         F0: ref={ev.details['reference_f0_hz']}Hz"
                    f"  segment={ev.details.get('segment_f0_hz', '?')}Hz"
                    f"  z={ev.details.get('z_score', '?')}"
                )
            elif ev.scenario == "A" and "f0_voices_hz" in ev.details:
                lines.append(f"         F0s detected: {ev.details['f0_voices_hz']}")
        lines.append("")
    else:
        lines += ["  ✓ No multi-speaker events detected.", ""]

    lines += ["=" * 72, ""]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))