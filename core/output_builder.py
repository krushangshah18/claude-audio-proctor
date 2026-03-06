"""
Output Builder
==============
Builds all analysis artifacts from VAD results:
  1. speech_only.wav       — original audio with non-speech zeroed out
  2. report.txt            — detailed timestamp + per-segment analysis
"""

from __future__ import annotations

import os
from typing import List
from collections import defaultdict

import numpy as np

from core.vad_engine import SpeechSegment, SAMPLE_RATE


def build_speech_only_audio(
    audio: np.ndarray,
    segments: List[SpeechSegment],
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Returns a copy of audio with all non-speech regions set to silence.
    Segments already include padding from post-processing.
    """
    output = np.zeros_like(audio, dtype=np.float32)
    for seg in segments:
        start = max(0, int(seg.start_s * sr))
        end = min(len(audio), int(seg.end_s * sr))
        output[start:end] = audio[start:end]
    return output


def build_report(
    audio_file: str,
    segments: List[SpeechSegment],
    frame_results: List[dict],
    total_duration_s: float,
    engine_name: str,
    output_path: str,
) -> str:

    lines = []
    SEP = "=" * 70
    sep = "─" * 70

    lines += [SEP, "VAD ANALYSIS REPORT  —  Stage 1", SEP]
    lines += [
        f"  Source      : {audio_file}",
        f"  Duration    : {total_duration_s:.2f}s",
        f"  Engine      : {engine_name}",
        "",
    ]

    total_speech = sum(s.duration_s for s in segments)
    total_silence = total_duration_s - total_speech
    whisper_segs = [s for s in segments if s.is_whisper]

    lines += [
        "SUMMARY",
        f"  Speech detected : {total_speech:.2f}s  ({100*total_speech/max(total_duration_s,1):.1f}%)",
        f"  Silence/noise   : {total_silence:.2f}s  ({100*total_silence/max(total_duration_s,1):.1f}%)",
        f"  Segments        : {len(segments)}",
        f"  Whisper segs    : {len(whisper_segs)}",
    ]
    if frame_results:
        lines.append(f"  Final noise RMS : {frame_results[-1]['noise_floor']:.5f}")
    lines.append("")

    # Segment table
    lines += [
        "DETECTED SPEECH SEGMENTS",
        f"  {'#':>3}  {'Start':>8}  {'End':>8}  {'Duration':>9}  "
        f"{'Avg Prob':>9}  {'Avg RMS':>9}  Type",
        f"  {'─'*3}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}",
    ]
    for i, seg in enumerate(segments, 1):
        stype = "WHISPER" if seg.is_whisper else "speech"
        lines.append(
            f"  {i:>3}  {seg.start_s:>7.2f}s  {seg.end_s:>7.2f}s  "
            f"{seg.duration_s:>8.2f}s  {seg.avg_prob:>9.3f}  "
            f"{seg.avg_rms:>9.5f}  {stype}"
        )
    lines.append("")

    # State transitions
    lines += [
        "FRAME-LEVEL TRANSITIONS",
        f"  {'Time':>8}  {'State':>8}  {'Prob':>7}  {'RMS':>10}  {'NoiseFloor':>11}",
        f"  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*10}  {'─'*11}",
    ]
    prev_state = None
    for r in frame_results:
        state = r["state"]
        if state != prev_state:
            lines.append(
                f"  {r['timestamp']:>7.2f}s  {state:>8}  {r['speech_prob']:>7.3f}  "
                f"{r['rms']:>10.5f}  {r['noise_floor']:>11.5f}"
            )
            prev_state = state
    lines.append("")

    # Per-second visual summary
    lines += [
        "PER-SECOND SUMMARY",
        f"  {'Sec':>5}  {'Speech%':>8}  {'AvgProb':>8}  {'AvgRMS':>9}  Visual (each █ = 5%)",
        f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*40}",
    ]
    by_second: dict[int, list] = defaultdict(list)
    for r in frame_results:
        by_second[int(r["timestamp"])].append(r)

    for sec in sorted(by_second.keys()):
        frames = by_second[sec]
        speech_pct = 100 * sum(1 for f in frames if f["is_speech"]) / len(frames)
        avg_prob = np.mean([f["speech_prob"] for f in frames])
        avg_rms = np.mean([f["rms"] for f in frames])
        filled = int(speech_pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        lines.append(
            f"  {sec:>4}s  {speech_pct:>7.0f}%  {avg_prob:>8.3f}  {avg_rms:>9.5f}  {bar}"
        )

    lines += ["", SEP, "HOW TO INTERPRET", SEP]
    lines.append("""
  Segments marked [WHISPER] have RMS < 5× noise floor.
  Check _analysis.png for visual confirmation of each segment.

  If road noise / honking appears as speech:
    → Increase --threshold (try 0.55)
    → Increase --onset-chunks (try 5)

  If real speech is being missed:
    → Decrease --threshold (try 0.35)
    → Check _analysis.png Panel 2 — is the speech_prob spiking?

  If whisper is missed (not in segments list):
    → Decrease --threshold (try 0.35)
    → Check Panel 3 — is the noise floor too high?
""")

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    return text
