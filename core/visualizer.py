"""
VAD Visualizer
==============
4-panel analysis plot for each audio file:
  1. Waveform + detected speech segments (blue=speech, purple=whisper)
  2. Silero speech probability + threshold lines
  3. RMS vs adaptive noise floor (shows sensitivity)
  4. Per-second speech% bar chart
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List

from core.vad_engine import SpeechSegment, SAMPLE_RATE


def plot_vad_analysis(
    audio: np.ndarray,
    sr: int,
    frame_results: List[dict],
    segments: List[SpeechSegment],
    title: str,
    output_path: str,
    engine_name: str = "Silero VAD",
):
    fig, axes = plt.subplots(4, 1, figsize=(18, 13), sharex=True)
    fig.suptitle(
        f"VAD Analysis — {title}\n[Engine: {engine_name}]",
        fontsize=13, fontweight="bold", y=0.98
    )

    audio_f = audio.astype(np.float32)
    if np.abs(audio_f).max() > 1.0:
        audio_f /= np.abs(audio_f).max()

    t_audio = np.linspace(0, len(audio_f) / sr, len(audio_f))
    frame_times = [r["timestamp"] for r in frame_results]

    # colour scheme
    C_SPEECH   = "#1565C0"   # blue
    C_WHISPER  = "#6A1B9A"   # purple
    C_PROB     = "#2E7D32"
    C_PROB_RAW = "#A5D6A7"
    C_RMS      = "#1976D2"
    C_NOISE    = "#E53935"
    C_SNR      = "#FF9800"

    # ── Panel 1: Waveform ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_audio, audio_f, color="#444", linewidth=0.35, alpha=0.85)
    for seg in segments:
        c = C_WHISPER if seg.is_whisper else C_SPEECH
        ax.axvspan(seg.start_s, seg.end_s, alpha=0.20, color=c)
        ax.axvline(seg.start_s, color=c, linewidth=0.9, alpha=0.75)
        ax.axvline(seg.end_s,   color=c, linewidth=0.9, alpha=0.75)
        mid = (seg.start_s + seg.end_s) / 2
        label = f"W {seg.duration_s:.1f}s" if seg.is_whisper else f"{seg.duration_s:.1f}s"
        ax.text(mid, 0.80, label, ha="center", fontsize=6.5, color=c,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    legend = [
        mpatches.Patch(color=C_SPEECH,  alpha=0.4, label="Speech"),
        mpatches.Patch(color=C_WHISPER, alpha=0.4, label="Whisper"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Waveform  +  Detected Segments", fontsize=10)
    ax.grid(True, alpha=0.25)

    # ── Panel 2: Silero speech probability ───────────────────────────────────
    ax = axes[1]
    probs = [r["speech_prob"] for r in frame_results]
    ax.fill_between(frame_times, probs, alpha=0.3, color=C_PROB_RAW, label="Raw prob")
    # Smooth for display
    if len(probs) > 5:
        kernel = np.ones(5) / 5
        smooth = np.convolve(probs, kernel, mode="same")
        ax.plot(frame_times, smooth, color=C_PROB, linewidth=1.4, label="Smoothed (×5)")

    threshold = frame_results[0].get("threshold", 0.45) if frame_results else 0.45
    ax.axhline(0.45, color="#E53935", linewidth=1.2, linestyle="--", label="Threshold 0.45")

    # Shade speech state
    for i in range(len(frame_times) - 1):
        if frame_results[i]["is_speech"]:
            ax.axvspan(frame_times[i], frame_times[i + 1], alpha=0.08, color="green")

    ax.set_ylabel("Speech probability")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Silero VAD — Speech Probability", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel 3: RMS vs noise floor ───────────────────────────────────────────
    ax = axes[2]
    rms_vals   = [r["rms"] for r in frame_results]
    noise_vals = [r["noise_floor"] for r in frame_results]
    snr_vals   = [n * 5.0 for n in noise_vals]  # whisper threshold = 5× floor

    ax.semilogy(frame_times, rms_vals,   color=C_RMS,   linewidth=0.8, label="Frame RMS", alpha=0.9)
    ax.semilogy(frame_times, noise_vals, color=C_NOISE,  linewidth=1.5, linestyle="--", label="Noise floor")
    ax.semilogy(frame_times, snr_vals,   color=C_SNR,    linewidth=1.0, linestyle=":",
                label="Whisper threshold (5× floor)", alpha=0.85)

    ax.set_ylabel("RMS (log)")
    ax.set_title("Energy vs Adaptive Noise Floor  [adaptive tracking room conditions]", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── Panel 4: Per-second speech % bar ─────────────────────────────────────
    ax = axes[3]
    from collections import defaultdict
    by_sec: dict = defaultdict(list)
    for r in frame_results:
        by_sec[int(r["timestamp"])].append(r)

    secs = sorted(by_sec.keys())
    speech_pcts = [
        100 * sum(1 for f in by_sec[s] if f["is_speech"]) / len(by_sec[s])
        for s in secs
    ]
    colors = [C_WHISPER if p > 0 and p < 50 else (C_SPEECH if p >= 50 else "#BDBDBD") for p in speech_pcts]
    ax.bar(secs, speech_pcts, color=colors, width=0.8, alpha=0.75)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Speech %")
    ax.set_ylim(0, 110)
    ax.set_title("Per-second Speech Activity", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Plot: {os.path.basename(output_path)}")
