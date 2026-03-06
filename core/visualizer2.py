"""
Visualizer
==========
4-panel analysis plot — mirrors Stage 1 style.

Panels:
  1. Waveform + flag regions highlighted
  2. Scenario A confidence (simultaneous speech) over time
  3. Scenario B z-score (voice change) over time
  4. Noise classifier score over time
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.confidence_aggregator import FlagEvent


def plot_stage2_analysis(
    audio:          np.ndarray,
    sample_rate:    int,
    frame_size:     int,
    scenario_a_confidences: list[float],
    scenario_b_zscores:     list[float],
    noise_scores:           list[int],
    flag_events:    list[FlagEvent],
    stem:           str,
    output_path:    str,
) -> None:

    duration_s   = len(audio) / sample_rate
    time_audio   = np.linspace(0, duration_s, len(audio))
    frame_dur    = frame_size / sample_rate
    n_frames     = max(
        len(scenario_a_confidences),
        len(scenario_b_zscores),
        len(noise_scores)
    )
    time_frames  = np.arange(n_frames) * frame_dur

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Stage 2 — Multi-Speaker Analysis: {stem}", fontsize=13, fontweight="bold")

    # ── Panel 1: Waveform + flag regions ──────────────────────────────
    ax = axes[0]
    ax.plot(time_audio, audio, color="#4a9eff", linewidth=0.4, alpha=0.8)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title("Waveform + Detected Events", fontsize=9)
    ax.set_ylim(-1.1, 1.1)

    colors = {"A": "#ff6b6b", "B": "#ffa94d", "AB": "#cc5de8"}
    patches = []
    for ev in flag_events:
        c = colors.get(ev.scenario, "#ff6b6b")
        ax.axvspan(ev.start_s, ev.end_s, alpha=0.35, color=c, linewidth=0)
        label = f"{ev.scenario} ({ev.severity})"
        patches.append(mpatches.Patch(color=c, alpha=0.5, label=label))

    if patches:
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.7)

    # ── Panel 2: Scenario A confidence ────────────────────────────────
    ax = axes[1]
    t_a = np.arange(len(scenario_a_confidences)) * frame_dur
    ax.plot(t_a, scenario_a_confidences, color="#ff6b6b", linewidth=1.0, label="Confidence")
    ax.axhline(0.75, color="#ff6b6b", linewidth=0.8, linestyle="--", alpha=0.6, label="Flag threshold")
    ax.fill_between(t_a, scenario_a_confidences, 0, alpha=0.15, color="#ff6b6b")
    ax.set_ylabel("Confidence", fontsize=9)
    ax.set_title("Scenario A — Simultaneous Speech (Harmonic Comb)", fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right", fontsize=7)

    # ── Panel 3: Scenario B z-score ────────────────────────────────────
    ax = axes[2]
    t_b = np.arange(len(scenario_b_zscores)) * frame_dur
    ax.plot(t_b, scenario_b_zscores, color="#ffa94d", linewidth=1.0, label="Z-score")
    ax.axhline(4.0,  color="#ffa94d", linewidth=0.8, linestyle="--", alpha=0.5, label="Suspicious (4 SD)")
    ax.axhline(7.0,  color="#e03131", linewidth=0.8, linestyle="--", alpha=0.5, label="Flag (7 SD)")
    ax.fill_between(t_b, scenario_b_zscores, 0, alpha=0.15, color="#ffa94d")
    ax.set_ylabel("Z-score (SD)", fontsize=9)
    ax.set_title("Scenario B — Voice Change Detection (Distribution Shift)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    # ── Panel 4: Noise classifier score ───────────────────────────────
    ax = axes[3]
    t_n = np.arange(len(noise_scores)) * frame_dur
    ax.step(t_n, noise_scores, color="#51cf66", linewidth=0.8, where="post", label="Score (0–4)")
    ax.axhline(2.0, color="#51cf66", linewidth=0.8, linestyle="--", alpha=0.6, label="Pass threshold")
    ax.fill_between(t_n, noise_scores, 0, alpha=0.15, color="#51cf66", step="post")
    ax.set_ylabel("Score (0–4)", fontsize=9)
    ax.set_title("Noise Classifier — Frame Gate", fontsize=9)
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylim(-0.2, 4.5)
    ax.legend(loc="upper right", fontsize=7)

    # Shared formatting
    for ax in axes:
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → analysis plot : {output_path}")
