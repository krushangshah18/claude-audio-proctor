"""
Scenario A — Simultaneous Speech Detector
==========================================
Uses harmonic comb analysis on each voiced FFT frame.

When two people speak at once, the microphone captures both voices mixed.
The FFT of that mix shows TWO sets of harmonic series simultaneously:
  Voice 1: F0=120Hz → peaks at 120, 240, 360, 480, 600Hz
  Voice 2: F0=200Hz → peaks at 200, 400, 600, 800, 1000Hz

This detector:
  1. Finds all candidate F0s in 80–400Hz range
  2. Scores each F0 by how well its harmonic comb matches the spectrum
  3. If 2+ strong combs found simultaneously → flag

Output per frame: ScenarioAResult with confidence 0.0–1.0
Sustained confidence > 0.75 for > 1.5s = MULTIPLE_SPEAKERS_SIMULTANEOUS event
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# F0 search range for human voice
F0_MIN_HZ = 80
F0_MAX_HZ = 400

# How many harmonics to score per F0 candidate
N_HARMONICS = 6

# A comb is "strong" if its score is at least this fraction of the strongest comb
SECONDARY_COMB_MIN_RATIO = 0.45

# Minimum absolute comb score to be considered a real voice
MIN_COMB_SCORE = 0.08

# Smoothing window for confidence (frames)
SMOOTH_WINDOW = 5


@dataclass
class CombResult:
    f0_hz:      float
    score:      float          # 0–1, energy fraction matched by this comb
    n_harmonics_matched: int


@dataclass
class ScenarioAResult:
    confidence:   float        # 0.0–1.0
    n_combs:      int          # number of distinct F0 series found
    combs:        list[CombResult] = field(default_factory=list)
    frame_index:  int = 0


class SimultaneousSpeechDetector:
    """
    Stateful detector — tracks per-frame results and applies
    temporal smoothing + sustained-flag logic.
    """

    def __init__(
        self,
        sample_rate:          int   = 16000,
        frame_size:           int   = 512,
        sustained_frames:     int   = 47,    # ~1.5s @ 32ms frames
        confidence_threshold: float = 0.75,
    ):
        self.sr                   = sample_rate
        self.frame_size           = frame_size
        self.sustained_frames     = sustained_frames
        self.confidence_threshold = confidence_threshold

        self._frame_results: list[ScenarioAResult] = []
        self._smooth_buf:    list[float]            = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> ScenarioAResult:
        fft_mag = self._fft_magnitude(frame)
        freqs   = np.fft.rfftfreq(len(frame), d=1.0 / self.sr)

        combs    = self._find_combs(fft_mag, freqs)
        raw_conf = self._score_combs(combs)

        # Temporal smoothing
        self._smooth_buf.append(raw_conf)
        if len(self._smooth_buf) > SMOOTH_WINDOW:
            self._smooth_buf.pop(0)
        smoothed = float(np.mean(self._smooth_buf))

        result = ScenarioAResult(
            confidence=smoothed,
            n_combs=len(combs),
            combs=combs,
            frame_index=frame_index,
        )
        self._frame_results.append(result)
        return result

    def get_flag_events(self, frame_duration_s: float = 0.032) -> list[dict]:
        """
        Scan frame results for sustained high-confidence windows.
        Returns list of event dicts (matches browser event format).
        """
        events = []
        in_event = False
        event_start = 0

        for i, r in enumerate(self._frame_results):
            triggered = r.confidence >= self.confidence_threshold
            if triggered and not in_event:
                in_event    = True
                event_start = i
            elif not triggered and in_event:
                duration_frames = i - event_start
                if duration_frames >= self.sustained_frames:
                    events.append(self._build_event(
                        event_start, i, frame_duration_s
                    ))
                in_event = False

        # Close open event at end
        if in_event:
            duration_frames = len(self._frame_results) - event_start
            if duration_frames >= self.sustained_frames:
                events.append(self._build_event(
                    event_start, len(self._frame_results), frame_duration_s
                ))

        return events

    def get_frame_results(self) -> list[ScenarioAResult]:
        return self._frame_results

    def reset(self):
        self._frame_results.clear()
        self._smooth_buf.clear()

    # ------------------------------------------------------------------ #
    #  Core algorithm                                                      #
    # ------------------------------------------------------------------ #

    def _find_combs(self, fft_mag: np.ndarray, freqs: np.ndarray) -> list[CombResult]:
        """
        For each candidate F0 in voice range, score how well
        its harmonic series (F0, 2F0, 3F0 ... N*F0) matches spectrum peaks.
        Return combs sorted by score, filtered by MIN_COMB_SCORE.
        """
        freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        total_energy    = float(np.sum(fft_mag ** 2)) + 1e-10
        n_bins          = len(fft_mag)
        noise_floor     = float(np.median(fft_mag)) + 1e-10

        # Candidate F0s: every 5Hz step in voice range
        candidates = np.arange(F0_MIN_HZ, F0_MAX_HZ, 5.0)
        combs: list[CombResult] = []

        for f0 in candidates:
            harmonic_energy = 0.0
            n_matched       = 0

            for h in range(1, N_HARMONICS + 1):
                target_hz = f0 * h
                if target_hz > freqs[-1]:
                    break

                bin_idx = int(round(target_hz / freq_resolution))
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)

                lo = max(0, bin_idx - 2)
                hi = min(n_bins, bin_idx + 3)
                peak = float(np.max(fft_mag[lo:hi]))

                # Must exceed global noise floor by 3× (tighter than before)
                if peak > noise_floor * 3.0:
                    harmonic_energy += peak ** 2
                    n_matched       += 1

            score = harmonic_energy / total_energy
            # Require 4+ harmonics matched (was 3) to reduce false positives
            if score >= MIN_COMB_SCORE and n_matched >= 4:
                combs.append(CombResult(
                    f0_hz=float(f0),
                    score=score,
                    n_harmonics_matched=n_matched,
                ))

        combs.sort(key=lambda c: c.score, reverse=True)
        return self._deduplicate_combs(combs, tolerance_hz=15.0)

    @staticmethod
    def _deduplicate_combs(combs: list[CombResult], tolerance_hz: float) -> list[CombResult]:
        """Remove combs whose F0 is too close to a stronger one."""
        kept: list[CombResult] = []
        for comb in combs:
            too_close = any(abs(comb.f0_hz - k.f0_hz) < tolerance_hz for k in kept)
            if not too_close:
                kept.append(comb)
        return kept

    @staticmethod
    def _score_combs(combs: list[CombResult]) -> float:
        """
        Convert comb list to a single 0–1 confidence score.
        0 combs = 0.0, 1 strong comb = low score, 2+ combs = higher score.
        """
        if len(combs) == 0:
            return 0.0
        if len(combs) == 1:
            return 0.0  # Single speaker — not flagging

        strongest = combs[0].score
        # Check if secondary comb is genuinely strong (not just room echo)
        strong_secondaries = [
            c for c in combs[1:]
            if c.score >= strongest * SECONDARY_COMB_MIN_RATIO
        ]

        if not strong_secondaries:
            return 0.1  # Only one real comb

        # Confidence based on secondary comb strength relative to primary
        best_secondary_ratio = strong_secondaries[0].score / (strongest + 1e-10)
        # Scale: ratio 0.3→0.5 maps to confidence 0.5→1.0
        confidence = np.clip((best_secondary_ratio - 0.30) / 0.20, 0.0, 1.0)
        return float(confidence)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fft_magnitude(frame: np.ndarray) -> np.ndarray:
        windowed = frame * np.hanning(len(frame))
        return np.abs(np.fft.rfft(windowed))

    def _build_event(self, start_frame: int, end_frame: int, frame_dur: float) -> dict:
        segment = self._frame_results[start_frame:end_frame]
        max_conf = max(r.confidence for r in segment)
        avg_conf = float(np.mean([r.confidence for r in segment]))

        # Best frame's comb details
        best = max(segment, key=lambda r: r.confidence)

        return {
            "type":                "MULTIPLE_SPEAKERS_SIMULTANEOUS",
            "scenario":            "A",
            "start_s":             round(start_frame * frame_dur, 3),
            "end_s":               round(end_frame   * frame_dur, 3),
            "duration_s":          round((end_frame - start_frame) * frame_dur, 3),
            "confidence_max":      round(max_conf, 3),
            "confidence_avg":      round(avg_conf, 3),
            "f0_voices_hz":        [round(c.f0_hz, 1) for c in best.combs[:2]],
        }
