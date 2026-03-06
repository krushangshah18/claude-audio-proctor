"""
Noise Classifier
================
4-check gate that runs on every 32ms frame BEFORE the speaker detectors.
Prevents road noise, car horns, train rumble from triggering false flags.

Checks:
  1. HNR  — Harmonics-to-Noise Ratio  (voice has strong harmonics)
  2. Spectral Flatness                 (voice is peaked, noise is flat)
  3. Temporal Pattern                  (speech has rhythmic energy bursts)
  4. Frequency Range                   (only 80Hz–3400Hz matters for voice)

Score 0–4. Frame needs >= 2 to be passed downstream.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

VOICE_LOW_HZ  = 80
VOICE_HIGH_HZ = 3400

# HNR: ratio of harmonic energy to noise floor in dB
HNR_VOICE_MIN_DB  = 6.0   # clean voice typically >10dB
HNR_NOISE_MAX_DB  = 3.0   # road noise typically <3dB

# Spectral flatness: 0=pure tone, 1=white noise
FLATNESS_VOICE_MAX = 0.35  # voice-like signals are peaked
FLATNESS_NOISE_MIN = 0.60  # noise is broad and flat

# Temporal modulation: speech syllable rate 4–8 Hz
SYLLABLE_RATE_LOW  = 3.0
SYLLABLE_RATE_HIGH = 9.0


@dataclass
class NoiseClassifierConfig:
    min_pass_score: int = 2          # frames need >= this to proceed
    voice_low_hz:   int = VOICE_LOW_HZ
    voice_high_hz:  int = VOICE_HIGH_HZ
    hnr_min_db:   float = HNR_VOICE_MIN_DB
    flatness_max: float = FLATNESS_VOICE_MAX


@dataclass
class FrameClassification:
    is_voice: bool
    score: int                        # 0–4
    hnr_db: float
    spectral_flatness: float
    in_voice_band: bool
    check_details: dict = field(default_factory=dict)


class NoiseClassifier:
    """
    Stateless per-frame classifier + stateful temporal pattern tracker.
    Call classify_frame() for each 32ms audio chunk.
    """

    def __init__(self, config: NoiseClassifierConfig, sample_rate: int = 16000):
        self.cfg = config
        self.sr  = sample_rate

        # Temporal pattern tracking — rolling energy envelope
        self._energy_history: list[float] = []
        self._history_max = 40   # ~1.3 seconds of 32ms frames

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def classify_frame(self, frame: np.ndarray) -> FrameClassification:
        """
        frame: 1-D float32 audio samples at self.sr sample rate.
               Expected ~512 samples (32ms @ 16kHz).
        """
        fft_mag = self._fft_magnitude(frame)
        freqs   = np.fft.rfftfreq(len(frame), d=1.0 / self.sr)

        # ── Check 1: Frequency range gate ──────────────────────────────
        band_mask = (freqs >= self.cfg.voice_low_hz) & (freqs <= self.cfg.voice_high_hz)
        band_energy = float(np.sum(fft_mag[band_mask] ** 2))
        total_energy = float(np.sum(fft_mag ** 2)) + 1e-10
        in_voice_band = (band_energy / total_energy) > 0.45

        # ── Check 2: HNR ───────────────────────────────────────────────
        hnr_db = self._compute_hnr(frame, fft_mag, freqs)
        hnr_pass = hnr_db >= self.cfg.hnr_min_db

        # ── Check 3: Spectral Flatness ──────────────────────────────────
        flatness = self._spectral_flatness(fft_mag[band_mask] + 1e-10)
        flatness_pass = flatness <= self.cfg.flatness_max

        # ── Check 4: Temporal Pattern ───────────────────────────────────
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self._energy_history.append(rms)
        if len(self._energy_history) > self._history_max:
            self._energy_history.pop(0)
        temporal_pass = self._check_temporal_pattern()

        score = sum([in_voice_band, hnr_pass, flatness_pass, temporal_pass])
        is_voice = score >= self.cfg.min_pass_score

        return FrameClassification(
            is_voice=is_voice,
            score=score,
            hnr_db=hnr_db,
            spectral_flatness=flatness,
            in_voice_band=in_voice_band,
            check_details={
                "hnr_pass":      hnr_pass,
                "flatness_pass": flatness_pass,
                "band_pass":     in_voice_band,
                "temporal_pass": temporal_pass,
            },
        )

    def reset(self):
        self._energy_history.clear()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fft_magnitude(frame: np.ndarray) -> np.ndarray:
        windowed = frame * np.hanning(len(frame))
        return np.abs(np.fft.rfft(windowed))

    def _compute_hnr(
        self, frame: np.ndarray, fft_mag: np.ndarray, freqs: np.ndarray
    ) -> float:
        """
        Estimate HNR via autocorrelation.
        Peak in autocorrelation at lag > 0 indicates periodicity (harmonics).
        Returns dB value; higher = more harmonic = more voice-like.
        """
        if len(frame) < 64:
            return 0.0

        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]   # keep positive lags
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Search for peak in pitch range 80–500Hz
        sr = self.sr
        min_lag = int(sr / 500)
        max_lag = int(sr / 80)
        max_lag = min(max_lag, len(autocorr) - 1)

        if min_lag >= max_lag:
            return 0.0

        peak_val = float(np.max(autocorr[min_lag:max_lag]))
        peak_val = np.clip(peak_val, 0.0, 1.0 - 1e-10)

        # Convert correlation to HNR in dB
        hnr_linear = peak_val / (1.0 - peak_val + 1e-10)
        return float(10.0 * np.log10(hnr_linear + 1e-10))

    @staticmethod
    def _spectral_flatness(magnitudes: np.ndarray) -> float:
        """
        Wiener entropy / spectral flatness.
        0 = pure sinusoid (harmonic voice), 1 = white noise.
        """
        log_mean = np.mean(np.log(magnitudes + 1e-10))
        arithmetic_mean = np.mean(magnitudes)
        geometric_mean  = np.exp(log_mean)
        return float(np.clip(geometric_mean / (arithmetic_mean + 1e-10), 0.0, 1.0))

    def _check_temporal_pattern(self) -> bool:
        """
        Speech has energy modulation at syllable rate ~4–8 Hz.
        Check if the energy envelope over ~1s has variance consistent
        with speech rhythm (not flat noise, not single burst).
        """
        if len(self._energy_history) < 12:
            return True   # not enough history yet, don't penalise

        energies = np.array(self._energy_history)
        mean_e   = np.mean(energies) + 1e-10
        cv       = np.std(energies) / mean_e   # coefficient of variation

        # Pure noise: CV very low (flat energy)
        # Single event: CV very high (one spike, rest silence)
        # Speech: CV in middle range
        return 0.15 <= cv <= 2.5
