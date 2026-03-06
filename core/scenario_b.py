"""
Scenario B — Turn-Based Speaker Change Detector
================================================
Detects when a DIFFERENT person speaks in a new turn after the candidate pauses.

Unlike Scenario A (simultaneous), here only one voice is present at a time —
so harmonic analysis won't help. Instead we track the candidate's voice
DISTRIBUTION over time and flag when a new speech segment is statistically
inconsistent with that established distribution.

Features tracked per voiced frame:
  - F0 (fundamental frequency / pitch) via autocorrelation
  - Spectral Centroid (timbral "brightness")
  - MFCC-1 + MFCC-2 (vocal tract shape — first 2 cepstral coefficients)

Algorithm:
  1. Build reference profile from first N seconds of speech (enrollment window)
  2. After each silence gap, compare incoming speech distribution vs reference
  3. Use z-score distance on F0 + centroid:
       distance = |new_mean - ref_mean| / ref_std
     If distance > THRESHOLD → flag as potential voice change

Design notes:
  - Candidate's OWN voice variation (tired, excited) stays within ~2–3 SD
  - Different speaker typically lands > 4 SD, often > 8 SD
  - Enroll from first 8s of exam speech, update slowly (EMA) after that
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# Enrollment: build reference from this many seconds of initial speech
ENROLLMENT_MIN_FRAMES = 60       # ~2s minimum before we trust the profile
ENROLLMENT_TARGET_S   = 8.0      # aim for 8s of speech for solid reference

# Detection thresholds (z-score / SD distance)
THRESHOLD_SUSPICIOUS  = 4.0      # > 4 SD → suspicious
THRESHOLD_FLAG        = 7.0      # > 7 SD → strong flag

# A "new turn" requires silence gap of at least this many frames
MIN_SILENCE_GAP_FRAMES = 10      # ~320ms gap

# Minimum frames in new segment before comparing
MIN_COMPARISON_FRAMES  = 15      # ~480ms of new speech

# EMA update factor for reference profile after enrollment
PROFILE_EMA_ALPHA = 0.05         # slow drift to allow for exam fatigue


class DetectorState(Enum):
    ENROLLING    = "enrolling"
    MONITORING   = "monitoring"
    IN_SILENCE   = "in_silence"
    COMPARING    = "comparing"


@dataclass
class VoiceFeatures:
    f0_hz:      float    # fundamental frequency
    centroid:   float    # spectral centroid
    mfcc1:      float    # first MFCC coefficient
    mfcc2:      float    # second MFCC coefficient
    rms:        float    # frame energy


@dataclass
class VoiceProfile:
    """Running statistics for a speaker's voice features."""
    f0_mean:       float = 0.0
    f0_std:        float = 1.0
    centroid_mean: float = 0.0
    centroid_std:  float = 1.0
    mfcc1_mean:    float = 0.0
    mfcc1_std:     float = 1.0
    n_frames:      int   = 0

    def is_valid(self) -> bool:
        return self.n_frames >= ENROLLMENT_MIN_FRAMES and self.f0_std > 0.5


@dataclass
class ScenarioBResult:
    state:        DetectorState
    confidence:   float          # 0.0–1.0
    z_score:      float          # raw distance from profile
    is_flagged:   bool
    features:     VoiceFeatures | None = None
    frame_index:  int = 0


class TurnBasedSpeakerDetector:
    """
    Stateful detector — must process frames in order.
    Designed to receive only voiced frames (from VAD output).
    """

    def __init__(
        self,
        sample_rate:          int   = 16000,
        frame_size:           int   = 512,
        confidence_threshold: float = 0.75,
    ):
        self.sr                   = sample_rate
        self.frame_size           = frame_size
        self.confidence_threshold = confidence_threshold

        self._state               = DetectorState.ENROLLING
        self._reference           = VoiceProfile()
        self._enrollment_frames:  list[VoiceFeatures] = []
        self._current_segment:    list[VoiceFeatures] = []
        self._silence_counter     = 0
        self._frame_results:      list[ScenarioBResult] = []
        self._flag_events:        list[dict] = []
        self._frame_index         = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_voiced_frame(self, frame: np.ndarray) -> ScenarioBResult:
        """Call with every frame that passed VAD (speech-only frames)."""
        features = self._extract_features(frame)
        self._silence_counter = 0

        if self._state == DetectorState.ENROLLING:
            result = self._handle_enrollment(features)
        else:
            result = self._handle_monitoring(features)

        result.frame_index = self._frame_index
        self._frame_index += 1
        self._frame_results.append(result)
        return result

    def process_silence_frame(self) -> None:
        """
        Call for every frame where VAD found NO speech.
        Tracks silence gaps to detect turn boundaries.
        """
        self._silence_counter += 1

        if self._silence_counter >= MIN_SILENCE_GAP_FRAMES:
            if self._state == DetectorState.MONITORING:
                # End of a speech turn — prepare to compare next turn
                if len(self._current_segment) >= MIN_COMPARISON_FRAMES:
                    self._compare_segment_to_reference(self._current_segment)
                self._current_segment = []
                self._state = DetectorState.IN_SILENCE

    def get_flag_events(self) -> list[dict]:
        # Flush any open segment before returning
        if len(self._current_segment) >= MIN_COMPARISON_FRAMES:
            self._compare_segment_to_reference(self._current_segment)
            self._current_segment = []
        return self._flag_events

    def get_frame_results(self) -> list[ScenarioBResult]:
        return self._frame_results

    def get_state(self) -> DetectorState:
        return self._state

    def get_profile(self) -> VoiceProfile:
        return self._reference

    def reset(self):
        self.__init__(self.sr, self.frame_size, self.confidence_threshold)

    # ------------------------------------------------------------------ #
    #  State handlers                                                      #
    # ------------------------------------------------------------------ #

    def _handle_enrollment(self, features: VoiceFeatures) -> ScenarioBResult:
        self._enrollment_frames.append(features)

        if len(self._enrollment_frames) >= ENROLLMENT_MIN_FRAMES:
            self._build_reference_profile(self._enrollment_frames)
            self._state = DetectorState.MONITORING

        return ScenarioBResult(
            state=DetectorState.ENROLLING,
            confidence=0.0,
            z_score=0.0,
            is_flagged=False,
            features=features,
        )

    def _handle_monitoring(self, features: VoiceFeatures) -> ScenarioBResult:
        # Resume from silence → start accumulating new segment
        if self._state == DetectorState.IN_SILENCE:
            self._state = DetectorState.MONITORING
            self._current_segment = []

        self._current_segment.append(features)
        self._silence_counter = 0

        # Don't compare until we have enough frames in this segment
        if len(self._current_segment) < MIN_COMPARISON_FRAMES:
            return ScenarioBResult(
                state=DetectorState.MONITORING,
                confidence=0.0,
                z_score=0.0,
                is_flagged=False,
                features=features,
            )

        # Compare current frame's features to reference
        z_score   = self._compute_z_score(features)
        confidence = self._z_to_confidence(z_score)
        is_flagged = confidence >= self.confidence_threshold

        return ScenarioBResult(
            state=DetectorState.MONITORING,
            confidence=confidence,
            z_score=z_score,
            is_flagged=is_flagged,
            features=features,
        )

    # ------------------------------------------------------------------ #
    #  Profile management                                                  #
    # ------------------------------------------------------------------ #

    def _build_reference_profile(self, frames: list[VoiceFeatures]) -> None:
        f0s       = [f.f0_hz    for f in frames if f.f0_hz > 0]
        centroids = [f.centroid for f in frames]
        mfcc1s    = [f.mfcc1   for f in frames]

        if not f0s:
            return

        self._reference = VoiceProfile(
            f0_mean       = float(np.mean(f0s)),
            f0_std        = max(float(np.std(f0s)), 1.0),
            centroid_mean = float(np.mean(centroids)),
            centroid_std  = max(float(np.std(centroids)), 10.0),
            mfcc1_mean    = float(np.mean(mfcc1s)),
            mfcc1_std     = max(float(np.std(mfcc1s)), 0.5),
            n_frames      = len(frames),
        )

    def _update_reference_ema(self, features: VoiceFeatures) -> None:
        """Slow EMA drift — allows for voice changes due to fatigue."""
        if not self._reference.is_valid():
            return
        a = PROFILE_EMA_ALPHA
        self._reference.f0_mean       = (1-a)*self._reference.f0_mean       + a*features.f0_hz
        self._reference.centroid_mean = (1-a)*self._reference.centroid_mean + a*features.centroid

    def _compare_segment_to_reference(self, segment: list[VoiceFeatures]) -> None:
        """
        Compare completed segment's distribution to reference.
        Emits a flag event if distance is large.
        """
        if not self._reference.is_valid():
            return

        f0s       = [f.f0_hz    for f in segment if f.f0_hz > 0]
        centroids = [f.centroid for f in segment]

        if len(f0s) < 5:
            return

        seg_f0_mean  = float(np.mean(f0s))
        seg_cen_mean = float(np.mean(centroids))

        z_f0  = abs(seg_f0_mean - self._reference.f0_mean)  / self._reference.f0_std
        z_cen = abs(seg_cen_mean - self._reference.centroid_mean) / self._reference.centroid_std

        # Combined z-score (weighted — F0 is more reliable)
        z_combined = 0.65 * z_f0 + 0.35 * z_cen
        confidence = self._z_to_confidence(z_combined)

        if confidence >= self.confidence_threshold:
            start_frame = self._frame_index - len(self._current_segment)
            self._flag_events.append({
                "type":           "POSSIBLE_VOICE_CHANGE",
                "scenario":       "B",
                "start_s":        round(start_frame * (self.frame_size / self.sr), 3),
                "end_s":          round(self._frame_index * (self.frame_size / self.sr), 3),
                "duration_s":     round(len(segment) * (self.frame_size / self.sr), 3),
                "confidence":     round(confidence, 3),
                "z_score":        round(z_combined, 2),
                "z_f0":           round(z_f0, 2),
                "z_centroid":     round(z_cen, 2),
                "segment_f0_hz":  round(seg_f0_mean, 1),
                "reference_f0_hz": round(self._reference.f0_mean, 1),
            })

        # After confirmed candidate turns, slowly update reference
        elif z_combined < 2.0:
            self._update_reference_ema(VoiceFeatures(
                f0_hz=seg_f0_mean, centroid=seg_cen_mean,
                mfcc1=0, mfcc2=0, rms=0
            ))

    # ------------------------------------------------------------------ #
    #  Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def _extract_features(self, frame: np.ndarray) -> VoiceFeatures:
        fft_mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs   = np.fft.rfftfreq(len(frame), d=1.0 / self.sr)

        f0       = self._extract_f0(frame)
        centroid = self._spectral_centroid(fft_mag, freqs)
        mfcc1, mfcc2 = self._extract_mfcc12(fft_mag, freqs)
        rms      = float(np.sqrt(np.mean(frame ** 2)))

        return VoiceFeatures(
            f0_hz=f0, centroid=centroid,
            mfcc1=mfcc1, mfcc2=mfcc2, rms=rms
        )

    def _extract_f0(self, frame: np.ndarray) -> float:
        """Autocorrelation-based F0 estimation."""
        autocorr = np.correlate(frame, frame, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        min_lag = int(self.sr / 400)   # 400Hz max
        max_lag = int(self.sr / 80)    # 80Hz min
        max_lag = min(max_lag, len(autocorr) - 1)

        if min_lag >= max_lag:
            return 0.0

        peak_lag = int(np.argmax(autocorr[min_lag:max_lag])) + min_lag
        if autocorr[peak_lag] < 0.3:    # weak periodicity = unvoiced
            return 0.0

        return float(self.sr / peak_lag)

    @staticmethod
    def _spectral_centroid(fft_mag: np.ndarray, freqs: np.ndarray) -> float:
        energy = fft_mag ** 2
        total  = np.sum(energy) + 1e-10
        return float(np.sum(freqs * energy) / total)

    def _extract_mfcc12(self, fft_mag: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
        """
        Lightweight 2-coefficient MFCC approximation using mel filterbank.
        Full 13-coeff MFCC lives in Stage 3. Here we just need timbral shape.
        """
        n_filters = 12
        mel_low  = self._hz_to_mel(80)
        mel_high = self._hz_to_mel(min(self.sr / 2, 4000))
        mel_points = np.linspace(mel_low, mel_high, n_filters + 2)
        hz_points  = self._mel_to_hz(mel_points)

        filterbank_energies = np.zeros(n_filters)
        for i in range(n_filters):
            lo, center, hi = hz_points[i], hz_points[i+1], hz_points[i+2]
            for j, f in enumerate(freqs):
                if lo <= f <= center:
                    filterbank_energies[i] += fft_mag[j] ** 2 * (f - lo) / (center - lo + 1e-10)
                elif center < f <= hi:
                    filterbank_energies[i] += fft_mag[j] ** 2 * (hi - f) / (hi - center + 1e-10)

        log_energies = np.log(filterbank_energies + 1e-10)
        # DCT first 2 coefficients
        mfcc1 = float(np.sum(log_energies * np.cos(np.pi * 1 * (np.arange(n_filters) + 0.5) / n_filters)))
        mfcc2 = float(np.sum(log_energies * np.cos(np.pi * 2 * (np.arange(n_filters) + 0.5) / n_filters)))
        return mfcc1, mfcc2

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # ------------------------------------------------------------------ #
    #  Scoring                                                             #
    # ------------------------------------------------------------------ #

    def _compute_z_score(self, features: VoiceFeatures) -> float:
        """Per-frame z-score distance from reference profile."""
        if not self._reference.is_valid():
            return 0.0
        z_f0  = abs(features.f0_hz - self._reference.f0_mean) / self._reference.f0_std if features.f0_hz > 0 else 0.0
        z_cen = abs(features.centroid - self._reference.centroid_mean) / self._reference.centroid_std
        return float(0.65 * z_f0 + 0.35 * z_cen)

    @staticmethod
    def _z_to_confidence(z: float) -> float:
        """
        Map z-score distance to 0–1 confidence.
        < 4 SD  → below 0.5 (candidate's own variation)
          7 SD  → 0.75 (flag threshold)
         10 SD  → 1.0  (certain)
        """
        if z < THRESHOLD_SUSPICIOUS:
            return float(np.clip(z / (THRESHOLD_SUSPICIOUS * 2), 0.0, 0.49))
        return float(np.clip(
            0.5 + (z - THRESHOLD_SUSPICIOUS) / ((THRESHOLD_FLAG - THRESHOLD_SUSPICIOUS) * 2),
            0.5, 1.0
        ))
