"""
VAD Engine — Stage 1
====================
Primary:  Silero VAD (silero-vad package, v5+)
          - Trained on 6000+ languages, 6000h+ of audio
          - ~1ms per 30ms chunk on CPU
          - VADIterator for streaming-style per-frame output with internal state
          - get_speech_timestamps for batch processing with full heuristics

Fallback: Signal-processing VAD (no model needed)
          - Used automatically if silero-vad / torch not installed

Architecture:
  Audio (any SR/format)
      → resample to 16kHz mono float32
      → Silero VADIterator  (512 samples / 32ms per chunk)
      → Hysteresis smoother (onset/offset counters)
      → SpeechSegment list + per-frame metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000          # Silero only accepts 8000 or 16000
CHUNK_SAMPLES = 512          # Silero v5 window: 512 samples @ 16kHz = 32ms


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VADConfig:
    sample_rate: int = SAMPLE_RATE

    # Silero threshold: probability above which a chunk is considered speech
    # 0.5 = default (good for most cases)
    # Lower (0.3-0.4) = more sensitive, catches quiet/whisper speech
    # Higher (0.6-0.7) = stricter, fewer false positives
    silero_threshold: float = 0.45

    # Hysteresis — prevents rapid on/off toggling
    # onset:  N consecutive speech chunks needed to START a segment (~frames × 32ms)
    # offset: N consecutive silence chunks needed to END a segment
    onset_chunks: int = 3     # 3 × 32ms = ~100ms confirmation lag
    offset_chunks: int = 10   # 10 × 32ms = ~320ms silence before ending segment

    # Post-processing
    min_speech_ms: int = 250          # drop segments shorter than this
    min_silence_between_ms: int = 150 # merge segments closer than this
    speech_pad_ms: int = 80           # pad each segment start/end

    # Calibration: first N seconds used to estimate noise floor for whisper detection
    calib_seconds: float = 3.0


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float
    avg_rms: float = 0.0
    avg_prob: float = 0.0      # avg Silero speech probability
    is_whisper: bool = False
    noise_floor_at_start: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load any audio file to float32 mono at target_sr.
    Uses soundfile (primary) → scipy.io.wavfile (fallback).
    Handles stereo → mono, resampling, int16/int32/float normalization.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32', always_2d=False)
    except Exception:
        import scipy.io.wavfile as wav_io
        sr, audio = wav_io.read(path)
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio /= 32768.0

    # Stereo → mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

    # Normalize to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio /= peak

    return audio


def save_audio(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE):
    """Save float32 audio as 16-bit PCM WAV."""
    try:
        import soundfile as sf
        sf.write(path, audio, sr, subtype='PCM_16')
    except Exception:
        import scipy.io.wavfile as wav_io
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        wav_io.write(path, sr, audio_int16)


# ─────────────────────────────────────────────────────────────────────────────
# Noise floor tracker (for whisper labelling only — Silero handles actual VAD)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseFloor:
    """
    Tracks background noise RMS during silence periods.
    Used only to label whisper segments — Silero handles actual detection.
    """

    def __init__(self, calib_seconds: float = 3.0, sr: int = SAMPLE_RATE):
        self._calib_chunks = int(calib_seconds * sr / CHUNK_SAMPLES)
        self._calib_buf: list[float] = []
        self.rms: float = 0.005
        self.calibrated: bool = False
        self._alpha = 0.002   # very slow tracking

    def update(self, chunk: np.ndarray, is_speech: bool):
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if not self.calibrated:
            self._calib_buf.append(rms)
            if len(self._calib_buf) >= self._calib_chunks:
                self.rms = float(np.percentile(self._calib_buf, 20))
                self.rms = max(self.rms, 1e-5)
                self.calibrated = True
        elif not is_speech:
            self.rms += self._alpha * (rms - self.rms)
            self.rms = max(self.rms, 1e-5)

    def is_whisper(self, avg_rms: float) -> bool:
        return avg_rms < self.rms * 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Silero VAD engine
# ─────────────────────────────────────────────────────────────────────────────

class SileroVADEngine:
    """
    Wraps Silero VADIterator for streaming-style per-chunk processing.

    Silero VADIterator feeds 512-sample chunks and maintains internal RNN state.
    We add our own onset/offset hysteresis on top for cleaner segment boundaries.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._model = None
        self._iterator = None
        self._load_silero()
        self._noise_floor = NoiseFloor(self.config.calib_seconds)
        self._reset_state()

    def _load_silero(self):
        try:
            import torch
            torch.set_num_threads(1)  # single thread — critical for browser parity
            from silero_vad import load_silero_vad, VADIterator
            self._model = load_silero_vad()
            self._model.eval()
            self._VADIterator = VADIterator
            logger.info("Silero VAD loaded successfully")
        except ImportError as e:
            raise ImportError(
                f"silero-vad not installed: {e}\n"
                "Run:  uv add silero-vad torch torchaudio soundfile\n"
                "Or:   pip install silero-vad torch torchaudio soundfile"
            )

    def _reset_state(self):
        self._speech_counter = 0     # consecutive speech chunks
        self._silence_counter = 0    # consecutive silence chunks
        self._is_speech = False      # current VAD state (after hysteresis)
        self._seg_start: Optional[float] = None
        self._seg_probs: list[float] = []
        self._seg_rms: list[float] = []
        self._segments: list[SpeechSegment] = []
        self._frame_results: list[dict] = []
        if self._model is not None:
            self._model.reset_states()

    # ── Public API ────────────────────────────────────────────────────────────

    def process_audio(self, audio: np.ndarray) -> list[dict]:
        """
        Process a full audio array.
        Returns per-chunk result list (for visualization / reporting).
        """
        self._reset_state()
        import torch

        n_chunks = len(audio) // CHUNK_SAMPLES
        sr = self.config.sample_rate

        for i in range(n_chunks):
            chunk = audio[i * CHUNK_SAMPLES: (i + 1) * CHUNK_SAMPLES]
            ts = i * CHUNK_SAMPLES / sr

            # Silero forward pass: returns speech probability 0–1
            tensor = torch.from_numpy(chunk).unsqueeze(0)  # [1, 512]
            with torch.no_grad():
                prob = float(self._model(tensor, sr).item())

            result = self._process_chunk(chunk, prob, ts)
            self._frame_results.append(result)

        self._finalize(n_chunks * CHUNK_SAMPLES / sr)
        self._post_process()
        return self._frame_results

    def get_segments(self) -> list[SpeechSegment]:
        return self._segments.copy()

    # ── Internal processing ───────────────────────────────────────────────────

    def _process_chunk(self, chunk: np.ndarray, prob: float, ts: float) -> dict:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        prev_is_speech = self._is_speech

        # Hysteresis state machine on top of Silero probabilities
        if prob >= self.config.silero_threshold:
            self._speech_counter += 1
            self._silence_counter = 0
            if self._speech_counter >= self.config.onset_chunks:
                self._is_speech = True
        else:
            self._silence_counter += 1
            self._speech_counter = 0
            if self._silence_counter >= self.config.offset_chunks:
                self._is_speech = False

        # Segment tracking
        if self._is_speech and not prev_is_speech:
            self._seg_start = ts
            self._seg_probs = []
            self._seg_rms = []

        if self._is_speech:
            self._seg_probs.append(prob)
            self._seg_rms.append(rms)

        if not self._is_speech and prev_is_speech:
            self._close_segment(ts)

        self._noise_floor.update(chunk, self._is_speech)

        return {
            "timestamp": ts,
            "is_speech": self._is_speech,
            "speech_prob": prob,
            "rms": rms,
            "noise_floor": self._noise_floor.rms,
            "state": "speech" if self._is_speech else "silence",
        }

    def _close_segment(self, end_ts: float):
        if self._seg_start is None:
            return
        avg_rms = float(np.mean(self._seg_rms)) if self._seg_rms else 0.0
        avg_prob = float(np.mean(self._seg_probs)) if self._seg_probs else 0.0
        duration_ms = (end_ts - self._seg_start) * 1000

        if duration_ms >= self.config.min_speech_ms:
            self._segments.append(SpeechSegment(
                start_s=self._seg_start,
                end_s=end_ts,
                avg_rms=avg_rms,
                avg_prob=avg_prob,
                is_whisper=self._noise_floor.is_whisper(avg_rms),
                noise_floor_at_start=self._noise_floor.rms,
            ))
        self._seg_start = None

    def _finalize(self, end_ts: float):
        """Close any open segment at end of audio."""
        if self._is_speech and self._seg_start is not None:
            self._close_segment(end_ts)

    def _post_process(self):
        """
        Merge segments closer than min_silence_between_ms.
        Apply speech_pad_ms to each segment boundary.
        """
        if not self._segments:
            return

        min_gap = self.config.min_silence_between_ms / 1000
        pad = self.config.speech_pad_ms / 1000

        # 1. Merge segments with small gaps
        merged: list[SpeechSegment] = [self._segments[0]]
        for seg in self._segments[1:]:
            prev = merged[-1]
            if seg.start_s - prev.end_s < min_gap:
                # Merge: extend previous segment
                n_prev = max(1, round(prev.duration_s / (CHUNK_SAMPLES / SAMPLE_RATE)))
                n_curr = max(1, round(seg.duration_s / (CHUNK_SAMPLES / SAMPLE_RATE)))
                total = n_prev + n_curr
                merged[-1] = SpeechSegment(
                    start_s=prev.start_s,
                    end_s=seg.end_s,
                    avg_rms=(prev.avg_rms * n_prev + seg.avg_rms * n_curr) / total,
                    avg_prob=(prev.avg_prob * n_prev + seg.avg_prob * n_curr) / total,
                    is_whisper=prev.is_whisper and seg.is_whisper,
                    noise_floor_at_start=prev.noise_floor_at_start,
                )
            else:
                merged.append(seg)

        # 2. Apply padding
        for seg in merged:
            seg.start_s = max(0.0, seg.start_s - pad)
            seg.end_s = seg.end_s + pad

        # 3. Re-label whisper based on merged segment RMS
        for seg in merged:
            seg.is_whisper = self._noise_floor.is_whisper(seg.avg_rms)

        self._segments = merged


# ─────────────────────────────────────────────────────────────────────────────
# Signal-processing fallback (no torch/silero needed)
# ─────────────────────────────────────────────────────────────────────────────

class FallbackVADEngine:
    """
    Pure signal-processing VAD for environments without torch.
    Uses spectral flatness, band energy, HNR, and adaptive noise floor.
    Same interface as SileroVADEngine.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._noise_floor = NoiseFloor(self.config.calib_seconds)
        self._segments: list[SpeechSegment] = []
        self._frame_results: list[dict] = []
        logger.warning("Using fallback signal-processing VAD (Silero not available)")

    def _score_chunk(self, chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < self._noise_floor.rms * 2.0:
            return 0.0

        # Spectral flatness
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)))) + 1e-10
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))

        # Band energy ratio (200–3400 Hz = speech band)
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)
        band = (freqs >= 200) & (freqs <= 3400)
        band_ratio = float(np.sum(spectrum[band] ** 2) / (np.sum(spectrum ** 2) + 1e-10))

        # Single-tone detection (car horn killer)
        band_spec = spectrum[band] ** 2
        peak_ratio = float(np.max(band_spec) / (np.sum(band_spec) + 1e-10))

        if flatness > 0.65 or band_ratio < 0.08 or peak_ratio > 0.55:
            return 0.0

        energy_score = min(1.0, rms / (self._noise_floor.rms * 4 + 1e-10))
        flatness_score = max(0.0, 1.0 - flatness / 0.65)
        band_score = min(1.0, band_ratio / 0.45)
        return float(0.45 * energy_score + 0.30 * flatness_score + 0.25 * band_score)

    def process_audio(self, audio: np.ndarray) -> list[dict]:
        speech_counter = 0
        silence_counter = 0
        is_speech = False
        seg_start = None
        seg_rms: list[float] = []
        seg_probs: list[float] = []
        results: list[dict] = []

        n_chunks = len(audio) // CHUNK_SAMPLES
        onset = self.config.onset_chunks
        offset = self.config.offset_chunks
        threshold = self.config.silero_threshold

        for i in range(n_chunks):
            chunk = audio[i * CHUNK_SAMPLES: (i + 1) * CHUNK_SAMPLES]
            ts = i * CHUNK_SAMPLES / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            score = self._score_chunk(chunk)
            prev = is_speech

            if score >= threshold:
                speech_counter += 1
                silence_counter = 0
                if speech_counter >= onset:
                    is_speech = True
            else:
                silence_counter += 1
                speech_counter = 0
                if silence_counter >= offset:
                    is_speech = False

            if is_speech and not prev:
                seg_start = ts
                seg_rms, seg_probs = [], []
            if is_speech:
                seg_rms.append(rms)
                seg_probs.append(score)
            if not is_speech and prev and seg_start is not None:
                dur_ms = (ts - seg_start) * 1000
                if dur_ms >= self.config.min_speech_ms:
                    avg_rms = float(np.mean(seg_rms))
                    self._segments.append(SpeechSegment(
                        start_s=max(0.0, seg_start - self.config.speech_pad_ms / 1000),
                        end_s=ts + self.config.speech_pad_ms / 1000,
                        avg_rms=avg_rms,
                        avg_prob=float(np.mean(seg_probs)),
                        is_whisper=self._noise_floor.is_whisper(avg_rms),
                        noise_floor_at_start=self._noise_floor.rms,
                    ))
                seg_start = None

            self._noise_floor.update(chunk, is_speech)
            results.append({
                "timestamp": ts,
                "is_speech": is_speech,
                "speech_prob": score,
                "rms": rms,
                "noise_floor": self._noise_floor.rms,
                "state": "speech" if is_speech else "silence",
            })

        self._frame_results = results
        return results

    def get_segments(self) -> list[SpeechSegment]:
        return self._segments.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Factory: auto-select best available engine
# ─────────────────────────────────────────────────────────────────────────────

def create_vad_engine(config: Optional[VADConfig] = None) -> SileroVADEngine | FallbackVADEngine:
    """
    Returns SileroVADEngine if silero-vad + torch are available,
    otherwise returns FallbackVADEngine with a warning.
    """
    try:
        engine = SileroVADEngine(config)
        return engine
    except ImportError as e:
        logger.warning(str(e))
        return FallbackVADEngine(config)
