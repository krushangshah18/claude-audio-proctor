"""
ml_proctoring.audio_buffer
==========================
Rolling circular buffer — holds the last N seconds of raw audio.
On any cheat flag, call to_wav_bytes() to extract the evidence window.
"""
from __future__ import annotations

import numpy as np
from .audio_utils import audio_to_wav_bytes


class RollingAudioBuffer:
    """
    Circular float32 audio buffer.

    Capacity = sr * max_seconds samples.
    After the buffer is full, oldest samples are overwritten.
    """

    def __init__(self, sr: int = 16000, max_seconds: float = 30.0):
        self.sr = sr
        self._capacity = int(sr * max_seconds)
        self._buf  = np.zeros(self._capacity, dtype=np.float32)
        self._pos  = 0       # write head
        self._full = False   # becomes True once buffer wraps once

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(self, chunk: np.ndarray) -> None:
        """Append chunk to the rolling buffer (overwrites oldest if full)."""
        n   = len(chunk)
        end = self._pos + n
        if end <= self._capacity:
            self._buf[self._pos:end] = chunk
        else:
            first = self._capacity - self._pos
            self._buf[self._pos:]    = chunk[:first]
            self._buf[:end - self._capacity] = chunk[first:]
        self._pos = end % self._capacity
        if end >= self._capacity:
            self._full = True

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_window(self, seconds: float | None = None) -> np.ndarray:
        """
        Return a contiguous copy of the most recent `seconds` of audio.
        If seconds is None, returns the entire available buffer.
        """
        if self._full:
            # Linearise: from write head to write head (oldest → newest)
            linear = np.concatenate([self._buf[self._pos:], self._buf[:self._pos]])
        else:
            linear = self._buf[:self._pos].copy()

        if seconds is None or seconds * self.sr >= len(linear):
            return linear.copy()
        n = int(seconds * self.sr)
        return linear[-n:].copy()

    def to_wav_bytes(self, seconds: float | None = None) -> bytes:
        """Extract evidence window and return as WAV bytes for backend upload."""
        return audio_to_wav_bytes(self.get_window(seconds), self.sr)

    # ── State ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos    = 0
        self._full   = False

    @property
    def duration_s(self) -> float:
        """Seconds of audio currently in the buffer."""
        if self._full:
            return self._capacity / self.sr
        return self._pos / self.sr
