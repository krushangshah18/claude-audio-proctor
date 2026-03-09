"""
ml_proctoring.vad_bridge
========================
Thin wrapper around Silero VAD for streaming (chunk-by-chunk) use.
Matches the hysteresis logic from core/vad_engine.py but is
completely independent — no imports from core/.
"""
from __future__ import annotations

import numpy as np

_ONSET_CHUNKS  = 3    # consecutive speech chunks to START a segment (~96ms)
_OFFSET_CHUNKS = 10   # consecutive silence chunks to END a segment (~320ms)
_CHUNK_SAMPLES = 512  # Silero v5 @ 16kHz


class VADBridge:
    """
    Streaming VAD using Silero + hysteresis smoothing.

    Usage:
        vad = VADBridge(threshold=0.45)
        speaking = vad.is_speech(chunk_512_samples)
    """

    def __init__(self, threshold: float = 0.45, sample_rate: int = 16000):
        self._threshold  = threshold
        self._sr         = sample_rate
        self._model      = None
        self._iterator   = None
        self._onset_cnt  = 0
        self._offset_cnt = 0
        self._active     = False   # hysteresis state

    # ── Lazy model load ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad, VADIterator
        except ImportError:
            raise ImportError(
                "silero-vad is required: pip install silero-vad"
            )
        self._model    = load_silero_vad()
        self._iterator = VADIterator(
            self._model,
            threshold=self._threshold,
            sampling_rate=self._sr,
            min_silence_duration_ms=0,   # we handle hysteresis ourselves
            speech_pad_ms=0,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Process one 512-sample chunk. Returns True when the hysteresis
        state machine says we are inside a speech segment.

        Shorter/longer chunks are zero-padded / truncated to 512 samples.
        """
        self._ensure_loaded()

        # Pad/trim to exactly 512 samples
        if len(chunk) < _CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES - len(chunk)))
        else:
            chunk = chunk[:_CHUNK_SAMPLES]

        out = self._iterator(chunk.astype(np.float32), return_seconds=False)
        raw_speech = out is not None and "start" in out

        # Re-check probability directly for smoother hysteresis
        import torch
        with torch.no_grad():
            prob = float(self._model(
                torch.from_numpy(chunk.astype(np.float32)), self._sr
            ).item())
        raw_speech = prob >= self._threshold

        # Hysteresis
        if raw_speech:
            self._onset_cnt  += 1
            self._offset_cnt  = 0
            if self._onset_cnt >= _ONSET_CHUNKS:
                self._active = True
        else:
            self._offset_cnt += 1
            self._onset_cnt   = 0
            if self._offset_cnt >= _OFFSET_CHUNKS:
                self._active = False

        return self._active

    def reset(self) -> None:
        """Reset all state (call between exam sessions)."""
        self._onset_cnt  = 0
        self._offset_cnt = 0
        self._active     = False
        if self._iterator is not None:
            self._iterator.reset_states()
