"""
ml_proctoring.embedder
======================
Resemblyzer-based speaker embedder.
Returns 256-dim L2-normalised d-vectors trained with GE2E loss
on 1M+ utterances — discriminates speakers that share similar F0.

Lazy-loads VoiceEncoder on first call so session startup is instant.
"""
from __future__ import annotations

import numpy as np

_EMBED_DIM      = 256
_MIN_AUDIO_S    = 0.5    # Resemblyzer minimum for a stable embedding
_ENROLL_CHUNK_S = 3.0    # split enrollment into N-second pieces


class Embedder:
    """
    Wraps Resemblyzer's VoiceEncoder.

    embed(audio, sr)  → 256-dim float32 array
    embed_enrollment(segments, sr) → centroid of all segment embeddings
    """

    def __init__(self):
        self._encoder = None   # lazy

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._encoder is not None:
            return
        try:
            from resemblyzer import VoiceEncoder
        except ImportError:
            raise ImportError(
                "resemblyzer is required for ml_proctoring: pip install resemblyzer"
            )
        self._encoder = VoiceEncoder(device="cpu", verbose=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
        """
        Embed a single audio segment.

        Returns a 256-dim L2-normalised float32 array,
        or None if the segment is too short / silent.
        """
        self._ensure_loaded()
        from resemblyzer import preprocess_wav

        if len(audio) / sr < _MIN_AUDIO_S:
            return None

        try:
            wav = preprocess_wav(audio, source_sr=sr)
            if len(wav) < int(_MIN_AUDIO_S * 16000):
                return None
            emb = self._encoder.embed_utterance(wav)
            return emb.astype(np.float32)
        except Exception:
            return None

    def embed_enrollment(
        self, audio: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """
        Build an enrollment centroid from a longer recording.

        Splits audio into _ENROLL_CHUNK_S-second pieces, embeds each,
        and returns the mean of L2-normalised embeddings (re-normalised).

        Raises ValueError if no valid embeddings could be extracted.
        """
        self._ensure_loaded()
        chunk_n = int(_ENROLL_CHUNK_S * sr)
        embeds  = []

        for start in range(0, len(audio) - chunk_n + 1, chunk_n):
            seg = audio[start : start + chunk_n]
            emb = self.embed(seg, sr)
            if emb is not None:
                embeds.append(emb)

        # Also try the whole clip if it's < 1 chunk
        if not embeds:
            emb = self.embed(audio, sr)
            if emb is not None:
                embeds.append(emb)

        if not embeds:
            raise ValueError(
                "Could not extract any speaker embedding from enrollment audio. "
                "Ensure the recording contains at least 0.5s of clear speech."
            )

        centroid = np.mean(embeds, axis=0).astype(np.float32)
        n = np.linalg.norm(centroid)
        return centroid / n if n > 1e-8 else centroid
