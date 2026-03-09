"""
ml_proctoring.session
=====================
ProctorSession — the single public entry point for the video pipeline.

Typical lifecycle:
    session = ProctorSession()
    session.enroll(enrollment_audio_float32)   # once before exam

    # per audio chunk from mic thread:
    events = session.push(audio_chunk, lip_activity)
    for ev in events:
        backend.upload(ev.audio_proof, ev.event_type)

    summary = session.get_summary()
    session.reset()                            # between exam sessions

Thread safety:
    update_lip_activity() may be called from the video thread.
    push() should be called from the audio thread.
    A threading.Lock guards the shared _lip_active flag.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from .audio_buffer  import RollingAudioBuffer
from .audio_utils   import ensure_float32_mono, resample
from .embedder      import Embedder
from .fusion        import CrossModalFusion
from .models        import CheatEvent, FusionInputs, SpeakerProfile
from .speaker_tracker import OnlineSpeakerTracker
from .vad_bridge    import VADBridge

# ── Tuneable constants ────────────────────────────────────────────────────────
_SR              = 16000
_CHUNK_SAMPLES   = 512       # 32ms — matches Silero requirement
_EMBED_VOICED_S  = 2.0       # accumulate this many seconds of voiced audio before embedding
_EMBED_VOICED_N  = int(_EMBED_VOICED_S * _SR)
_BUFFER_S        = 30.0      # rolling evidence buffer duration


class ProctorSession:
    """
    Online audio proctoring session.

    All AI/ML components are lazy-loaded on first push() call
    so instantiation is instant.
    """

    def __init__(
        self,
        sample_rate: int = _SR,
        buffer_seconds: float = _BUFFER_S,
        vad_threshold: float = 0.45,
        speaker_merge_threshold: float = 0.75,
    ):
        self._sr = sample_rate

        # Subsystems
        self._buffer  = RollingAudioBuffer(sr=sample_rate, max_seconds=buffer_seconds)
        self._vad     = VADBridge(threshold=vad_threshold, sample_rate=sample_rate)
        self._embedder = Embedder()
        self._tracker  = OnlineSpeakerTracker(merge_threshold=speaker_merge_threshold)
        self._fusion   = CrossModalFusion()

        # Enrollment
        self._profile: Optional[SpeakerProfile] = None

        # Per-push state
        self._voiced_accum: list[np.ndarray] = []   # voiced chunks waiting to embed
        self._voiced_accum_n: int = 0               # total samples accumulated
        self._last_verify_score: float = 0.0
        self._chunk_count:  int = 0
        self._start_time:   float | None = None
        self._total_speech_chunks: int = 0

        # Cross-thread lip activity state
        self._lip_lock   = threading.Lock()
        self._lip_active: bool = False

        # Event log for summary
        self._events: list[CheatEvent] = []
        self._verify_scores: list[float] = []

    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll(self, audio: np.ndarray, sr: int | None = None) -> None:
        """
        Enroll the reference speaker.

        Parameters
        ----------
        audio : float32 mono numpy array (or int16 — will be converted)
        sr    : sample rate of `audio`. Defaults to session sample_rate.
                Will resample if sr ≠ session sample_rate.

        Raises
        ------
        ValueError if audio is too short or contains no voiced frames.
        """
        src_sr = sr or self._sr
        audio  = ensure_float32_mono(audio)
        if src_sr != self._sr:
            audio = resample(audio, src_sr, self._sr)

        if len(audio) / self._sr < 0.5:
            raise ValueError("Enrollment audio too short (minimum 0.5s required).")

        centroid = self._embedder.embed_enrollment(audio, self._sr)
        self._profile = SpeakerProfile(centroid=centroid, n_segments=1)

    @property
    def is_enrolled(self) -> bool:
        return self._profile is not None

    # ── Cross-thread lip activity update (called from video thread) ───────────

    def update_lip_activity(self, lips_moving: bool) -> None:
        """
        Update the current lip-activity state.
        Safe to call from the video pipeline thread.
        """
        with self._lip_lock:
            self._lip_active = lips_moving

    # ── Main hot path ─────────────────────────────────────────────────────────

    def push(
        self,
        audio_chunk: np.ndarray,
        lip_activity: bool | None = None,
    ) -> list[CheatEvent]:
        """
        Process one audio chunk from the live stream.

        Parameters
        ----------
        audio_chunk  : float32 mono array, ideally 512 samples (32ms @ 16kHz).
                       Other sizes are accepted but may reduce VAD accuracy.
        lip_activity : bool or None. If provided, overrides the last value set
                       via update_lip_activity(). Pass None to use the last
                       value set from the video thread.

        Returns
        -------
        list[CheatEvent] — empty most of the time; each event carries
                           .audio_proof (WAV bytes) ready for backend upload.
        """
        if self._start_time is None:
            self._start_time = time.monotonic()

        chunk = ensure_float32_mono(audio_chunk)

        # 1. Feed rolling evidence buffer
        self._buffer.push(chunk)

        # 2. VAD
        speaking = self._vad.is_speech(chunk)
        if speaking:
            self._total_speech_chunks += 1
            self._voiced_accum.append(chunk)
            self._voiced_accum_n += len(chunk)

        # 3. Embed when we have enough voiced audio
        if self._voiced_accum_n >= _EMBED_VOICED_N and self._profile is not None:
            voiced_seg = np.concatenate(self._voiced_accum)
            self._voiced_accum.clear()
            self._voiced_accum_n = 0

            emb = self._embedder.embed(voiced_seg, self._sr)
            if emb is not None:
                self._tracker.update(emb)
                score = self._tracker.verify_score(self._profile.centroid)
                self._last_verify_score = score
                self._verify_scores.append(score)

        # 4. Read lip state (caller override takes priority)
        if lip_activity is not None:
            with self._lip_lock:
                self._lip_active = lip_activity
        with self._lip_lock:
            lip_now = self._lip_active

        # 5. Cross-modal fusion
        self._chunk_count += 1
        events: list[CheatEvent] = []

        if self._profile is not None:
            ts = time.monotonic() - self._start_time
            inputs = FusionInputs(
                vad_active   = speaking,
                lip_active   = lip_now,
                verify_score = self._last_verify_score,
                n_speakers   = self._tracker.n_speakers,
                timestamp_s  = ts,
            )
            ev = self._fusion.evaluate(inputs, self._buffer)
            if ev is not None:
                self._events.append(ev)
                events.append(ev)

        return events

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a summary dict covering the full session."""
        return {
            "total_chunks_processed" : self._chunk_count,
            "total_speech_s"         : round(
                self._total_speech_chunks * _CHUNK_SAMPLES / self._sr, 2
            ),
            "n_speakers_detected"    : self._tracker.n_speakers,
            "cheat_events"           : list(self._events),
            "verify_score_mean"      : round(float(np.mean(self._verify_scores)), 4)
                                       if self._verify_scores else 0.0,
            "verify_score_min"       : round(float(np.min(self._verify_scores)), 4)
                                       if self._verify_scores else 0.0,
            "enrolled"               : self.is_enrolled,
        }

    # ── Session reset (preserve enrollment) ───────────────────────────────────

    def reset(self) -> None:
        """
        Clear all session state. Enrolled speaker centroid is preserved —
        no need to re-enroll for a new exam session for the same student.
        """
        self._buffer.reset()
        self._vad.reset()
        self._tracker.reset()
        self._fusion.reset()
        self._voiced_accum.clear()
        self._voiced_accum_n   = 0
        self._last_verify_score = 0.0
        self._chunk_count      = 0
        self._start_time       = None
        self._total_speech_chunks = 0
        self._events.clear()
        self._verify_scores.clear()
        with self._lip_lock:
            self._lip_active = False
