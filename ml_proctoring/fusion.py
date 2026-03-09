"""
ml_proctoring.fusion
====================
Cross-modal decision engine.

Inputs per evaluation cycle:
  vad_active    — is anyone speaking right now?
  lip_active    — is the enrolled student's lip movement detected?
  verify_score  — cosine similarity of current voice to enrollment (0-1)
  n_speakers    — distinct speaker clusters detected this session
  timestamp_s   — seconds elapsed since session start

Output: CheatEvent | None

Thresholds are module-level constants — tune here, logic stays unchanged.
Cooldown prevents the same event type from flooding the backend.
"""
from __future__ import annotations

from .models import CheatEvent, CheatType, FusionInputs
from .audio_buffer import RollingAudioBuffer

# ── Verification thresholds (cosine similarity) ───────────────────────────────
THRESH_MATCH     = 0.72   # ≥ → confident match with enrolled speaker
THRESH_LIKELY    = 0.55   # ≥ → probable match
THRESH_UNCERTAIN = 0.40   # ≥ → borderline
# < THRESH_UNCERTAIN → MISMATCH

# ── Fusion decision thresholds ────────────────────────────────────────────────
# Minimum cooldown (evaluation cycles) between two flags of the same type.
# At 512-sample chunks @ 16kHz: 1 cycle ≈ 32ms → 50 cycles ≈ 1.6s
_COOLDOWN_CYCLES: dict[CheatType, int] = {
    CheatType.IMPERSONATION  : 100,   # ~3.2s
    CheatType.GHOST_VOICE    : 100,
    CheatType.EXTRA_SPEAKER  : 150,   # ~4.8s
    CheatType.VOICE_MISMATCH : 80,    # ~2.6s
}

# Minimum verify_score before we trust a "lips moving but wrong voice" flag.
# Avoids false IMPERSONATION flags during the first ~2s when no embedding exists.
_MIN_SCORE_FOR_VOICE_FLAG = 0.05


class CrossModalFusion:
    """
    Stateful fusion engine.
    Maintains per-CheatType cooldown counters to suppress flood events.
    """

    def __init__(self) -> None:
        self._cooldowns: dict[CheatType, int] = {t: 0 for t in CheatType}

    def evaluate(
        self,
        inputs: FusionInputs,
        buffer: RollingAudioBuffer,
    ) -> CheatEvent | None:
        """
        Run the decision tree for one evaluation cycle.
        Returns a CheatEvent (with WAV proof attached) or None.
        """
        # Tick down all cooldowns
        for t in self._cooldowns:
            if self._cooldowns[t] > 0:
                self._cooldowns[t] -= 1

        event = self._decide(inputs)
        if event is None:
            return None

        # Check cooldown
        if self._cooldowns[event] > 0:
            return None

        # Arm cooldown for this type
        self._cooldowns[event] = _COOLDOWN_CYCLES[event]

        # Extract audio proof from buffer
        proof = buffer.to_wav_bytes()

        confidence = self._confidence(event, inputs)
        return CheatEvent(
            event_type   = event,
            confidence   = round(confidence, 3),
            timestamp_s  = inputs.timestamp_s,
            verify_score = round(inputs.verify_score, 4),
            n_speakers   = inputs.n_speakers,
            lip_active   = inputs.lip_active,
            audio_proof  = proof,
        )

    def reset(self) -> None:
        for t in self._cooldowns:
            self._cooldowns[t] = 0

    # ── Decision tree (priority order) ────────────────────────────────────────

    def _decide(self, inp: FusionInputs) -> CheatType | None:
        score_available = inp.verify_score >= _MIN_SCORE_FOR_VOICE_FLAG

        # 1. IMPERSONATION — voice heard, lips still, voice ≠ enrolled
        if (inp.vad_active
                and not inp.lip_active
                and score_available
                and inp.verify_score < THRESH_UNCERTAIN):
            return CheatType.IMPERSONATION

        # 2. GHOST_VOICE — enrolled voice heard but lips not moving
        #    (pre-recorded audio playback trick)
        if (inp.vad_active
                and not inp.lip_active
                and score_available
                and inp.verify_score >= THRESH_MATCH):
            return CheatType.GHOST_VOICE

        # 3. EXTRA_SPEAKER — two or more distinct voices detected in session
        if inp.vad_active and inp.n_speakers >= 2:
            return CheatType.EXTRA_SPEAKER

        # 4. VOICE_MISMATCH — speaking, but voice doesn't match enrollment
        if (inp.vad_active
                and score_available
                and inp.verify_score < THRESH_UNCERTAIN):
            return CheatType.VOICE_MISMATCH

        return None

    # ── Confidence scoring ────────────────────────────────────────────────────

    @staticmethod
    def _confidence(event: CheatType, inp: FusionInputs) -> float:
        if event == CheatType.IMPERSONATION:
            # Stronger signal when verify_score is very low and lips clearly still
            return min(1.0, (THRESH_UNCERTAIN - inp.verify_score) / THRESH_UNCERTAIN
                       + (0.3 if not inp.lip_active else 0.0))

        if event == CheatType.GHOST_VOICE:
            # High confidence when both voice matches AND lips are still
            return min(1.0, inp.verify_score)

        if event == CheatType.EXTRA_SPEAKER:
            return min(1.0, 0.5 + 0.2 * (inp.n_speakers - 1))

        if event == CheatType.VOICE_MISMATCH:
            return min(1.0, (THRESH_UNCERTAIN - inp.verify_score) / THRESH_UNCERTAIN)

        return 0.5
