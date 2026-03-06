"""
Speaker Verifier
=================
Compares test speaker embeddings against an enrolled voiceprint.

Enrollment:
  Call enroll(embedding) once per enrollment segment.
  Internally averages all enrollment embeddings → one centroid.
  The more enrollment audio, the more robust the centroid.

Thresholds (empirically tuned for the F0+mel extractor):
  sim ≥ 0.82  →  MATCH      (same person, high confidence)
  sim ≥ 0.68  →  LIKELY     (probably same person)
  sim ≥ 0.50  →  UNCERTAIN  (borderline — flag for review)
  sim <  0.50 →  MISMATCH   (different person)

Note: a trained ECAPA-TDNN model would use tighter thresholds
(~0.90 / 0.78 / 0.60) because its embeddings are more discriminative.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class VerifyResult(str, Enum):
    MATCH     = "MATCH"
    LIKELY    = "LIKELY"
    UNCERTAIN = "UNCERTAIN"
    MISMATCH  = "MISMATCH"


THRESH_MATCH     = 0.85
THRESH_LIKELY    = 0.72
THRESH_UNCERTAIN = 0.50


@dataclass
class VerificationResult:
    filename:   str
    similarity: float
    result:     VerifyResult
    confidence: float
    note:       str = ""

    @property
    def is_enrolled_student(self) -> bool:
        return self.result in (VerifyResult.MATCH, VerifyResult.LIKELY)

    def __str__(self) -> str:
        icon = {"MATCH": "✓", "LIKELY": "~", "UNCERTAIN": "?", "MISMATCH": "✗"}
        s = (f"  {icon[self.result.value]} {self.filename:<45} "
             f"sim={self.similarity:+.4f}  [{self.result.value:<9}]  conf={self.confidence:.2f}")
        if self.note:
            s += f"  ← {self.note}"
        return s


class SpeakerVerifier:
    """
    Holds one enrolled voiceprint (average of all enrollment embeddings)
    and verifies test embeddings against it.
    """

    def __init__(self):
        self._enrollment_embeds: list[np.ndarray] = []
        self._centroid: np.ndarray | None = None

    # ── Enrollment ──────────────────────────────────────────────────────────

    def add_enrollment(self, embedding: np.ndarray) -> None:
        """Add one enrollment embedding. Call multiple times for more coverage."""
        n = np.linalg.norm(embedding)
        if n > 1e-8:
            self._enrollment_embeds.append(embedding / n)
        self._centroid = None   # invalidate cache

    def finalize_enrollment(self) -> None:
        """Compute and cache the centroid from all enrollment embeddings."""
        if not self._enrollment_embeds:
            raise RuntimeError("No enrollment embeddings added")
        centroid = np.mean(self._enrollment_embeds, axis=0)
        n = np.linalg.norm(centroid)
        self._centroid = centroid / n if n > 1e-8 else centroid

    @property
    def is_enrolled(self) -> bool:
        return len(self._enrollment_embeds) > 0

    # ── Verification ────────────────────────────────────────────────────────

    def verify(self, embedding: np.ndarray, filename: str) -> VerificationResult:
        if not self.is_enrolled:
            raise RuntimeError("No enrollment — call add_enrollment() first")
        if self._centroid is None:
            self.finalize_enrollment()

        n = np.linalg.norm(embedding)
        test = embedding / n if n > 1e-8 else embedding
        similarity = float(np.dot(self._centroid, test))

        result, confidence, note = self._decide(similarity, filename)
        return VerificationResult(
            filename=filename,
            similarity=similarity,
            result=result,
            confidence=confidence,
            note=note,
        )

    @staticmethod
    def _decide(sim: float, filename: str) -> tuple[VerifyResult, float, str]:
        if sim >= THRESH_MATCH:
            conf   = min(1.0, (sim - THRESH_MATCH) / (1.0 - THRESH_MATCH) + 0.75)
            result = VerifyResult.MATCH
            note   = ""
        elif sim >= THRESH_LIKELY:
            conf   = 0.50 + 0.25 * (sim - THRESH_LIKELY) / (THRESH_MATCH - THRESH_LIKELY)
            result = VerifyResult.LIKELY
            note   = ""
        elif sim >= THRESH_UNCERTAIN:
            conf   = 0.25 * (sim - THRESH_UNCERTAIN) / (THRESH_LIKELY - THRESH_UNCERTAIN)
            result = VerifyResult.UNCERTAIN
            note   = "review recommended"
        else:
            conf   = min(1.0, (THRESH_UNCERTAIN - sim) / 0.30)
            result = VerifyResult.MISMATCH
            note   = "unknown speaker"

        # Annotate whisper-split segments
        if "_speaker_1" in filename and result in (VerifyResult.LIKELY, VerifyResult.UNCERTAIN):
            note = ("possible whisper of enrolled student — " + note).strip(" —")

        return result, round(float(conf), 3), note
