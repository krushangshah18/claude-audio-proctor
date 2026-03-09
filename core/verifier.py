"""
Speaker Verifier — v3
======================
Two-stream scoring: F0 histogram cosine + MFCC variance ratio.

Fused score = W_F0 * f0_cosine + W_VAR * var_ratio
  W_F0  = 0.65   W_VAR = 0.35

Variance ratio: mean(min(v1,v2) / max(v1,v2)) per MFCC coefficient.
  Same person → ratios close to 1.0
  Different person → ratios spread 0.5-0.9

Whisper exception:
  Whispering shifts F0 +50–100Hz, dropping f0_cosine to ~0.2.
  If a segment is tagged _speaker_1 (Stage 2 whisper split) AND
  var_ratio > 0.75 (vocal tract dynamics still match), 
  add a 0.18 bonus to the fused score before thresholding.
  This rescues legitimate whisper-of-enrolled-student from MISMATCH.

Thresholds on fused score:
  >= 0.72  MATCH      — enrolled student, confident
  >= 0.55  LIKELY     — enrolled student, probable
  >= 0.35  UNCERTAIN  — borderline, flag for review
  <  0.35  MISMATCH   — different person
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum
from core.embedding_extractor import W_F0, W_VAR

THRESH_MATCH     = 0.72
THRESH_LIKELY    = 0.55
THRESH_UNCERTAIN = 0.35


class VerifyResult(str, Enum):
    MATCH     = "MATCH"
    LIKELY    = "LIKELY"
    UNCERTAIN = "UNCERTAIN"
    MISMATCH  = "MISMATCH"


@dataclass
class VerificationResult:
    filename:    str
    similarity:  float    # final fused score (after any whisper bonus)
    sim_f0:      float    # F0 cosine similarity
    sim_var:     float    # MFCC variance ratio
    result:      VerifyResult
    confidence:  float
    note:        str = ""

    @property
    def is_enrolled_student(self) -> bool:
        return self.result in (VerifyResult.MATCH, VerifyResult.LIKELY)


class SpeakerVerifier:
    def __init__(self):
        self._f0_embeds:  list[np.ndarray] = []
        self._var_embeds: list[np.ndarray] = []
        self._centroid_f0:  np.ndarray | None = None
        self._centroid_var: np.ndarray | None = None

    def add_enrollment(self, streams: tuple[np.ndarray, np.ndarray]) -> None:
        f0_vec, var_vec = streams
        n = np.linalg.norm(f0_vec)
        if n > 1e-8:
            self._f0_embeds.append(f0_vec / n)
        # Reject silent/empty chunks (var_vec is all ~1e-8 sentinel)
        if np.any(var_vec > 1.0):   # real variance values are >> 1.0
            self._var_embeds.append(var_vec)
        self._centroid_f0 = self._centroid_var = None

    def finalize_enrollment(self) -> None:
        if not self._f0_embeds:
            raise RuntimeError("No enrollment F0 embeddings")
        c = np.mean(self._f0_embeds, axis=0)
        n = np.linalg.norm(c)
        self._centroid_f0  = c / n if n > 1e-8 else c
        self._centroid_var = np.mean(self._var_embeds, axis=0) if self._var_embeds \
                             else np.ones(10, np.float32)

    @property
    def is_enrolled(self) -> bool:
        return len(self._f0_embeds) > 0

    def verify(self, streams: tuple[np.ndarray, np.ndarray],
               filename: str) -> VerificationResult:
        if not self.is_enrolled:
            raise RuntimeError("Not enrolled")
        if self._centroid_f0 is None:
            self.finalize_enrollment()

        f0_test, var_test = streams

        # F0 cosine similarity
        n = np.linalg.norm(f0_test)
        f0_n = f0_test / n if n > 1e-8 else f0_test
        s_f0 = float(np.dot(self._centroid_f0, f0_n))

        # MFCC variance ratio (element-wise min/max mean)
        s_var = float(np.mean(
            np.minimum(self._centroid_var, var_test) /
            (np.maximum(self._centroid_var, var_test) + 1e-8)
        ))

        fused = W_F0 * max(0.0, s_f0) + W_VAR * s_var

        result, confidence, note = self._decide(fused, s_f0, s_var, filename)
        return VerificationResult(
            filename=filename,
            similarity=round(fused, 4),
            sim_f0=round(s_f0, 4),
            sim_var=round(s_var, 4),
            result=result,
            confidence=confidence,
            note=note,
        )

    @staticmethod
    def _decide(fused, s_f0, s_var, filename) -> tuple:
        note = ""
        if fused >= THRESH_MATCH:
            conf   = min(1.0, 0.75 + (fused - THRESH_MATCH)/(1.0 - THRESH_MATCH))
            result = VerifyResult.MATCH
        elif fused >= THRESH_LIKELY:
            conf   = 0.50 + 0.25*(fused - THRESH_LIKELY)/(THRESH_MATCH - THRESH_LIKELY)
            result = VerifyResult.LIKELY
        elif fused >= THRESH_UNCERTAIN:
            conf   = 0.25*(fused - THRESH_UNCERTAIN)/(THRESH_LIKELY - THRESH_UNCERTAIN)
            result = VerifyResult.UNCERTAIN
            note   = "ambiguous — close F0, review recommended"
        else:
            conf   = min(1.0, (THRESH_UNCERTAIN - fused)/0.20)
            result = VerifyResult.MISMATCH
            if s_f0 < 0.30 and s_var > 0.70:
                note = "shifted F0 — possible different speaker or whispering"
            else:
                note = "unknown speaker"

        return result, round(float(conf), 3), note