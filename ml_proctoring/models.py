"""
ml_proctoring.models
====================
Pure dataclasses and enums. No logic, no imports from core/.
This is the shared contract between all subsystems and the video pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np


# ── Cheat event taxonomy ──────────────────────────────────────────────────────

class CheatType(str, Enum):
    IMPERSONATION  = "IMPERSONATION"   # speech detected, lips still, voice ≠ enrolled
    GHOST_VOICE    = "GHOST_VOICE"     # enrolled voice heard, lips NOT moving (playback)
    EXTRA_SPEAKER  = "EXTRA_SPEAKER"   # 2+ distinct speaker clusters detected
    VOICE_MISMATCH = "VOICE_MISMATCH"  # speech detected, voice does not match enrollment


@dataclass
class CheatEvent:
    """
    One flagged anomaly.

    audio_proof — WAV bytes covering the last buffer_seconds of audio at the
                  moment of the flag. Ready for POST to backend as multipart/form-data.
    """
    event_type   : CheatType
    confidence   : float           # 0.0 – 1.0
    timestamp_s  : float           # seconds since session start
    verify_score : float           # cosine similarity to enrolled speaker
    n_speakers   : int             # distinct speaker clusters at flag time
    lip_active   : bool            # lip_activity value at flag time
    audio_proof  : bytes = field(repr=False)  # WAV bytes, potentially large


# ── Internal signal bundle passed from session → fusion ──────────────────────

class FusionInputs(NamedTuple):
    vad_active   : bool
    lip_active   : bool
    verify_score : float   # 0.0 if no embedding computed yet
    n_speakers   : int
    timestamp_s  : float


# ── Enrollment state ──────────────────────────────────────────────────────────

@dataclass
class SpeakerProfile:
    centroid    : np.ndarray        # 256-dim L2-normalised d-vector
    n_segments  : int               # number of segments used to build centroid
