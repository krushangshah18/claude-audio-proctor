"""
ml_proctoring
=============
Isolated AI/ML audio proctoring module.

Public surface — only these names are stable API:

    ProctorSession  — orchestrates all subsystems
    CheatEvent      — one flagged anomaly (carries .audio_proof WAV bytes)
    CheatType       — IMPERSONATION | GHOST_VOICE | EXTRA_SPEAKER | VOICE_MISMATCH

Usage (video pipeline integration):
    from ml_proctoring import ProctorSession, CheatEvent, CheatType

    session = ProctorSession()
    session.enroll(enrollment_audio_np)

    # audio thread
    events = session.push(audio_chunk, lip_activity=lips_moving_bool)
    for ev in events:
        print(ev.event_type, ev.confidence, ev.timestamp_s)
        backend.upload_proof(ev.audio_proof)   # WAV bytes

    # video thread (alternative lip update path)
    session.update_lip_activity(lips_moving=True)
"""
from .session import ProctorSession
from .models  import CheatEvent, CheatType

__all__ = ["ProctorSession", "CheatEvent", "CheatType"]
