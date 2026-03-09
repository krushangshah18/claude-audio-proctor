"""
ml_proctoring.audio_utils
=========================
Stateless audio helpers. No disk I/O — everything through io.BytesIO.
No dependency on core/ or any project module.
"""
from __future__ import annotations

import io
import struct
import wave

import numpy as np


def ensure_float32_mono(audio: np.ndarray, squeeze: bool = True) -> np.ndarray:
    """
    Normalise any numpy audio array to float32 mono in [-1, 1].
    Accepts int16, int32, float32, float64; handles stereo by averaging channels.
    """
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[1] <= audio.shape[0] else audio.mean(axis=0)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    return audio


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Serialise a float32 mono numpy array to WAV bytes (PCM-16).
    Returns raw bytes ready for HTTP upload — no disk I/O.
    """
    audio = ensure_float32_mono(audio)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio via scipy if src_sr ≠ dst_sr, else return as-is."""
    if src_sr == dst_sr:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
