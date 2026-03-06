"""
Test Audio Generator
====================
Creates 6 WAV files covering key VAD test scenarios.
All at 16kHz mono (Silero's native format).
"""

from __future__ import annotations
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, sosfilt
from scipy.ndimage import uniform_filter1d

SR = 16000
OUT = "audio_samples"
os.makedirs(OUT, exist_ok=True)


def silence(dur: float) -> np.ndarray:
    return np.zeros(int(dur * SR), dtype=np.float32)


def white_noise(dur: float, amp: float = 0.025) -> np.ndarray:
    return (np.random.randn(int(dur * SR)) * amp).astype(np.float32)


def road_rumble(dur: float, amp: float = 0.05) -> np.ndarray:
    n = int(dur * SR)
    noise = np.random.randn(n) * amp
    sos = butter(4, 150 / (SR / 2), btype="low", output="sos")
    return sosfilt(sos, noise).astype(np.float32)


def car_horn(dur: float = 0.55, freq: float = 440.0, amp: float = 0.35) -> np.ndarray:
    t = np.linspace(0, dur, int(dur * SR))
    sig = amp * (np.sin(2 * np.pi * freq * t) + 0.12 * np.sin(4 * np.pi * freq * t))
    env = np.ones_like(t)
    ramp = int(0.04 * SR)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return (sig * env).astype(np.float32)


def voice(
    dur: float,
    f0: float = 130.0,
    amp: float = 0.38,
    n_harmonics: int = 8,
    jitter: float = 0.012,
    breathiness: float = 0.04,
    syllable_rate: float = 4.5,
    voiced_ratio: float = 0.72,
) -> np.ndarray:
    n = int(dur * SR)
    src = np.zeros(n, dtype=np.float32)
    phase = 0.0
    for i in range(n):
        f0_i = f0 * (1 + jitter * (np.random.rand() - 0.5))
        phase += 2 * np.pi * f0_i / SR
        src[i] = sum((1.0 / h) * np.sin(phase * h) for h in range(1, n_harmonics + 1))
    src += breathiness * np.random.randn(n).astype(np.float32)

    # Formant shaping
    from scipy.signal import iirpeak
    shaped = src.copy()
    for fc, bw in [(700, 130), (1220, 90), (2600, 160)]:
        if fc < SR / 2:
            b, a = iirpeak(fc / (SR / 2), fc / bw)
            shaped = np.convolve(shaped, b, mode="same").astype(np.float32)
    shaped /= np.abs(shaped).max() + 1e-9

    # Syllabic amplitude envelope
    sps = int(SR / syllable_rate)
    env = np.zeros(n, dtype=np.float32)
    for s in range(0, n, sps):
        e = min(s + sps, n)
        seg_len = e - s
        if np.random.rand() < voiced_ratio:
            t_s = np.linspace(0, np.pi, seg_len)
            env[s:e] = np.sin(t_s) ** 0.5
    env = uniform_filter1d(env, size=int(0.018 * SR)).astype(np.float32)

    return (shaped * env * amp).astype(np.float32)


def whisper(dur: float, amp: float = 0.07) -> np.ndarray:
    """Whisper = mostly noise with weak spectral envelope, no clear F0."""
    n = int(dur * SR)
    src = np.random.randn(n) * amp
    from scipy.signal import iirpeak
    shaped = src.copy()
    for fc, bw in [(1000, 400), (2200, 500)]:
        if fc < SR / 2:
            b, a = iirpeak(fc / (SR / 2), fc / bw)
            shaped = np.convolve(shaped, b, mode="same").astype(np.float32)
    # Syllabic envelope so it has rhythm (unlike flat white noise)
    sps = int(SR / 4.0)
    env = np.zeros(n, dtype=np.float32)
    for s in range(0, n, sps):
        e = min(s + sps, n)
        if np.random.rand() < 0.65:
            t_s = np.linspace(0, np.pi, e - s)
            env[s:e] = np.sin(t_s) ** 0.5
    env = uniform_filter1d(env, size=int(0.015 * SR)).astype(np.float32)
    return (shaped * env * 1.5).astype(np.float32)


def mix(*parts: np.ndarray) -> np.ndarray:
    maxlen = max(len(p) for p in parts)
    out = np.zeros(maxlen, dtype=np.float32)
    for p in parts:
        out[:len(p)] += p
    return np.clip(out, -0.95, 0.95)


def cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def save(name: str, audio: np.ndarray):
    path = os.path.join(OUT, name)
    a16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    wav.write(path, SR, a16)
    dur = len(audio) / SR
    print(f"  {name:<42} ({dur:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────

def gen_all():
    print("=" * 60)
    print("Generating VAD Test Audio Samples")
    print("=" * 60)
    np.random.seed(42)

    # 1. Clean speech — baseline test
    print("\n[1] clean_speech.wav")
    print("    Expected speech: 1-4s, 4.9-7.4s, 7.9-11.9s, 13.3-15.3s")
    save("clean_speech.wav", cat(
        silence(1.0),
        voice(3.0, f0=130),
        silence(0.9),
        voice(2.5, f0=134),
        silence(0.5),
        voice(4.0, f0=128),
        silence(1.4),
        voice(2.0, f0=132),
        silence(1.0),
    ))

    # 2. Speech + road noise + car honks
    print("\n[2] speech_with_road_noise.wav")
    print("    Honks at 0.3, 0.8, 8.5, 9.0, 15.5, 17.0s — should NOT be flagged")
    total = 18.0
    bg = road_rumble(total, amp=0.055)
    sp = cat(
        silence(1.5), voice(3.0, f0=125), silence(1.0),
        voice(2.5, f0=129), silence(0.8),
        voice(3.5, f0=127), silence(1.2),
        voice(2.0, f0=128), silence(2.5),
    )
    horns = np.zeros(int(total * SR), dtype=np.float32)
    for t_h in [0.3, 0.8, 8.5, 9.0, 15.5, 17.0]:
        s = int(t_h * SR)
        h = car_horn(dur=0.5, freq=420 + np.random.randint(-20, 20))
        horns[s: s + len(h)] += h[: len(horns) - s]
    save("speech_with_road_noise.wav", mix(sp, bg, horns))

    # 3. Whisper only — sensitivity test
    print("\n[3] whisper_speech.wav")
    print("    Expected: whisper detected at 0.5-3.5s, 4.0-6.5s  (low energy)")
    save("whisper_speech.wav", cat(
        silence(0.5),
        whisper(3.0, amp=0.065),
        silence(0.5),
        whisper(2.5, amp=0.055),
        silence(0.5),
    ))

    # 4. Normal → whisper → normal (SAME PERSON — must NOT be flagged as new speaker)
    print("\n[4] normal_then_whisper.wav")
    print("    Normal:0.5-3.5s | Whisper:3.8-6.3s | Normal:6.6-9.6s")
    print("    ALL segments = same person. Stage 2 must not flag this.")
    save("normal_then_whisper.wav", cat(
        silence(0.5),
        voice(3.0, f0=130, amp=0.38),
        silence(0.3),
        whisper(2.5, amp=0.065),
        silence(0.3),
        voice(3.0, f0=129, amp=0.36),
        silence(0.5),
    ))

    # 5. Silence + noise only — should produce ZERO speech segments
    print("\n[5] silence_and_noise.wav")
    print("    Expected: 0 speech segments")
    horn_burst = np.zeros(int(13.5 * SR), dtype=np.float32)
    for t_h in [11.2]:
        s = int(t_h * SR)
        h = car_horn(dur=0.55, freq=430)
        horn_burst[s: s + len(h)] += h[: len(horn_burst) - s]
    save("silence_and_noise.wav", mix(
        cat(silence(3.0), white_noise(2.0, amp=0.03),
            silence(2.0), road_rumble(4.0, amp=0.07), silence(2.5)),
        horn_burst,
    ))

    # 6. Realistic exam scenario
    print("\n[6] mixed_realistic.wav")
    print("    Speech: 1-5s, 6.5-10s, 12-17s, 17.5-19s(whisper), 19.5-23.5s, 24.5-27s")
    print("    Honks: 0.2, 5.5, 11, 27.5, 28.5s  |  Continuous road bg")
    total2 = 30.0
    bg2 = road_rumble(total2, amp=0.038)
    sp2 = cat(
        silence(1.0),
        voice(4.0, f0=132, amp=0.34),
        silence(1.5),
        voice(3.5, f0=130, amp=0.37),
        silence(2.0),
        voice(5.0, f0=131, amp=0.35),
        silence(0.5),
        whisper(1.5, amp=0.062),
        silence(0.5),
        voice(4.0, f0=130, amp=0.36),
        silence(1.0),
        voice(2.5, f0=133, amp=0.33),
        silence(3.0),
    )
    horns2 = np.zeros(int(total2 * SR), dtype=np.float32)
    for t_h in [0.2, 5.5, 11.0, 27.5, 28.5]:
        s = int(t_h * SR)
        h = car_horn(dur=0.5, freq=400 + np.random.randint(-30, 50))
        horns2[s: s + len(h)] += h[: len(horns2) - s]
    save("mixed_realistic.wav", mix(sp2, bg2, horns2))

    print(f"\nAll audio saved to: ./{OUT}/")


if __name__ == "__main__":
    gen_all()
