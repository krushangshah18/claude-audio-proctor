"""
Live Audio Pipeline — record from microphone and run all 3 stages

Records audio from the system microphone, saves it, then runs the full
Stage 1 -> 2 -> 3 pipeline.

Usage:
  python run_live.py                                     # 30s recording, no Stage 3
  python run_live.py --enroll enrollment.wav             # 30s + speaker verification
  python run_live.py --enroll enrollment.wav --duration 60
  python run_live.py --device 2 --enroll enrollment.wav  # specific mic
  python run_live.py --list-devices                      # show available mics

Requirements:
  uv add sounddevice    # or: pip install sounddevice
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import sounddevice as sd
except ImportError:
    sys.exit(
        "sounddevice not installed.\n"
        "Run:  uv add sounddevice\n"
        "  or: pip install sounddevice"
    )

from run_pipeline import run_pipeline

SAMPLE_RATE = 16000
OUTPUT_DIR  = "output_live"


def list_devices() -> None:
    print("\nAvailable audio input devices:")
    print("─" * 50)
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " <- default" if i == sd.default.device[0] else ""
            print(f"  [{i:>2}] {dev['name']}{marker}")
    print()


def record(duration: float, device: int | None = None) -> np.ndarray:
    n_samples = int(duration * SAMPLE_RATE)
    dev_label = f"device {device}" if device is not None else "default mic"
    print(f"\n  Recording {duration:.0f}s from {dev_label}...")
    print("  Press Ctrl+C to stop early.\n")

    audio = sd.rec(n_samples, samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
    start = time.perf_counter()
    try:
        for _ in range(int(duration)):
            time.sleep(1)
            elapsed = int(time.perf_counter() - start)
            filled  = min(elapsed, int(duration))
            bar     = "█" * filled + "░" * (int(duration) - filled)
            print(f"\r  [{bar}] {elapsed}s / {int(duration)}s", end="", flush=True)
        sd.wait()
        print()
    except KeyboardInterrupt:
        sd.stop()
        elapsed_s = time.perf_counter() - start
        audio     = audio[:int(elapsed_s * SAMPLE_RATE)]
        print(f"\n  Recording stopped at {elapsed_s:.1f}s")

    return audio.flatten()


def save_wav(audio: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Live Audio Pipeline")
    parser.add_argument("--enroll",      type=str,   default=None,
                        help="Enrollment WAV for speaker verification (optional)")
    parser.add_argument("--duration",    type=float, default=30.0,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--device",      type=int,   default=None,
                        help="Input device index (default: system default)")
    parser.add_argument("--out",         type=str,   default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available microphones and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.enroll and not os.path.exists(args.enroll):
        sys.exit(f"Enrollment file not found: {args.enroll}")

    print("=" * 60)
    print("  LIVE AUDIO PIPELINE")
    print("=" * 60)

    audio    = record(args.duration, args.device)
    wav_path = os.path.join(args.out, "live_recording.wav")
    save_wav(audio, wav_path)
    print(f"  Saved recording: {wav_path}")

    run_pipeline(wav_path, args.enroll, args.out)


if __name__ == "__main__":
    main()
