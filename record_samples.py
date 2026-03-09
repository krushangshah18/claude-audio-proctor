"""
record_samples.py — Interactive audio sample recorder
Records multiple WAV samples to inputStage1/ (or a custom directory).

Usage:
    uv run record_samples.py
    uv run record_samples.py --output-dir my_samples --duration 5 --count 3
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHANNELS = 1


def list_input_devices():
    devices = sd.query_devices()
    print("\nAvailable input devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}")
    print()


def record_clip(duration: float, device=None) -> np.ndarray:
    """Record audio and return as float32 numpy array."""
    frames = int(SAMPLE_RATE * duration)
    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )
    # Live countdown
    for remaining in range(int(duration), 0, -1):
        print(f"  Recording... {remaining}s remaining", end="\r")
        time.sleep(1)
    sd.wait()
    print(" " * 40, end="\r")  # clear line
    return audio.squeeze()


def save_wav(audio: np.ndarray, path: Path):
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Record audio samples")
    parser.add_argument(
        "--output-dir",
        default="inputStage1",
        help="Directory to save WAV files (default: inputStage1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of samples to record (default: prompt interactively)",
    )
    parser.add_argument(
        "--prefix",
        default="sample",
        help="Filename prefix, e.g. 'enrollment' → enrollment_1.wav (default: sample)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (omit to use system default)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        sys.exit(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nAudio recorder — {SAMPLE_RATE} Hz, mono")
    print(f"Output dir : {out_dir.resolve()}")
    print(f"Duration   : {args.duration}s per clip")
    if args.device is not None:
        print(f"Device     : [{args.device}] {sd.query_devices(args.device)['name']}")
    else:
        default_dev = sd.query_devices(kind="input")
        print(f"Device     : {default_dev['name']} (system default)")

    sample_num = 1
    recorded = 0

    try:
        while True:
            if args.count is not None and recorded >= args.count:
                break

            # Find a free filename
            while True:
                out_path = out_dir / f"{args.prefix}_{sample_num}.wav"
                if not out_path.exists():
                    break
                sample_num += 1

            if args.count is None:
                answer = input(
                    f"\nRecord '{out_path.name}'? [Enter to record / 'q' to quit] "
                ).strip().lower()
                if answer == "q":
                    break
            else:
                print(f"\nRecording {recorded + 1}/{args.count}: {out_path.name}")
                input("  Press Enter to start...")

            print(f"  -> Recording {args.duration}s...")
            audio = record_clip(args.duration, device=args.device)

            peak = float(np.abs(audio).max())
            if peak < 0.01:
                print("  WARNING: Very low signal — check your microphone.")

            save_wav(audio, out_path)
            print(f"  Saved: {out_path}  (peak={peak:.3f})")

            recorded += 1
            sample_num += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nDone. {recorded} sample(s) saved to '{out_dir}/'.")


if __name__ == "__main__":
    main()
