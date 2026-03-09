"""
Stage 1 — Voice Activity Detection

Usage:
  python run_stage1.py --file recording.wav
  python run_stage1.py --dir ./inputStage1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.vad_engine import VADConfig, create_vad_engine, load_audio, save_audio, SAMPLE_RATE
from core.output_builder import build_speech_only_audio, build_report
from core.visualizer import plot_vad_analysis

OUTPUT_DIR = "output_stage1"


def run_stage1(wav_path: str, out_dir: str, config: VADConfig | None = None) -> dict:
    """
    Run Stage 1 VAD on a single WAV file.
    Returns a result dict including audio arrays, detected segments, and output paths.
    """
    if config is None:
        config = VADConfig()

    filename = os.path.basename(wav_path)
    stem     = os.path.splitext(filename)[0]

    print(f"\n{'─' * 60}")
    print(f"  Stage 1 — {filename}")
    print(f"{'─' * 60}")

    audio    = load_audio(wav_path, target_sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration  : {duration:.2f}s")

    engine       = create_vad_engine(config)
    t0           = time.perf_counter()
    frame_results = engine.process_audio(audio)
    segments     = engine.get_segments()
    elapsed_ms   = (time.perf_counter() - t0) * 1000
    rtf          = elapsed_ms / (duration * 1000)

    print(f"  Engine    : {type(engine).__name__}")
    print(f"  Segments  : {len(segments)}  |  RTF: {rtf:.4f}")
    for i, seg in enumerate(segments, 1):
        tag = " [WHISPER]" if seg.is_whisper else ""
        print(f"    [{i:>2}] {seg.start_s:.2f}s → {seg.end_s:.2f}s  ({seg.duration_s:.2f}s){tag}")

    os.makedirs(out_dir, exist_ok=True)

    speech_audio = build_speech_only_audio(audio, segments, SAMPLE_RATE)
    speech_path  = os.path.join(out_dir, f"{stem}_speech_only.wav")
    save_audio(speech_path, speech_audio, SAMPLE_RATE)

    build_report(filename, segments, frame_results, duration,
                 type(engine).__name__, os.path.join(out_dir, f"{stem}_report.txt"))

    plot_vad_analysis(audio, SAMPLE_RATE, frame_results, segments, stem,
                      os.path.join(out_dir, f"{stem}_analysis.png"), type(engine).__name__)

    total_speech = sum(s.duration_s for s in segments)
    whispers     = sum(1 for s in segments if s.is_whisper)
    print(f"  Speech    : {total_speech:.2f}s ({100 * total_speech / max(duration, 0.001):.1f}%)  |  Whispers: {whispers}")

    return {
        "file":          filename,
        "stem":          stem,
        "audio":         audio,
        "segments":      segments,
        "speech_audio":  speech_audio,
        "speech_wav":    speech_path,
        "duration_s":    duration,
        "total_speech_s": total_speech,
        "rtf":           rtf,
        "whisper_count": whispers,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1 — Voice Activity Detection")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="WAV file to process")
    group.add_argument("--dir",  type=str, help="Folder of WAV files")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            sys.exit(f"File not found: {args.file}")
        files = [args.file]
    else:
        if not os.path.isdir(args.dir):
            sys.exit(f"Directory not found: {args.dir}")
        files = sorted(
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith(".wav")
        )
        if not files:
            sys.exit(f"No WAV files in {args.dir}")

    config = VADConfig()
    for f in files:
        run_stage1(f, args.out, config)

    print(f"\n  Outputs in: ./{args.out}/\n")


if __name__ == "__main__":
    main()
