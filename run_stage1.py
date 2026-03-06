"""
VAD Stage 1 — Runner
====================
Usage examples:

  # Run on built-in test audio (auto-generated)
  python run_vad.py

  # Your own recording
  python run_vad.py --file /path/to/recording.wav

  # A whole folder of recordings
  python run_stage1.py --dir ./inputStage1

  # Tune sensitivity (if missing quiet speech)
  python run_vad.py --file rec.wav --threshold 0.35

  # Tune for noisy environments (more false positives OK)
  python run_vad.py --file rec.wav --threshold 0.55 --onset 5

  # Convert non-WAV first:
  #   ffmpeg -i recording.m4a -ar 16000 -ac 1 recording.wav

Outputs (in ./output/):
  {name}_speech_only.wav   — original audio with silence zeroed out
  {name}_report.txt        — timestamp + per-segment breakdown
  {name}_analysis.png      — 4-panel waveform + probability plot
  SUMMARY.txt              — combined results across all files
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

AUDIO_DIR = "audio_samples"
OUTPUT_DIR = "output_stage1"


# ─────────────────────────────────────────────────────────────────────────────

def process_file(wav_path: str, config: VADConfig, out_dir: str) -> dict:
    filename = os.path.basename(wav_path)
    stem = os.path.splitext(filename)[0]

    print(f"\n{'─' * 62}")
    print(f"  Processing: {filename}")
    print(f"{'─' * 62}")

    # Load + normalise
    audio = load_audio(wav_path, target_sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration  : {duration:.2f}s  |  Samples: {len(audio)}")

    # Run VAD
    engine = create_vad_engine(config)
    engine_name = type(engine).__name__

    t0 = time.perf_counter()
    frame_results = engine.process_audio(audio)
    segments = engine.get_segments()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    rtf = elapsed_ms / (duration * 1000)

    print(f"  Engine    : {engine_name}")
    print(f"  VAD time  : {elapsed_ms:.1f}ms  (RTF {rtf:.4f}  — target <0.05)")
    print(f"  Segments  : {len(segments)}")

    for i, seg in enumerate(segments, 1):
        wt = " [WHISPER]" if seg.is_whisper else ""
        print(
            f"    [{i:>2}] {seg.start_s:.2f}s → {seg.end_s:.2f}s  "
            f"({seg.duration_s:.2f}s){wt}  "
            f"prob={seg.avg_prob:.3f}  rms={seg.avg_rms:.5f}"
        )

    os.makedirs(out_dir, exist_ok=True)

    # Speech-only audio
    speech_audio = build_speech_only_audio(audio, segments, SAMPLE_RATE)
    speech_path = os.path.join(out_dir, f"{stem}_speech_only.wav")
    save_audio(speech_path, speech_audio, SAMPLE_RATE)
    print(f"  → speech_only : {stem}_speech_only.wav")

    # Report
    report_path = os.path.join(out_dir, f"{stem}_report.txt")
    build_report(filename, segments, frame_results, duration, engine_name, report_path)
    print(f"  → report      : {stem}_report.txt")

    # Plot
    plot_path = os.path.join(out_dir, f"{stem}_analysis.png")
    plot_vad_analysis(audio, SAMPLE_RATE, frame_results, segments, stem, plot_path, engine_name)

    total_speech = sum(s.duration_s for s in segments)
    return {
        "file": filename,
        "duration_s": duration,
        "segments": len(segments),
        "total_speech_s": total_speech,
        "speech_pct": 100 * total_speech / max(duration, 0.001),
        "rtf": rtf,
        "elapsed_ms": elapsed_ms,
        "whisper_count": sum(1 for s in segments if s.is_whisper),
        "engine": engine_name,
    }


def write_summary(results: list[dict], out_dir: str):
    lines = [
        "=" * 72,
        "VAD STAGE 1 — SUMMARY",
        "=" * 72,
        "",
        f"  {'File':<38} {'Segs':>5} {'Speech%':>8} {'Whispers':>9} {'RTF':>8}",
        "  " + "─" * 68,
    ]
    for r in results:
        lines.append(
            f"  {r['file']:<38} {r['segments']:>5} "
            f"{r['speech_pct']:>7.1f}%  {r['whisper_count']:>8}  {r['rtf']:>7.4f}"
        )

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    status = "✓ EXCELLENT" if avg_rtf < 0.05 else ("~ ACCEPTABLE" if avg_rtf < 0.20 else "✗ SLOW")
    lines += [
        "",
        f"  Average RTF : {avg_rtf:.4f}  ({status})",
        "  (RTF < 0.05 = running 20× real-time — leaves full CPU budget for YOLO)",
        "",
        "=" * 72,
    ]
    text = "\n".join(lines)
    path = os.path.join(out_dir, "SUMMARY.txt")
    with open(path, "w") as f:
        f.write(text)
    print("\n" + text)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VAD Stage 1 — Silero-powered speech detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file",      type=str, help="Single WAV file to process")
    parser.add_argument("--dir",       type=str, help="Directory of WAV files to process")
    parser.add_argument("--generate",  action="store_true", help="Regenerate test audio first")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Silero speech probability threshold (default: 0.45)\n"
                             "Lower = more sensitive  |  Higher = stricter")
    parser.add_argument("--onset",     type=int, default=3,
                        help="Consecutive speech chunks before starting segment (default: 3 ≈ 96ms)")
    parser.add_argument("--offset",    type=int, default=10,
                        help="Consecutive silence chunks before ending segment (default: 10 ≈ 320ms)")
    parser.add_argument("--min-speech-ms", type=int, default=250,
                        help="Minimum segment duration in ms (default: 250)")
    parser.add_argument("--out",       type=str, default=OUTPUT_DIR,
                        help="Output directory (default: ./output/)")
    args = parser.parse_args()

    config = VADConfig(
        silero_threshold=args.threshold,
        onset_chunks=args.onset,
        offset_chunks=args.offset,
        min_speech_ms=args.min_speech_ms,
    )

    print("=" * 62)
    print("  VAD STAGE 1")
    print(f"  threshold={config.silero_threshold}  "
          f"onset={config.onset_chunks}  offset={config.offset_chunks}  "
          f"min_speech={config.min_speech_ms}ms")
    print("=" * 62)

    if args.generate:
        print("\nGenerating test audio...")
        import subprocess
        subprocess.run([sys.executable, "generate_test_audio.py"], check=True)

    # Collect files
    if args.file:
        if not os.path.exists(args.file):
            print(f"✗ File not found: {args.file}")
            sys.exit(1)
        files = [args.file]

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"✗ Directory not found: {args.dir}")
            sys.exit(1)
        files = sorted(
            os.path.join(args.dir, f)
            for f in os.listdir(args.dir)
            if f.lower().endswith(".wav")
        )
        if not files:
            print(f"✗ No WAV files in {args.dir}")
            print("  Convert with: ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav")
            sys.exit(1)

    else:
        if not os.path.exists(AUDIO_DIR) or not os.listdir(AUDIO_DIR):
            print("No audio_samples/ found — generating test audio...")
            import subprocess
            subprocess.run([sys.executable, "generate_test_audio.py"], check=True)
        files = sorted(
            os.path.join(AUDIO_DIR, f)
            for f in os.listdir(AUDIO_DIR)
            if f.lower().endswith(".wav")
        )

    # Process
    results = [process_file(f, config, args.out) for f in files]
    if len(results) > 1:
        write_summary(results, args.out)

    print(f"\n{'═' * 62}")
    print(f"  Outputs in: ./{args.out}/")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()
