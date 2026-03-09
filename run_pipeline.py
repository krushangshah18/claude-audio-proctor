"""
Full Pipeline — Stage 1 -> 2 -> 3

Chains all three stages end-to-end on a recording:
  Stage 1: VAD          -> speech_only.wav
  Stage 2: Diarization  -> speaker_N.wav files
  Stage 3: Verification -> MATCH / LIKELY / UNCERTAIN / MISMATCH per speaker

Usage:
  python run_pipeline.py --file recording.wav --enroll enrollment.wav
  python run_pipeline.py --dir  ./recordings  --enroll enrollment.wav
  python run_pipeline.py --file recording.wav   # skip Stage 3 (no enrollment)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_stage1 import run_stage1
from run_stage2 import run_stage2
from run_stage3 import run_stage3

OUTPUT_DIR = "output_pipeline"


def run_pipeline(wav_path: str, enroll_path: str | None, out_dir: str) -> dict:
    """
    Run all 3 stages on a single recording.
    Returns {"stage1": ..., "stage2": ..., "stage3": ...}
    """
    s1_dir = os.path.join(out_dir, "stage1")
    s2_dir = os.path.join(out_dir, "stage2")
    s3_dir = os.path.join(out_dir, "stage3")

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE — {os.path.basename(wav_path)}")
    print(f"{'=' * 60}")

    # Stage 1: VAD
    r1 = run_stage1(wav_path, s1_dir)

    # Stage 2: Diarization — feed the speech-only WAV from Stage 1
    r2 = run_stage2(r1["speech_wav"], s2_dir)

    # Stage 3: Verification — compare each speaker track against enrollment
    r3 = None
    if enroll_path:
        if not os.path.exists(enroll_path):
            print(f"\n  [WARN] Enrollment file not found: {enroll_path} — skipping Stage 3")
        elif not r2["speaker_wavs"]:
            print("\n  [WARN] No speaker WAVs from Stage 2 — skipping Stage 3")
        else:
            r3 = run_stage3(r2["speaker_wavs"], enroll_path, s3_dir)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {os.path.basename(wav_path)}")
    print(f"{'=' * 60}")
    print(f"  Stage 1  : {len(r1['segments'])} segments  "
          f"{r1['total_speech_s']:.1f}s speech  RTF={r1['rtf']:.4f}")
    print(f"  Stage 2  : {r2['n_speakers']} speakers  "
          f"{r2['summary']['total_flags']} flags")
    if r3:
        n_enrolled = sum(1 for r in r3 if r.result.value in ("MATCH", "LIKELY"))
        n_unknown  = sum(1 for r in r3 if r.result.value == "MISMATCH")
        print(f"  Stage 3  : {n_enrolled} enrolled / {n_unknown} unknown "
              f"out of {len(r3)} speaker tracks")
    print(f"  Output   : ./{out_dir}/")
    print(f"{'=' * 60}\n")

    return {"stage1": r1, "stage2": r2, "stage3": r3}


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline — Stages 1 -> 2 -> 3")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Single WAV recording")
    group.add_argument("--dir",  type=str, help="Folder of WAV recordings")
    parser.add_argument("--enroll", type=str, default=None,
                        help="Enrollment WAV for Stage 3 speaker verification (optional)")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
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

    for f in files:
        # When processing multiple files, give each its own subdirectory
        out = os.path.join(args.out, os.path.splitext(os.path.basename(f))[0]) \
              if len(files) > 1 else args.out
        run_pipeline(f, args.enroll, out)


if __name__ == "__main__":
    main()
