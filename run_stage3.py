"""
Stage 3 — Speaker Verification

Usage:
  python run_stage3.py --input inputStage3

  inputStage3/ must contain:
    enrollment.wav      ← reference recording (>= 3s)
    speaker_0.wav       ← files to verify (e.g. Stage 2 outputs)
    speaker_1.wav
    ...

  Or specify enrollment explicitly:
    python run_stage3.py --input inputStage3 --enroll /path/to/enrollment.wav

Similarity thresholds:
  >= 0.85  MATCH      — enrolled student (confident)
  >= 0.72  LIKELY     — enrolled student (probable)
  >= 0.50  UNCERTAIN  — borderline, flag for review
  <  0.50  MISMATCH   — different person
"""

from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.embedding_extractor import EmbeddingExtractor, EMBED_DIM
from core.verifier import (SpeakerVerifier, VerifyResult,
                            THRESH_MATCH, THRESH_LIKELY, THRESH_UNCERTAIN)

OUTPUT_DIR      = "output_stage3"
ENROLLMENT_NAME = "enrollment.wav"
SAMPLE_RATE     = 16000


def _load_wav(path: str) -> np.ndarray:
    with wave.open(path) as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        sw  = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = np.int16 if sw == 2 else np.int8
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= 32768.0 if sw == 2 else 128.0
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * SAMPLE_RATE / sr)),
            np.arange(len(audio)), audio,
        ).astype(np.float32)
    return audio


def run_stage3(test_wavs: list[str], enroll_path: str, out_dir: str) -> list:
    """
    Verify a list of WAV files against an enrollment recording.
    Returns list of VerificationResult objects.
    """
    print(f"\n{'─' * 60}")
    print(f"  Stage 3 — Speaker Verification")
    print(f"  Thresholds: MATCH>={THRESH_MATCH}  LIKELY>={THRESH_LIKELY}  UNCERTAIN>={THRESH_UNCERTAIN}")
    print(f"{'─' * 60}")

    extractor = EmbeddingExtractor(sample_rate=SAMPLE_RATE)
    verifier  = SpeakerVerifier()

    # Enrollment — split into 5s chunks for a robust centroid
    enroll_audio = _load_wav(enroll_path)
    enroll_dur   = len(enroll_audio) / SAMPLE_RATE
    CHUNK        = SAMPLE_RATE * 5
    chunks       = [enroll_audio[i:i+CHUNK] for i in range(0, len(enroll_audio), CHUNK)
                    if len(enroll_audio[i:i+CHUNK]) >= SAMPLE_RATE * 2]
    if not chunks:
        chunks = [enroll_audio]

    for chunk in chunks:
        verifier.add_enrollment(extractor.extract(chunk))
    verifier.finalize_enrollment()
    print(f"  Enrollment: {os.path.basename(enroll_path)}  ({enroll_dur:.1f}s  {len(chunks)} chunks  {EMBED_DIM}-dim embedding)")

    # Verification
    print(f"\n  {'File':<45} {'Sim':>7}  Result")
    print(f"  {'─' * 60}")
    results = []
    for fpath in test_wavs:
        audio = _load_wav(fpath)
        emb   = extractor.extract(audio)
        r     = verifier.verify(emb, os.path.basename(fpath))
        results.append(r)
        icon  = {"MATCH": "✓", "LIKELY": "~", "UNCERTAIN": "?", "MISMATCH": "✗"}[r.result.value]
        note  = f"  <- {r.note}" if r.note else ""
        print(f"  {icon} {r.filename:<45} {r.similarity:>+7.4f}  [{r.result.value}]{note}")

    # Summary
    n = {v: sum(1 for r in results if r.result == v) for v in VerifyResult}
    print(f"\n  MATCH: {n[VerifyResult.MATCH]}  "
          f"LIKELY: {n[VerifyResult.LIKELY]}  "
          f"UNCERTAIN: {n[VerifyResult.UNCERTAIN]}  "
          f"MISMATCH: {n[VerifyResult.MISMATCH]}")

    flagged = [r for r in results if r.result in (VerifyResult.MISMATCH, VerifyResult.UNCERTAIN)]
    if flagged:
        print(f"\n  ALERT — non-enrolled voices detected:")
        for r in flagged:
            print(f"     -> {r.filename}  (sim={r.similarity:+.4f})")
    else:
        print(f"\n  All audio matches enrolled student.")

    # Report
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "stage3_report.txt")
    _write_report(results, enroll_dur, report_path)
    print(f"  Report -> {report_path}")

    return results


def _write_report(results, enroll_dur: float, out_path: str) -> None:
    lines = [
        "=" * 72,
        "STAGE 3 — SPEAKER VERIFICATION REPORT",
        "=" * 72,
        f"  Enrollment duration : {enroll_dur:.2f}s",
        f"  Files tested        : {len(results)}",
        f"  Thresholds          : MATCH>={THRESH_MATCH}  LIKELY>={THRESH_LIKELY}  UNCERTAIN>={THRESH_UNCERTAIN}",
        "",
        f"  {'File':<45} {'Similarity':>10}  {'Result':<12}  Conf  Note",
        "  " + "─" * 70,
    ]
    for r in results:
        icon = {"MATCH": "✓", "LIKELY": "~", "UNCERTAIN": "?", "MISMATCH": "✗"}[r.result.value]
        lines.append(
            f"  {icon} {r.filename:<45} {r.similarity:>+10.4f}  "
            f"{r.result.value:<12}  {r.confidence:.2f}"
            + (f"  {r.note}" if r.note else "")
        )
    n = {v: sum(1 for r in results if r.result == v) for v in VerifyResult}
    lines += [
        "",
        "── Summary " + "─" * 60,
        f"  MATCH     : {n[VerifyResult.MATCH]}",
        f"  LIKELY    : {n[VerifyResult.LIKELY]}",
        f"  UNCERTAIN : {n[VerifyResult.UNCERTAIN]}",
        f"  MISMATCH  : {n[VerifyResult.MISMATCH]}",
    ]
    flagged = [r for r in results if r.result in (VerifyResult.MISMATCH, VerifyResult.UNCERTAIN)]
    if flagged:
        lines += ["", "  ALERT — non-enrolled voices:"]
        for r in flagged:
            lines.append(f"     -> {r.filename}  (sim={r.similarity:+.4f})")
    lines.append("=" * 72)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 — Speaker Verification")
    parser.add_argument("--input",  default="inputStage3",
                        help="Folder with test WAVs (default: inputStage3)")
    parser.add_argument("--enroll", default=None,
                        help="Enrollment WAV path (default: enrollment.wav inside --input)")
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()

    enroll_path = args.enroll or os.path.join(args.input, ENROLLMENT_NAME)
    if not os.path.exists(enroll_path):
        sys.exit(f"Enrollment file not found: {enroll_path}")

    if not os.path.isdir(args.input):
        sys.exit(f"Input folder not found: {args.input}")

    enroll_basename = os.path.basename(enroll_path)
    test_wavs = sorted(
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if f.lower().endswith(".wav") and f != enroll_basename
    )
    if not test_wavs:
        sys.exit(f"No test WAV files found in {args.input}")

    run_stage3(test_wavs, enroll_path, args.output)
    print(f"\n  Output: ./{args.output}/\n")


if __name__ == "__main__":
    main()
