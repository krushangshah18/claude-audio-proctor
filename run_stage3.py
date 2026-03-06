"""
Stage 3 — Speaker Verification
================================
Usage:
  python run_stage3.py                          # uses ./inputStage3/
  python run_stage3.py --input /path/to/folder  # custom input folder

Input folder layout:
  inputStage3/
    enrollment.wav            ← student reference (required, any length ≥ 3s)
    KrushangSample_speaker_0.wav   ← Stage 2 split outputs to verify
    KrushangSample_speaker_1.wav
    any_other.wav
    ...

Output:
  Console table — filename + similarity score + MATCH/LIKELY/UNCERTAIN/MISMATCH
  output_stage3/stage3_report.txt

Similarity thresholds:
  ≥ 0.82  MATCH      — confidently the enrolled student
  ≥ 0.68  LIKELY     — probably the enrolled student
  ≥ 0.50  UNCERTAIN  — borderline, flag for review
  <  0.50 MISMATCH   — different person / unknown speaker
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.embedding_extractor import EmbeddingExtractor, EMBED_DIM
from core.verifier             import (SpeakerVerifier, VerifyResult,
                                        THRESH_MATCH, THRESH_LIKELY, THRESH_UNCERTAIN)

OUTPUT_DIR      = "output_stage3"
ENROLLMENT_NAME = "enrollment.wav"
SAMPLE_RATE     = 16000


# ─────────────────────────────────────────────────────────────────────────────
#  Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str) -> np.ndarray:
    """WAV → float32 mono array at 16kHz."""
    with wave.open(path) as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        sw  = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = np.int16 if sw == 2 else np.int8
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= (32768.0 if sw == 2 else 128.0)

    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    if sr != SAMPLE_RATE:
        new_len = int(len(audio) * SAMPLE_RATE / sr)
        audio   = np.interp(
            np.linspace(0, len(audio), new_len),
            np.arange(len(audio)), audio,
        ).astype(np.float32)

    return audio


# ─────────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(results, enroll_dur: float, out_path: str) -> None:
    lines = [
        "=" * 72,
        "STAGE 3 — SPEAKER VERIFICATION REPORT",
        "=" * 72,
        f"  Enrollment duration : {enroll_dur:.2f}s",
        f"  Files tested        : {len(results)}",
        f"  Thresholds          : MATCH≥{THRESH_MATCH}  "
        f"LIKELY≥{THRESH_LIKELY}  UNCERTAIN≥{THRESH_UNCERTAIN}",
        "",
        f"  {'File':<45} {'Similarity':>10}  {'Result':<12}  Conf  Note",
        "  " + "─" * 70,
    ]

    for r in results:
        icon = {"MATCH":"✓","LIKELY":"~","UNCERTAIN":"?","MISMATCH":"✗"}[r.result.value]
        lines.append(
            f"  {icon} {r.filename:<45} {r.similarity:>+10.4f}  "
            f"{r.result.value:<12}  {r.confidence:.2f}"
            + (f"  {r.note}" if r.note else "")
        )

    n = {v: sum(1 for r in results if r.result == v) for v in VerifyResult}
    lines += [
        "",
        "── Summary " + "─" * 60,
        f"  ✓ MATCH     : {n[VerifyResult.MATCH]}",
        f"  ~ LIKELY    : {n[VerifyResult.LIKELY]}",
        f"  ? UNCERTAIN : {n[VerifyResult.UNCERTAIN]}",
        f"  ✗ MISMATCH  : {n[VerifyResult.MISMATCH]}",
    ]

    flagged = [r for r in results
               if r.result in (VerifyResult.MISMATCH, VerifyResult.UNCERTAIN)]
    if flagged:
        lines += ["", "  ⚠  ALERT — non-enrolled voices:"]
        for r in flagged:
            lines.append(f"     → {r.filename}  (sim={r.similarity:+.4f})")

    lines.append("=" * 72)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run(input_dir: str, out_dir: str) -> list:
    print("=" * 62)
    print("  STAGE 3 — SPEAKER VERIFICATION")
    print(f"  Input  : {input_dir}")
    print(f"  Thresh : MATCH≥{THRESH_MATCH}  "
          f"LIKELY≥{THRESH_LIKELY}  UNCERTAIN≥{THRESH_UNCERTAIN}")
    print("=" * 62)

    # ── Validate input folder ─────────────────────────────────────────
    enroll_path = os.path.join(input_dir, ENROLLMENT_NAME)
    if not os.path.exists(enroll_path):
        print(f"\n  ✗ ERROR: {enroll_path} not found")
        print(f"    Put your enrollment recording as 'enrollment.wav' "
              f"in {input_dir}/")
        sys.exit(1)

    test_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".wav") and f != ENROLLMENT_NAME
    ])
    if not test_files:
        print(f"\n  ✗ No test WAV files found in {input_dir} "
              f"(besides enrollment.wav)")
        sys.exit(1)

    extractor = EmbeddingExtractor(sample_rate=SAMPLE_RATE)
    verifier  = SpeakerVerifier()

    # ── Enrollment ────────────────────────────────────────────────────
    print(f"\n  ┌─ Enrollment ───────────────────────────────────────")
    t0          = time.perf_counter()
    enroll_audio = load_wav(enroll_path)
    enroll_dur   = len(enroll_audio) / SAMPLE_RATE

    # Split enrollment into 5s chunks and average embeddings
    # → more robust centroid, handles recording-level variation
    CHUNK = SAMPLE_RATE * 5
    chunks = [enroll_audio[i:i+CHUNK] for i in range(0, len(enroll_audio), CHUNK)
              if len(enroll_audio[i:i+CHUNK]) >= SAMPLE_RATE * 2]
    if not chunks:
        chunks = [enroll_audio]

    for chunk in chunks:
        emb = extractor.extract(chunk)
        verifier.add_enrollment(emb)
    verifier.finalize_enrollment()

    enroll_ms = (time.perf_counter() - t0) * 1000
    print(f"  │  File      : {ENROLLMENT_NAME}  ({enroll_dur:.1f}s)")
    print(f"  │  Chunks    : {len(chunks)} × 5s")
    print(f"  │  Embedding : {EMBED_DIM}-dim  [{enroll_ms:.0f}ms]")
    print(f"  └────────────────────────────────────────────────────")

    # ── Verify test files ─────────────────────────────────────────────
    print(f"\n  {'File':<45} {'Sim':>7}  Result")
    print(f"  {'─'*62}")

    results = []
    total_ms = 0.0
    for fname in test_files:
        fpath = os.path.join(input_dir, fname)
        audio = load_wav(fpath)
        dur   = len(audio) / SAMPLE_RATE

        t0  = time.perf_counter()
        emb = extractor.extract(audio)
        r   = verifier.verify(emb, fname)
        ms  = (time.perf_counter() - t0) * 1000
        total_ms += ms

        results.append(r)
        icon = {"MATCH":"✓","LIKELY":"~","UNCERTAIN":"?","MISMATCH":"✗"}[r.result.value]
        note = f"  ← {r.note}" if r.note else ""
        print(f"  {icon} {fname:<45} {r.similarity:>+7.4f}  "
              f"[{r.result.value}]{note}")

    # ── Summary ───────────────────────────────────────────────────────
    n = {v: sum(1 for r in results if r.result == v) for v in VerifyResult}
    print(f"\n  {'═'*62}")
    print(f"  RESULTS")
    print(f"  ✓ MATCH     : {n[VerifyResult.MATCH]:<3} — enrolled student (confident)")
    print(f"  ~ LIKELY    : {n[VerifyResult.LIKELY]:<3} — enrolled student (probable)")
    print(f"  ? UNCERTAIN : {n[VerifyResult.UNCERTAIN]:<3} — borderline, needs review")
    print(f"  ✗ MISMATCH  : {n[VerifyResult.MISMATCH]:<3} — unknown / different speaker")

    flagged = [r for r in results
               if r.result in (VerifyResult.MISMATCH, VerifyResult.UNCERTAIN)]
    if flagged:
        print(f"\n  ⚠  ALERT — non-enrolled audio detected:")
        for r in flagged:
            print(f"     → {r.filename}  (sim={r.similarity:+.4f})")
    else:
        print(f"\n  ✓  All audio matches enrolled student.")

    print(f"\n  Avg embedding time : {total_ms/max(len(results),1):.1f}ms/file")

    # ── Write report ──────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "stage3_report.txt")
    write_report(results, enroll_dur, report_path)
    print(f"  Report → {report_path}")
    print(f"  {'═'*62}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 3 — Speaker Verification")
    parser.add_argument("--input",  default="inputStage3",
                        help="Input folder with enrollment.wav + test wavs")
    parser.add_argument("--output", default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
