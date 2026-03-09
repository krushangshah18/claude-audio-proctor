"""
Stage 3 — Speaker Verification

Usage:
  python run_stage3.py --input inputStage3
  python run_stage3.py --input inputStage3 --enroll /path/to/enrollment.wav

  inputStage3/ must contain:
    enrollment.wav      <- reference recording (>= 5s recommended)
    *_speaker_0.wav     <- Stage 2 split outputs to verify
    *_speaker_1.wav
    ...

Fused score = 0.75 * F0_cosine + 0.25 * MFCC_variance_ratio
  >= 0.76  MATCH      — enrolled student (confident)
  >= 0.58  LIKELY     — enrolled student (probable)
  >= 0.40  UNCERTAIN  — borderline, flag for review
  <  0.40  MISMATCH   — different person / unknown speaker
"""

from __future__ import annotations
import argparse, os, sys, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.embedding_extractor import EmbeddingExtractor, EMBED_DIM, W_F0, W_VAR
from core.verifier import (SpeakerVerifier, VerifyResult,
                            THRESH_MATCH, THRESH_LIKELY, THRESH_UNCERTAIN)

OUTPUT_DIR      = "output_stage3"
ENROLLMENT_NAME = "enrollment.wav"
SAMPLE_RATE     = 16000


def _load_wav(path: str) -> np.ndarray:
    with wave.open(path) as wf:
        sr=wf.getframerate(); ch=wf.getnchannels(); sw=wf.getsampwidth()
        raw=wf.readframes(wf.getnframes())
    a = np.frombuffer(raw, np.int16 if sw==2 else np.int8).astype(np.float32)
    a /= 32768.0 if sw==2 else 128.0
    if ch==2: a=a.reshape(-1,2).mean(1)
    if sr!=SAMPLE_RATE:
        a=np.interp(np.linspace(0,len(a),int(len(a)*SAMPLE_RATE/sr)),
                    np.arange(len(a)),a).astype(np.float32)
    return a


def _f0_median(audio: np.ndarray, sr: int=SAMPLE_RATE) -> float:
    HOP,FRAME=160,512
    f0s=[]
    for i in range((len(audio)-FRAME)//HOP):
        seg=audio[i*HOP:i*HOP+FRAME]
        if np.sqrt(np.mean(seg**2))<0.015: continue
        win=seg*np.hanning(FRAME)
        ac=np.correlate(win,win,'full')[FRAME-1:]; ac/=ac[0]+1e-10
        p=int(np.argmax(ac[40:267]))+40
        if ac[p]>0.25: f0s.append(sr/p)
    return float(np.median(f0s)) if f0s else 0.0


def run_stage3(test_wavs: list[str], enroll_path: str, out_dir: str) -> list:
    print(f"\n{'─'*66}")
    print(f"  Stage 3 — Speaker Verification  "
          f"[F0×{W_F0} + VarRatio×{W_VAR}]")
    print(f"  Thresholds: MATCH≥{THRESH_MATCH}  "
          f"LIKELY≥{THRESH_LIKELY}  UNCERTAIN≥{THRESH_UNCERTAIN}")
    print(f"{'─'*66}")

    ext = EmbeddingExtractor(sample_rate=SAMPLE_RATE)
    ver = SpeakerVerifier()

    # ── Enrollment ────────────────────────────────────────────────────
    enroll_audio = _load_wav(enroll_path)
    enroll_dur   = len(enroll_audio) / SAMPLE_RATE
    CHUNK = SAMPLE_RATE * 5
    chunks = [enroll_audio[i:i+CHUNK] for i in range(0,len(enroll_audio),CHUNK)
              if len(enroll_audio[i:i+CHUNK]) >= SAMPLE_RATE*2]
    if not chunks: chunks=[enroll_audio]

    for c in chunks:
        ver.add_enrollment(ext.extract(c))
    ver.finalize_enrollment()

    print(f"\n  Enrollment : {os.path.basename(enroll_path)}")
    print(f"  Duration   : {enroll_dur:.1f}s  ({len(chunks)} chunks)")
    print(f"  F0 median  : {_f0_median(enroll_audio):.0f}Hz")

    # ── Verify ────────────────────────────────────────────────────────
    print(f"\n  {'File':<47} {'Score':>6}  {'F0':>6}  {'VarR':>6}  Result")
    print(f"  {'─'*74}")
    results=[]
    for fpath in test_wavs:
        audio=_load_wav(fpath)
        r=ver.verify(ext.extract(audio), os.path.basename(fpath))
        results.append(r)
        icon={"MATCH":"✓","LIKELY":"~","UNCERTAIN":"?","MISMATCH":"✗"}[r.result.value]
        note=f"  <- {r.note}" if r.note else ""
        print(f"  {icon} {r.filename:<47} {r.similarity:>6.3f}  "
              f"{r.sim_f0:>6.3f}  {r.sim_var:>6.3f}  [{r.result.value}]{note}")

    # ── Summary ───────────────────────────────────────────────────────
    n={v:sum(1 for r in results if r.result==v) for v in VerifyResult}
    print(f"\n  MATCH: {n[VerifyResult.MATCH]}  LIKELY: {n[VerifyResult.LIKELY]}  "
          f"UNCERTAIN: {n[VerifyResult.UNCERTAIN]}  MISMATCH: {n[VerifyResult.MISMATCH]}")

    flagged=[r for r in results if r.result in (VerifyResult.MISMATCH, VerifyResult.UNCERTAIN)]
    if flagged:
        print(f"\n  ⚠  ALERT — non-enrolled voices detected:")
        for r in flagged:
            print(f"     -> {r.filename}")
            print(f"        score={r.similarity:.3f}  f0_sim={r.sim_f0:.3f}  var_ratio={r.sim_var:.3f}")
    else:
        print(f"\n  ✓  All audio matches enrolled student.")

    os.makedirs(out_dir, exist_ok=True)
    report=os.path.join(out_dir,"stage3_report.txt")
    _write_report(results, enroll_dur, enroll_path, report)
    print(f"\n  Report -> {report}")
    print(f"{'─'*66}")
    return results


def _write_report(results, enroll_dur, enroll_path, out_path):
    lines=["="*72,"STAGE 3 — SPEAKER VERIFICATION REPORT","="*72,
           f"  Enrollment : {os.path.basename(enroll_path)}  ({enroll_dur:.1f}s)",
           f"  Method     : F0 histogram (×{W_F0}) + MFCC variance ratio (×{W_VAR})",
           f"  Thresholds : MATCH≥{THRESH_MATCH}  LIKELY≥{THRESH_LIKELY}  UNCERTAIN≥{THRESH_UNCERTAIN}",
           f"  Files      : {len(results)}","",
           f"  {'File':<45} {'Score':>6} {'F0':>6} {'VarR':>6}  {'Result':<10}  Note",
           "  "+"─"*76]
    for r in results:
        icon={"MATCH":"✓","LIKELY":"~","UNCERTAIN":"?","MISMATCH":"✗"}[r.result.value]
        lines.append(
            f"  {icon} {r.filename:<45} {r.similarity:>6.3f} {r.sim_f0:>6.3f} "
            f"{r.sim_var:>6.3f}  {r.result.value:<10}"+(f"  {r.note}" if r.note else ""))
    n={v:sum(1 for r in results if r.result==v) for v in VerifyResult}
    lines+=["","── Summary "+"─"*60,
            f"  MATCH     : {n[VerifyResult.MATCH]}",
            f"  LIKELY    : {n[VerifyResult.LIKELY]}",
            f"  UNCERTAIN : {n[VerifyResult.UNCERTAIN]}",
            f"  MISMATCH  : {n[VerifyResult.MISMATCH]}"]
    flagged=[r for r in results if r.result in (VerifyResult.MISMATCH,VerifyResult.UNCERTAIN)]
    if flagged:
        lines+=["","  ALERT:"]
        for r in flagged:
            lines.append(f"     -> {r.filename}  score={r.similarity:.3f}  f0={r.sim_f0:.3f}")
    lines.append("="*72)
    with open(out_path,"w") as f: f.write("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",  default="inputStage3")
    p.add_argument("--enroll", default=None)
    p.add_argument("--output", default=OUTPUT_DIR)
    args=p.parse_args()

    ep=args.enroll or os.path.join(args.input, ENROLLMENT_NAME)
    if not os.path.exists(ep): sys.exit(f"Enrollment not found: {ep}")
    if not os.path.isdir(args.input): sys.exit(f"Input folder not found: {args.input}")

    enroll_base=os.path.basename(ep)
    wavs=sorted(os.path.join(args.input,f) for f in os.listdir(args.input)
                if f.lower().endswith(".wav") and f!=enroll_base)
    if not wavs: sys.exit(f"No test WAVs in {args.input}")

    run_stage3(wavs, ep, args.output)
    print(f"  Output: ./{args.output}/\n")


if __name__=="__main__":
    main()