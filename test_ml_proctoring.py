"""
test_ml_proctoring.py
=====================
Test harness for the ml_proctoring AI/ML pipeline.

Modes:
  --mode file   : process a WAV file as test input
  --mode live   : capture from microphone in real time

Usage examples:
  # File mode — enrollment + test file
  python test_ml_proctoring.py --mode file --enroll ./Krushangenroll.wav --test   test_audio.wav

  # Live mode — enrollment from file, test from mic
  python test_ml_proctoring.py --mode live \
      --enroll ./Krushangenroll.wav \
      --duration 30

  # Live mode — enrollment AND test both from mic (two separate recordings)
  python test_ml_proctoring.py --mode live \
      --enroll-live \
      --enroll-duration 10 \
      --duration 30

Cheat-event audio clips are saved automatically to --out-dir (default: ml_proof_clips/).
Each file is named:  001_IMPERSONATION_12.3s.wav
"""

import argparse
import sys
import time
import queue
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd

from ml_proctoring import ProctorSession, CheatEvent

# ── Constants ─────────────────────────────────────────────────────────────────
SR    = 16000
CHUNK = 512       # 32ms per push()

# ANSI colours
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_wav(path: str) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        from ml_proctoring.audio_utils import resample
        audio = resample(audio, file_sr, SR)
    return audio.astype(np.float32)


def record_mic(duration_s: float, label: str = "Recording") -> np.ndarray:
    n = int(duration_s * SR)
    buf = np.zeros(n, dtype=np.float32)
    print(f"\n{_CYAN}[ {label} — {duration_s:.0f}s ]{_RESET}  Speak now...")
    sd.rec(n, samplerate=SR, channels=1, dtype="float32", out=buf)
    for remaining in range(int(duration_s), 0, -1):
        print(f"  {remaining:3d}s remaining...", end="\r", flush=True)
        time.sleep(1)
    sd.wait()
    print("  Done.                    ")
    return buf.flatten()


def save_proof(ev: CheatEvent, out_dir: Path, index: int) -> Path:
    """Write ev.audio_proof to a WAV file. Returns the saved path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{index:03d}_{ev.event_type.value}_{ev.timestamp_s:.1f}s.wav"
    fpath = out_dir / fname
    with open(fpath, "wb") as f:
        f.write(ev.audio_proof)
    return fpath


def print_event(ev: CheatEvent, saved_path: Path) -> None:
    colour = _RED if ev.confidence >= 0.7 else _YELLOW
    print(
        f"\n{colour}{_BOLD}  ⚠  CHEAT EVENT DETECTED{_RESET}\n"
        f"     type        : {ev.event_type.value}\n"
        f"     confidence  : {ev.confidence:.3f}\n"
        f"     timestamp   : {ev.timestamp_s:.1f}s\n"
        f"     verify_score: {ev.verify_score:.4f}\n"
        f"     n_speakers  : {ev.n_speakers}\n"
        f"     lip_active  : {ev.lip_active}\n"
        f"     audio clip  : {_CYAN}{saved_path}{_RESET}  "
        f"({len(ev.audio_proof)/1024:.1f} KB)\n"
    )


def print_summary(session: ProctorSession, out_dir: Path) -> None:
    s = session.get_summary()
    print(f"\n{'─'*60}")
    print(f"{_BOLD}  Session Summary{_RESET}")
    print(f"{'─'*60}")
    print(f"  Chunks processed : {s['total_chunks_processed']}")
    print(f"  Speech detected  : {s['total_speech_s']:.1f}s")
    print(f"  Speakers detected: {s['n_speakers_detected']}")
    print(f"  Verify score mean: {s['verify_score_mean']:.4f}")
    print(f"  Verify score min : {s['verify_score_min']:.4f}")
    print(f"  Cheat events     : {len(s['cheat_events'])}")
    if s['cheat_events']:
        print(f"\n  Saved audio clips → {_CYAN}{out_dir}/{_RESET}")
        for i, ev in enumerate(s['cheat_events'], 1):
            fname = f"{i:03d}_{ev.event_type.value}_{ev.timestamp_s:.1f}s.wav"
            print(f"    [{ev.timestamp_s:6.1f}s] {ev.event_type.value:<20} "
                  f"conf={ev.confidence:.3f}  → {fname}")
    print(f"{'─'*60}\n")


# ── File mode ─────────────────────────────────────────────────────────────────

def run_file_mode(args):
    out_dir = Path(args.out_dir)
    print(f"\n{_BOLD}ml_proctoring — FILE MODE{_RESET}")
    print(f"  Enrollment : {args.enroll}")
    print(f"  Test file  : {args.test}")
    print(f"  Clips out  : {out_dir}/\n")

    session = ProctorSession()

    print("  Enrolling speaker...", end=" ", flush=True)
    session.enroll(load_wav(args.enroll))
    print(f"{_GREEN}OK{_RESET}")

    test_audio   = load_wav(args.test)
    total_chunks = len(test_audio) // CHUNK
    print(f"  Test audio : {len(test_audio)/SR:.1f}s  ({total_chunks} chunks)\n")
    print(f"{'─'*60}")
    print(f"  {'Time':>6}  {'Score':>7}  {'Spk':>4}  {'Events':>6}")
    print(f"{'─'*60}")

    event_count = 0
    for i in range(total_chunks):
        chunk = test_audio[i * CHUNK : (i + 1) * CHUNK]
        evs   = session.push(chunk, lip_activity=not args.no_lips)

        for ev in evs:
            event_count += 1
            path = save_proof(ev, out_dir, event_count)
            print_event(ev, path)

        if i % 50 == 0:
            s = session.get_summary()
            print(
                f"  {i*CHUNK/SR:>5.1f}s  {s['verify_score_mean']:>7.4f}"
                f"  {s['n_speakers_detected']:>4}  {event_count:>6}",
                end="\r"
            )

    print()
    print_summary(session, out_dir)


# ── Live mode ─────────────────────────────────────────────────────────────────

def run_live_mode(args):
    out_dir = Path(args.out_dir)
    print(f"\n{_BOLD}ml_proctoring — LIVE MODE{_RESET}")
    print(f"  Clips out  : {out_dir}/\n")

    session = ProctorSession()

    if args.enroll_live:
        enroll_audio = record_mic(args.enroll_duration, "ENROLLMENT")
    else:
        print(f"  Loading enrollment: {args.enroll}")
        enroll_audio = load_wav(args.enroll)

    print("  Enrolling speaker...", end=" ", flush=True)
    session.enroll(enroll_audio)
    print(f"{_GREEN}OK{_RESET}")

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def mic_callback(indata, _frames, _time, status):
        if status:
            print(f"  [mic] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy().astype(np.float32))

    print(f"\n  Proctoring for {args.duration}s  "
          f"({'lips=OFF' if args.no_lips else 'lips=ON (simulated)'})")
    print(f"{'─'*60}")
    print(f"  {'Time':>6}  {'Speech':>7}  {'Score':>7}  {'Spk':>4}  {'Events':>6}")
    print(f"{'─'*60}")

    start       = time.monotonic()
    event_count = 0
    chunk_buf   = np.zeros(0, dtype=np.float32)

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=CHUNK, callback=mic_callback):
        while time.monotonic() - start < args.duration:
            try:
                raw = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            chunk_buf = np.concatenate([chunk_buf, raw])
            while len(chunk_buf) >= CHUNK:
                chunk     = chunk_buf[:CHUNK]
                chunk_buf = chunk_buf[CHUNK:]
                evs       = session.push(chunk, lip_activity=not args.no_lips)

                for ev in evs:
                    event_count += 1
                    path = save_proof(ev, out_dir, event_count)
                    print_event(ev, path)

            elapsed   = time.monotonic() - start
            s         = session.get_summary()
            remaining = args.duration - elapsed
            print(
                f"  {elapsed:>5.1f}s  {s['total_speech_s']:>6.1f}s"
                f"  {s['verify_score_mean']:>7.4f}"
                f"  {s['n_speakers_detected']:>4}  {event_count:>6}"
                f"  [{remaining:.0f}s left]",
                end="\r", flush=True
            )

    print()
    print_summary(session, out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Test ml_proctoring pipeline — saves cheat-event audio clips"
    )
    p.add_argument("--mode", choices=["file", "live"], default="file")

    # Enrollment
    p.add_argument("--enroll", default=None,
                   help="Path to enrollment WAV file")
    p.add_argument("--enroll-live", action="store_true",
                   help="Record enrollment from mic")
    p.add_argument("--enroll-duration", type=float, default=10.0,
                   help="Seconds for live enrollment (default: 10)")

    # File mode
    p.add_argument("--test", default=None,
                   help="[file mode] Path to test WAV file")

    # Live mode
    p.add_argument("--duration", type=float, default=30.0,
                   help="[live mode] Seconds to record (default: 30)")

    # Shared
    p.add_argument("--no-lips", action="store_true",
                   help="Simulate lips=False (exercises IMPERSONATION / GHOST_VOICE)")
    p.add_argument("--out-dir", default="ml_proof_clips",
                   help="Folder to save cheat-event audio clips (default: ml_proof_clips/)")

    args = p.parse_args()

    if args.mode == "file":
        if not args.enroll and not args.enroll_live:
            p.error("--mode file requires --enroll <wav>")
        if not args.test:
            p.error("--mode file requires --test <wav>")
    else:
        if not args.enroll and not args.enroll_live:
            p.error("--mode live requires --enroll <wav> or --enroll-live")

    if args.mode == "file":
        run_file_mode(args)
    else:
        run_live_mode(args)


if __name__ == "__main__":
    main()
