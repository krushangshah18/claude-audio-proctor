"""
Stage 2 Multi-Speaker Detection — Runner
=========================================
Usage examples:

  # Run on auto-generated test audio
  python run_stage2.py

  # Your own recording
  python run_stage2.py --file /path/to/recording.wav

  # Process a folder
  python run_stage2.py --dir ./inputStage2

  # Chain from Stage 1 output (speech-only WAV)
  python run_stage2.py --file output/my_recording_speech_only.wav

  # Tune thresholds
  python run_stage2.py --file rec.wav --a-threshold 0.70 --b-threshold 5.5

  # Regenerate test audio first
  python run_stage2.py --generate

Outputs (in ./output_stage2/):
  {name}_report.txt          — flag events + noise stats
  {name}_analysis.png        — 4-panel waveform + confidence plot
  SUMMARY.txt                — combined results across files

Note: Input WAV must be 16kHz mono.
  Convert: ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.noise_classifier   import NoiseClassifier, NoiseClassifierConfig
from core.scenario_a         import SimultaneousSpeechDetector
from core.scenario_b         import TurnBasedSpeakerDetector
from core.confidence_aggregator import ConfidenceAggregator
from core.output_builder2     import build_report
from core.speaker_splitter   import SpeakerSplitter
from core.visualizer2              import plot_stage2_analysis

AUDIO_DIR  = "audio_samples"
OUTPUT_DIR = "output_stage2"
SAMPLE_RATE = 16000
FRAME_SIZE  = 512    # 32ms @ 16kHz


# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "r") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n  = wf.getnframes()
        raw = wf.readframes(n)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    if sr != SAMPLE_RATE:
        print(f"  ⚠ Sample rate {sr}Hz detected — resampling to {SAMPLE_RATE}Hz")
        # Simple linear resampling
        ratio  = SAMPLE_RATE / sr
        new_len = int(len(audio) * ratio)
        audio  = np.interp(
            np.linspace(0, len(audio), new_len),
            np.arange(len(audio)),
            audio
        )

    return audio.astype(np.float32)


def process_file(
    wav_path:    str,
    a_threshold: float,
    b_threshold: float,
    out_dir:     str,
) -> dict:
    filename = os.path.basename(wav_path)
    stem     = os.path.splitext(filename)[0]

    print(f"\n{'─' * 62}")
    print(f"  Processing: {filename}")
    print(f"{'─' * 62}")

    audio    = load_wav(wav_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration  : {duration:.2f}s  |  Frames: {len(audio)//FRAME_SIZE}")

    # ── Initialise detectors ──────────────────────────────────────────
    noise_clf = NoiseClassifier(NoiseClassifierConfig(), sample_rate=SAMPLE_RATE)
    det_a     = SimultaneousSpeechDetector(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        confidence_threshold=a_threshold,
    )
    det_b     = TurnBasedSpeakerDetector(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        confidence_threshold=b_threshold,
    )
    splitter  = SpeakerSplitter(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
    )

    # ── Per-frame processing ──────────────────────────────────────────
    t0 = time.perf_counter()

    noise_scores:    list[int]   = []
    a_confidences:   list[float] = []
    b_zscores:       list[float] = []
    voice_frames     = 0
    noise_frames     = 0

    n_frames = len(audio) // FRAME_SIZE

    for i in range(n_frames):
        frame = audio[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]

        # Stage: noise gate
        nc = noise_clf.classify_frame(frame)
        noise_scores.append(nc.score)

        if not nc.is_voice:
            noise_frames += 1
            det_b.process_silence_frame()
            splitter.process_silence()
            a_confidences.append(0.0)
            b_zscores.append(0.0)
            continue

        voice_frames += 1

        # Scenario A
        res_a = det_a.process_frame(frame, frame_index=i)
        a_confidences.append(res_a.confidence)

        # Scenario B
        res_b = det_b.process_voiced_frame(frame)
        b_zscores.append(res_b.z_score)

        # Speaker splitter
        splitter.process_frame(frame)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rtf        = elapsed_ms / (duration * 1000)

    print(f"  Analysis  : {elapsed_ms:.1f}ms  (RTF {rtf:.4f})")
    print(f"  Voice     : {voice_frames} frames  |  Noise: {noise_frames} frames")

    # ── Speaker split WAVs (run BEFORE flag aggregation) ────────────────
    split_paths = splitter.split_audio(audio, out_dir, stem)
    spk_stats   = splitter.get_speaker_stats()
    n_speakers  = splitter.get_speaker_count()

    # ── Aggregate flags ───────────────────────────────────────────────
    # Scenario A: use speaker splitter ground truth instead of unreliable
    # harmonic comb detector (which false-fires on all single-mic audio).
    # If splitter found 2+ speakers → generate a Scenario A event.
    aggregator  = ConfidenceAggregator(
        confidence_threshold=min(a_threshold, b_threshold),
    )
    frame_dur   = FRAME_SIZE / SAMPLE_RATE

    if n_speakers >= 2:
        # Build a synthetic Scenario A event spanning the whole audio
        f0_list = [st["f0_mean_hz"] for st in spk_stats[:2]]
        events_a = [{
            "type":           "MULTIPLE_SPEAKERS_DETECTED",
            "scenario":       "A",
            "start_s":        0.0,
            "end_s":          round(duration, 3),
            "duration_s":     round(duration, 3),
            "confidence_max": 1.0,
            "confidence_avg": 1.0,
            "f0_voices_hz":   f0_list,
        }]
    else:
        events_a = []

    events_b    = det_b.get_flag_events()
    flag_events = aggregator.aggregate(events_a, events_b)
    summary     = aggregator.summarise(flag_events)

    print(f"  Flags     : {summary['total_flags']}  "
          f"(A={summary['scenario_a']}, B={summary['scenario_b']}, "
          f"AB={summary['scenario_ab']})")

    for ev in flag_events:
        print(f"    [{ev.severity}] {ev.event_type}  "
              f"{ev.start_s:.2f}s → {ev.end_s:.2f}s  "
              f"conf={ev.confidence:.3f}  scenario={ev.scenario}")

    os.makedirs(out_dir, exist_ok=True)

    # ── Report ────────────────────────────────────────────────────────
    noise_stats = {
        "total_frames": n_frames,
        "voice_frames": voice_frames,
        "noise_frames": noise_frames,
        "voice_pct":    100 * voice_frames / max(n_frames, 1),
        "noise_pct":    100 * noise_frames / max(n_frames, 1),
    }
    report_path = os.path.join(out_dir, f"{stem}_report.txt")
    build_report(filename, duration, flag_events, summary, noise_stats, report_path)
    print(f"  → report  : {stem}_report.txt")

    print(f"  Speakers  : {n_speakers} detected")
    for st in spk_stats:
        print(f"    Speaker {st['speaker_id']}  F0≈{st['f0_mean_hz']}Hz  "
              f"{st['duration_s']}s  → {stem}_speaker_{st['speaker_id']}.wav")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_path = os.path.join(out_dir, f"{stem}_analysis.png")
    plot_stage2_analysis(
        audio=audio,
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        scenario_a_confidences=a_confidences,
        scenario_b_zscores=b_zscores,
        noise_scores=noise_scores,
        flag_events=flag_events,
        stem=stem,
        output_path=plot_path,
    )

    return {
        "file":          filename,
        "duration_s":    duration,
        "total_flags":   summary["total_flags"],
        "high_severity": summary["high_severity"],
        "scenario_a":    summary["scenario_a"],
        "scenario_b":    summary["scenario_b"],
        "rtf":           rtf,
        "elapsed_ms":    elapsed_ms,
    }


def write_summary(results: list[dict], out_dir: str):
    lines = [
        "=" * 72,
        "STAGE 2 — MULTI-SPEAKER DETECTION SUMMARY",
        "=" * 72,
        "",
        f"  {'File':<36} {'Flags':>6} {'HIGH':>5} {'ScA':>5} {'ScB':>5} {'RTF':>8}",
        "  " + "─" * 66,
    ]
    for r in results:
        lines.append(
            f"  {r['file']:<36} {r['total_flags']:>6} "
            f"{r['high_severity']:>5} {r['scenario_a']:>5} "
            f"{r['scenario_b']:>5} {r['rtf']:>7.4f}"
        )

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    status  = "✓ EXCELLENT" if avg_rtf < 0.05 else ("~ ACCEPTABLE" if avg_rtf < 0.20 else "✗ SLOW")
    lines  += [
        "",
        f"  Average RTF : {avg_rtf:.4f}  ({status})",
        "  (RTF < 0.05 = 20× real-time — full CPU headroom for YOLO)",
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
        description="Stage 2 — Multi-speaker detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file",        type=str,   help="Single WAV file")
    parser.add_argument("--dir",         type=str,   help="Directory of WAV files")
    parser.add_argument("--generate",    action="store_true", help="Regenerate test audio first")
    parser.add_argument("--a-threshold", type=float, default=0.75,
                        help="Scenario A confidence threshold (default: 0.75)")
    parser.add_argument("--b-threshold", type=float, default=0.75,
                        help="Scenario B confidence threshold (default: 0.75)")
    parser.add_argument("--out",         type=str,   default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 62)
    print("  STAGE 2 — MULTI-SPEAKER DETECTION")
    print(f"  threshold A={args.a_threshold}  B={args.b_threshold}")
    print("=" * 62)

    if args.generate:
        print("\nGenerating test audio...")
        import subprocess
        subprocess.run([sys.executable, "generate_test_audio.py"], check=True)

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

    results = [
        process_file(f, args.a_threshold, args.b_threshold, args.out)
        for f in files
    ]
    if len(results) > 1:
        write_summary(results, args.out)

    print(f"\n{'═' * 62}")
    print(f"  Outputs in: ./{args.out}/")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()