"""
Stage 2 — Multi-Speaker Detection

Usage:
  python run_stage2.py --file recording.wav
  python run_stage2.py --dir ./inputStage1
  python run_stage2.py --file output_stage1/recording_speech_only.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.noise_classifier      import NoiseClassifier, NoiseClassifierConfig
from core.scenario_a            import SimultaneousSpeechDetector
from core.scenario_b            import TurnBasedSpeakerDetector
from core.confidence_aggregator import ConfidenceAggregator
from core.output_builder2       import build_report
from core.speaker_splitter      import SpeakerSplitter
from core.visualizer2           import plot_stage2_analysis

OUTPUT_DIR  = "output_stage2"
SAMPLE_RATE = 16000
FRAME_SIZE  = 512


def _load_wav(path: str) -> np.ndarray:
    with wave.open(path, "r") as wf:
        sr  = wf.getframerate()
        ch  = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * SAMPLE_RATE / sr)),
            np.arange(len(audio)), audio,
        )
    return audio.astype(np.float32)


def run_stage2(wav_path: str, out_dir: str) -> dict:
    """
    Run Stage 2 multi-speaker detection on a single WAV file.
    Returns a result dict including flag events, speaker stats, and speaker WAV paths.
    """
    filename = os.path.basename(wav_path)
    stem     = os.path.splitext(filename)[0]

    print(f"\n{'─' * 60}")
    print(f"  Stage 2 — {filename}")
    print(f"{'─' * 60}")

    audio    = _load_wav(wav_path)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration  : {duration:.2f}s  |  Frames: {len(audio) // FRAME_SIZE}")

    noise_clf = NoiseClassifier(NoiseClassifierConfig(), sample_rate=SAMPLE_RATE)
    det_a     = SimultaneousSpeechDetector(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)
    det_b     = TurnBasedSpeakerDetector(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)
    splitter  = SpeakerSplitter(sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE)

    t0 = time.perf_counter()
    noise_scores, a_confidences, b_zscores = [], [], []
    voice_frames = noise_frames = 0

    for i in range(len(audio) // FRAME_SIZE):
        frame = audio[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        nc    = noise_clf.classify_frame(frame)
        noise_scores.append(nc.score)

        if not nc.is_voice:
            noise_frames += 1
            det_b.process_silence_frame()
            splitter.process_silence()
            a_confidences.append(0.0)
            b_zscores.append(0.0)
            continue

        voice_frames += 1
        a_confidences.append(det_a.process_frame(frame, frame_index=i).confidence)
        b_zscores.append(det_b.process_voiced_frame(frame).z_score)
        splitter.process_frame(frame)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    rtf        = elapsed_ms / (duration * 1000)
    print(f"  RTF       : {rtf:.4f}  |  Voice: {voice_frames}  Noise: {noise_frames} frames")

    os.makedirs(out_dir, exist_ok=True)

    split_paths = splitter.split_audio(audio, out_dir, stem)
    spk_stats   = splitter.get_speaker_stats()
    n_speakers  = splitter.get_speaker_count()

    aggregator = ConfidenceAggregator()
    events_a   = [{
        "type": "MULTIPLE_SPEAKERS_DETECTED", "scenario": "A",
        "start_s": 0.0, "end_s": round(duration, 3), "duration_s": round(duration, 3),
        "confidence_max": 1.0, "confidence_avg": 1.0,
        "f0_voices_hz": [st["f0_mean_hz"] for st in spk_stats[:2]],
    }] if n_speakers >= 2 else []

    flag_events = aggregator.aggregate(events_a, det_b.get_flag_events())
    summary     = aggregator.summarise(flag_events)

    print(f"  Speakers  : {n_speakers}  |  Flags: {summary['total_flags']} "
          f"(A={summary['scenario_a']}, B={summary['scenario_b']})")
    for ev in flag_events:
        print(f"    [{ev.severity}] {ev.event_type}  "
              f"{ev.start_s:.2f}s → {ev.end_s:.2f}s  conf={ev.confidence:.3f}")

    noise_stats = {
        "total_frames": len(audio) // FRAME_SIZE,
        "voice_frames": voice_frames, "noise_frames": noise_frames,
        "voice_pct": 100 * voice_frames / max(len(audio) // FRAME_SIZE, 1),
        "noise_pct": 100 * noise_frames / max(len(audio) // FRAME_SIZE, 1),
    }
    build_report(filename, duration, flag_events, summary, noise_stats,
                 os.path.join(out_dir, f"{stem}_report.txt"))

    plot_stage2_analysis(
        audio=audio, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE,
        scenario_a_confidences=a_confidences, scenario_b_zscores=b_zscores,
        noise_scores=noise_scores, flag_events=flag_events,
        stem=stem, output_path=os.path.join(out_dir, f"{stem}_analysis.png"),
    )

    return {
        "file":         filename,
        "stem":         stem,
        "duration_s":   duration,
        "n_speakers":   n_speakers,
        "spk_stats":    spk_stats,
        "speaker_wavs": split_paths,
        "flag_events":  flag_events,
        "summary":      summary,
        "rtf":          rtf,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 — Multi-Speaker Detection")
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

    for f in files:
        run_stage2(f, args.out)

    print(f"\n  Outputs in: ./{args.out}/\n")


if __name__ == "__main__":
    main()
