"""
Speaker Splitter v3
===================
Segment-first speaker diarization. Writes one WAV per detected speaker,
original-length with silence where that speaker wasn't talking.

Key design decision in v3:
  Segmentation uses DIRECT ENERGY THRESHOLDING on the audio,
  NOT the noise classifier's voiced_mask. The noise classifier is tuned
  for passing frames to the detectors (low false negatives), which means
  it passes nearly all frames including quiet inter-turn gaps. For
  segmentation we need the opposite: a strict silence gate.

  Adaptive threshold: median RMS of full audio × SILENCE_RATIO.
  This auto-adjusts per recording — no manual tuning needed.

Pipeline:
  1. Compute per-frame RMS energy
  2. Adaptive silence threshold = median(rms) × SILENCE_RATIO
  3. Build segments: contiguous voiced runs separated by SILENCE_GAP_FRAMES
  4. Extract 30-dim features per segment (13 MFCCs + delta + F0 + centroid)
  5. Agglomerative cosine+F0 clustering → speaker IDs
  6. Write per-speaker WAVs
"""

from __future__ import annotations
import os, wave
import numpy as np
from dataclasses import dataclass, field

SAMPLE_RATE  = 16000
FRAME_SIZE   = 512
N_MFCC       = 13

# Silence detection: frames with RMS < (median_rms * SILENCE_RATIO) = silence
# 0.5 means "below half the median energy" = silence/pause
SILENCE_RATIO         = 0.65
SILENCE_GAP_FRAMES    = 6     # ~192ms consecutive silence = segment boundary
MIN_SEGMENT_FRAMES    = 10    # ~320ms minimum segment to use for clustering

# Clustering: cosine+F0 combined similarity
# Fixed absolute F0 range — DO NOT use recording-relative range
# Human voice span: ~60Hz (low bass) to ~400Hz (high female) = 340Hz
# Same speaker (normal variation Δ<50Hz): combined ~0.95+
# Different speakers (Δ>80Hz):            combined ~0.87
# Threshold 0.91 sits cleanly in the gap
F0_RANGE_HZ     = 340.0
MERGE_THRESHOLD = 0.92


@dataclass
class AudioSegment:
    index:       int
    start_frame: int
    end_frame:   int
    speaker_id:  int = -1
    features:    np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_frames(self):     return self.end_frame - self.start_frame
    @property
    def start_sample(self): return self.start_frame * FRAME_SIZE
    @property
    def end_sample(self):   return self.end_frame   * FRAME_SIZE


class SpeakerSplitter:

    def __init__(self, sample_rate=SAMPLE_RATE, frame_size=FRAME_SIZE,
                 merge_threshold=MERGE_THRESHOLD):
        self.sr              = sample_rate
        self.frame_size      = frame_size
        self.merge_threshold = merge_threshold
        # voiced_mask kept for API compatibility with run_stage2.py
        self._voiced_mask:  list[bool]         = []
        self._segments:     list[AudioSegment] = []
        self._n_speakers:   int                = 0

    # ── Public API ─────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> None:
        self._voiced_mask.append(True)

    def process_silence(self) -> None:
        self._voiced_mask.append(False)

    def split_audio(self, audio: np.ndarray, out_dir: str, stem: str) -> list[str]:
        # Step 1: energy-based segmentation (ignores noise classifier mask)
        self._segments = self._energy_segments(audio)
        if not self._segments:
            print("  ⚠ No speech segments found for splitting")
            return []

        print(f"  Segments  : {len(self._segments)} speech segments found")

        # Step 2: extract features per segment
        for seg in self._segments:
            seg_audio = audio[seg.start_sample : seg.end_sample]
            seg.features = self._extract_segment_features(seg_audio)

        # Step 3: cluster
        self._cluster_segments()
        self._n_speakers = len(set(s.speaker_id for s in self._segments if s.speaker_id >= 0))
        print(f"  Speakers  : {self._n_speakers} detected")

        os.makedirs(out_dir, exist_ok=True)
        return self._write_speaker_wavs(audio, out_dir, stem)

    def get_speaker_count(self) -> int:
        return self._n_speakers

    def get_speaker_stats(self) -> list[dict]:
        stats: dict[int, dict] = {}
        for seg in self._segments:
            sid = seg.speaker_id
            if sid < 0: continue
            if sid not in stats:
                stats[sid] = {"speaker_id": sid, "frame_count": 0, "n_segments": 0}
            stats[sid]["frame_count"] += seg.n_frames
            stats[sid]["n_segments"]  += 1
        result = []
        for sid in sorted(stats):
            d = stats[sid]
            d["duration_s"] = round(d["frame_count"] * self.frame_size / self.sr, 2)
            spk_segs = [s for s in self._segments if s.speaker_id == sid]
            f0s = [s.features[-5] for s in spk_segs if len(s.features) > 4 and s.features[-5] > 0]
            d["f0_mean_hz"] = round(float(np.median(f0s)), 1) if f0s else 0.0
            result.append(d)
        return result

    # ── Step 1: energy-based segmentation ──────────────────────────────

    def _energy_segments(self, audio: np.ndarray) -> list[AudioSegment]:
        """
        Build segments purely from RMS energy.
        Threshold = median_rms × SILENCE_RATIO — adapts to each recording.
        """
        n_frames  = len(audio) // self.frame_size
        rms_vals  = np.array([
            float(np.sqrt(np.mean(audio[i*self.frame_size:(i+1)*self.frame_size]**2)))
            for i in range(n_frames)
        ])

        # Adaptive threshold: only use frames that have some energy for median
        nonzero = rms_vals[rms_vals > 0.001]
        median_rms = float(np.median(nonzero)) if len(nonzero) > 0 else 0.02
        threshold  = median_rms * SILENCE_RATIO

        print(f"  Energy    : median_rms={median_rms:.4f}  silence_thresh={threshold:.4f}")

        segments: list[AudioSegment] = []
        in_seg = False
        seg_start = silence_count = seg_index = 0

        for i, rms in enumerate(rms_vals):
            voiced = rms > threshold
            if voiced:
                if not in_seg:
                    in_seg    = True
                    seg_start = i
                silence_count = 0
            else:
                if in_seg:
                    silence_count += 1
                    if silence_count >= SILENCE_GAP_FRAMES:
                        end = i - silence_count + 1
                        if (end - seg_start) >= MIN_SEGMENT_FRAMES:
                            segments.append(AudioSegment(seg_index, seg_start, end))
                            seg_index += 1
                        in_seg        = False
                        silence_count = 0

        if in_seg:
            end = n_frames
            if (end - seg_start) >= MIN_SEGMENT_FRAMES:
                segments.append(AudioSegment(seg_index, seg_start, end))

        return segments

    # ── Step 2: 30-dim feature extraction ──────────────────────────────

    def _extract_segment_features(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.frame_size:
            return np.zeros(N_MFCC * 2 + 4)

        n_frames = len(audio) // self.frame_size
        mfcc_frames, f0s, centroids, zcrs = [], [], [], []

        for i in range(n_frames):
            frame = audio[i*self.frame_size:(i+1)*self.frame_size].copy()
            mag   = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            freqs = np.fft.rfftfreq(len(frame), d=1.0/self.sr)
            mfcc_frames.append(self._mfcc(mag, freqs))
            # Use 2048-sample window for F0 — far more reliable than 512
            # Stride at half-frame (256 samples) for better voiced-frame coverage
            f0_a = self._f0(audio[i*self.frame_size : i*self.frame_size + 2048])
            f0_b = self._f0(audio[i*self.frame_size + 256 : i*self.frame_size + 2048 + 256])
            # prefer non-zero; if both voiced take the one from better-resolved window
            if f0_a > 0 and f0_b > 0:
                f0s.append((f0_a + f0_b) / 2)
            else:
                f0s.append(f0_a if f0_a > 0 else f0_b)
            centroids.append(float(np.sum(freqs * mag**2) / (np.sum(mag**2) + 1e-10)))
            zcrs.append(float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2))

        mfcc_mat   = np.array(mfcc_frames)
        mfcc_mean  = mfcc_mat.mean(axis=0)
        delta_mean = np.diff(mfcc_mat, axis=0).mean(axis=0) if len(mfcc_mat) > 1 else np.zeros(N_MFCC)

        voiced_f0     = [f for f in f0s if f > 0]
        f0_feat       = float(np.median(voiced_f0)) if voiced_f0 else 0.0
        centroid_feat = float(np.mean(centroids))
        zcr_feat      = float(np.mean(zcrs))
        rolloff_feat  = self._rolloff(audio)

        flatness_feat = self._spectral_flatness(audio)
        return np.concatenate([mfcc_mean, delta_mean, [f0_feat, centroid_feat, zcr_feat, rolloff_feat, flatness_feat]])

    def _mfcc(self, mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        n_filters = 26
        mel_pts   = np.linspace(self._hz2mel(80), self._hz2mel(min(self.sr/2, 7600)), n_filters+2)
        hz_pts    = self._mel2hz(mel_pts)
        energies  = np.zeros(n_filters)
        for i in range(n_filters):
            lo, mid, hi = hz_pts[i], hz_pts[i+1], hz_pts[i+2]
            up   = (freqs >= lo)  & (freqs < mid)
            down = (freqs >= mid) & (freqs < hi)
            if up.any():
                energies[i] += np.sum(mag[up]**2   * (freqs[up]-lo)    / (mid-lo+1e-10))
            if down.any():
                energies[i] += np.sum(mag[down]**2 * (hi-freqs[down])  / (hi-mid+1e-10))
        log_e = np.log(energies + 1e-10)
        return np.array([np.sum(log_e * np.cos(np.pi*k*(np.arange(n_filters)+0.5)/n_filters))
                         for k in range(N_MFCC)])

    def _f0(self, frame: np.ndarray) -> float:
        """
        YIN-inspired F0 estimator using cumulative mean normalised difference
        function (CMNDF). More robust than pure autocorrelation on natural
        speech, correctly returns 0.0 for whisper (aperiodic/unvoiced).
        Uses up to 2048 samples for reliable resolution of low F0 (~80Hz).
        """
        work = frame[:2048] if len(frame) >= 2048 else frame
        N    = len(work)
        lo   = int(self.sr / 400)   # max F0 = 400 Hz
        hi   = min(int(self.sr / 60), N // 2)  # min F0 = 60 Hz
        if lo >= hi: return 0.0

        # Difference function via autocorrelation identity: d(t) = 2*(ac[0]-ac[t])
        ac   = np.correlate(work, work, mode="full")[N-1:]
        d    = 2.0 * (ac[0] - ac)

        # Cumulative mean normalisation
        cmndf = np.ones(len(d))
        running = 0.0
        for tau in range(1, hi + 5):
            running += d[tau]
            cmndf[tau] = d[tau] * tau / (running + 1e-10)

        # Find first dip below 0.20 (voiced speech threshold)
        for tau in range(lo, hi):
            if cmndf[tau] < 0.20:
                # Descend to local minimum
                while tau + 1 < hi and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                return float(self.sr / tau)

        # Fallback absolute minimum — only accept if reasonably periodic
        tau = int(np.argmin(cmndf[lo:hi])) + lo
        return float(self.sr / tau) if cmndf[tau] < 0.85 else 0.0

    def _rolloff(self, audio: np.ndarray) -> float:
        frame = audio[:self.frame_size]
        mag   = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), d=1.0/self.sr)
        ce    = np.cumsum(mag**2)
        idx   = np.searchsorted(ce, 0.85*(ce[-1]+1e-10))
        return float(freqs[min(idx, len(freqs)-1)])

    def _spectral_flatness(self, audio: np.ndarray) -> float:
        """
        Wiener entropy / spectral flatness: geometric_mean / arithmetic_mean of |FFT|.
        Near 0 = tonal/voiced speech. Near 1 = noise/whisper.
        Whisper typically > 0.12, normal speech < 0.12.
        """
        n = len(audio) // self.frame_size
        vals = []
        for i in range(n):
            frame = audio[i*self.frame_size:(i+1)*self.frame_size]
            mag   = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) + 1e-10
            gm    = np.exp(np.mean(np.log(mag)))
            am    = np.mean(mag)
            vals.append(gm / am)
        return float(np.mean(vals)) if vals else 0.1

    @staticmethod
    def _hz2mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    @staticmethod
    def _mel2hz(mel): return 700.0 * (10.0**(mel/2595.0) - 1.0)

    # ── Step 3: agglomerative clustering (cosine + F0) ──────────────────

    def _cluster_segments(self) -> None:
        """
        Complete-linkage agglomerative clustering with per-pair adaptive thresholds.

        Complete linkage: similarity(ClusterA, ClusterB) = MIN over all (i∈A, j∈B) pairs.
        This avoids centroid drift — when segment 3 is the bridge between cluster 2 and 4,
        merging (3,4) first can destroy the (2,3) merge by dragging the centroid away.

        Two thresholds:
          - Normal branch (MERGE_THRESHOLD=0.935): standard speech F0 range
          - Whisper branch (0.920): both segments >145Hz AND mfcc_sim>0.90
            Whisper raises F0 by 50-100Hz unpredictably, so we trust MFCCs more
            and use a lower merge bar.
        """
        n = len(self._segments)
        if n == 0: return
        if n == 1:
            self._segments[0].speaker_id = 0
            return

        feats     = np.array([s.features for s in self._segments], dtype=float)
        mfcc_part = feats[:, :N_MFCC].copy()
        f0s       = feats[:, -5].copy()
        mfcc_n    = mfcc_part / (np.linalg.norm(mfcc_part, axis=1, keepdims=True) + 1e-10)
        f0_range  = F0_RANGE_HZ

        cluster_id: list[int]              = list(range(n))
        c_members:  dict[int, list[int]]   = {i: [i] for i in range(n)}

        def pair_sim(mi: int, mj: int) -> tuple[float, float]:
            """
            Similarity + threshold for one original segment pair.

            Three regimes:
            1. Both breathy/whisper (flatness > 0.12):
               MFCC[0] tracks breathiness and drops drastically through a whisper
               continuum, making cosine unreliable. Instead use:
               - Flatness similarity (whisper tightness)
               - F0 direction (both rising = whisper, same person)
               This treats the entire whisper continuum as one speaker.

            2. One breathy + one voiced (normal):
               Different vocal modes → different speaker, SPLIT.
               Unless MFCC similarity is very high (>0.97).

            3. Both voiced:
               Standard 60% MFCC + 40% F0 blend.
               High-F0 voiced (>155Hz, strong MFCC match): reduce F0 weight.
            """
            mfcc_s  = float(np.dot(mfcc_n[mi], mfcc_n[mj]))
            fi, fj  = f0s[mi], f0s[mj]
            flat_i  = float(feats[mi, -1])
            flat_j  = float(feats[mj, -1])
            f0_s    = 1.0 - min(abs(fi-fj)/f0_range, 1.0) if fi>0 and fj>0 else mfcc_s

            # Both segments breathy = whisper regime (min flatness > 0.125)
            # MFCC[0] drops drastically with whisper depth — cosine unreliable.
            # Use 3-way blend + lower threshold 0.81 to bridge the continuum.
            if 0.125 < min(flat_i, flat_j) < 0.30:  # real whisper range; synthetic/pure tones have flat>0.35
                flat_s = 1.0 - min(abs(flat_i - flat_j) / 0.38, 1.0)
                return 0.50 * mfcc_s + 0.30 * f0_s + 0.20 * flat_s, 0.81

            # Both voiced, high F0: reduce F0 weight
            if fi > 155 and fj > 155 and mfcc_s > 0.92:
                return 0.82 * mfcc_s + 0.18 * f0_s, self.merge_threshold

            # Default: normal voiced speech
            return 0.60 * mfcc_s + 0.40 * f0_s, self.merge_threshold

        def cluster_sim(ci: int, cj: int) -> tuple[float, float]:
            """Single-linkage: MAXIMUM pairwise similarity across all member pairs.
            Allows whisper-transition segments to bridge clusters correctly.
            Two genuinely different speakers always score <0.89 on ALL pairs."""
            max_score, max_thr = -1.0, self.merge_threshold
            for mi in c_members[ci]:
                for mj in c_members[cj]:
                    s, thr = pair_sim(mi, mj)
                    if s > max_score:
                        max_score, max_thr = s, thr
            return max_score, max_thr

        while True:
            active = list(c_members.keys())
            if len(active) <= 1: break

            best_score, best_thresh = -1.0, self.merge_threshold
            best_ci, best_cj = active[0], active[1]

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    s, thr = cluster_sim(active[i], active[j])
                    if (s - thr) > (best_score - best_thresh):
                        best_score, best_thresh = s, thr
                        best_ci, best_cj = active[i], active[j]

            if best_score < best_thresh:
                break   # all remaining pairs are below threshold → done

            # Merge cj into ci
            c_members[best_ci].extend(c_members[best_cj])
            del c_members[best_cj]
            for k in range(n):
                if cluster_id[k] == best_cj:
                    cluster_id[k] = best_ci

        unique = sorted(set(cluster_id))
        remap  = {old: new for new, old in enumerate(unique)}
        for i, seg in enumerate(self._segments):
            seg.speaker_id = remap[cluster_id[i]]

    # ── Step 4: write per-speaker WAVs ──────────────────────────────────

    def _write_speaker_wavs(self, audio, out_dir, stem) -> list[str]:
        n_spk  = len(set(s.speaker_id for s in self._segments if s.speaker_id >= 0))
        paths  = []
        for spk_id in range(n_spk):
            out = np.zeros(len(audio), dtype=np.float32)
            total_frames = 0
            for seg in self._segments:
                if seg.speaker_id != spk_id: continue
                s = seg.start_sample
                e = min(seg.end_sample, len(audio))
                out[s:e] = audio[s:e]
                total_frames += seg.n_frames
            duration_s = total_frames * self.frame_size / self.sr
            path = os.path.join(out_dir, f"{stem}_speaker_{spk_id}.wav")
            _save_wav(path, out, self.sr)
            paths.append(path)
            spk_segs = [s for s in self._segments if s.speaker_id == spk_id]
            f0s = [s.features[-5] for s in spk_segs if len(s.features) > 4 and s.features[-5] > 0]
            f0_str = f"F0≈{np.median(f0s):.0f}Hz" if f0s else "F0=?"
            print(f"    Speaker {spk_id}  {f0_str}  {duration_s:.1f}s  → {stem}_speaker_{spk_id}.wav")
        return paths


def _save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2)
        wf.setframerate(sample_rate); wf.writeframes(pcm.tobytes())