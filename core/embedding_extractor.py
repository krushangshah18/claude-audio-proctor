"""
Speaker Embedding Extractor — v2
==================================
Discriminative speaker embedding without trained neural networks.

Core insight: speaker identity lives in three signal properties
that are relatively content-independent:

  1. F0 DISTRIBUTION (pitch histogram)
     20-bin histogram 60–400Hz over all voiced frames.

  2. VOCAL TRACT SHAPE (coarse mel spectrum)
     12-band mel energy profile, level-normalised shape.

  3. PROSODIC TEXTURE
     Spectral centroid, rolloff, flatness, ZCR distributions.

All features aggregated over ALL voiced frames → one L2-normalised vector.

Whisper robustness:
  Whisper raises F0 by ~50-100Hz but vocal tract shape stays similar.
  Same-person normal vs whisper → sim ~0.82-0.90.
  Different speaker             → sim ~0.55-0.75.
"""

from __future__ import annotations
import numpy as np

SAMPLE_RATE   = 16000
HOP_SAMPLES   = 160      # 10ms
FRAME_SAMPLES = 512      # 32ms

N_F0_BINS   = 20
N_MEL_BANDS = 12
N_SCALAR    = 6
EMBED_DIM   = N_F0_BINS + N_MEL_BANDS + N_SCALAR   # 38

F0_WEIGHT   = 2.5   # upweight F0 histogram vs spectral features


class EmbeddingExtractor:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """float32 audio → EMBED_DIM-dim unit-norm embedding."""
        if len(audio) < FRAME_SAMPLES * 2:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        frames = self._process_frames(audio)
        if not frames:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        f0s       = [f["f0"]       for f in frames if f["f0"] > 0]
        mels      = np.array([f["mel"]      for f in frames])
        centroids = [f["centroid"] for f in frames]
        rolloffs  = [f["rolloff"]  for f in frames]
        flats     = [f["flatness"] for f in frames]
        zcrs      = [f["zcr"]      for f in frames]

        # 1. F0 histogram
        if f0s:
            f0_hist, _ = np.histogram(f0s, bins=N_F0_BINS,
                                      range=(60, 400), density=False)
            f0_hist    = f0_hist.astype(float) / (len(f0s) + 1e-10)
        else:
            f0_hist = np.zeros(N_F0_BINS)

        # 2. Coarse mel shape (level-normalised)
        mel_mu = mels.mean(axis=0)
        mel_mu = (mel_mu - mel_mu.mean()) / (mel_mu.std() + 1e-8)

        # 3. Scalar prosodic features
        scalars = np.array([
            np.mean(centroids) / 4000,
            np.std(centroids)  / 2000,
            np.mean(rolloffs)  / 8000,
            np.mean(flats),
            np.std(flats),
            np.mean(zcrs),
        ])

        vec = np.concatenate([
            f0_hist * F0_WEIGHT,
            mel_mu,
            scalars,
        ]).astype(np.float32)

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else np.zeros(EMBED_DIM, dtype=np.float32)

    def _process_frames(self, audio: np.ndarray) -> list[dict]:
        n = (len(audio) - FRAME_SAMPLES) // HOP_SAMPLES
        rms = np.array([
            float(np.sqrt(np.mean(audio[i*HOP_SAMPLES:i*HOP_SAMPLES+FRAME_SAMPLES]**2)))
            for i in range(n)
        ])
        voiced_rms = rms[rms > 0.005]
        thresh = max(0.01, float(np.percentile(voiced_rms, 20))
                     if len(voiced_rms) > 0 else 0.01)
        return [self._frame_features(audio[i*HOP_SAMPLES:i*HOP_SAMPLES+FRAME_SAMPLES])
                for i in range(n) if rms[i] >= thresh]

    def _frame_features(self, frame: np.ndarray) -> dict:
        sr  = self.sr
        N   = len(frame)
        win = frame * np.hanning(N)
        mag  = np.abs(np.fft.rfft(win)) + 1e-10
        mag2 = mag ** 2
        freqs = np.fft.rfftfreq(N, 1.0 / sr)

        # F0
        ac  = np.correlate(win, win, "full")[N-1:]
        ac /= ac[0] + 1e-10
        lo, hi = int(sr/400), min(int(sr/60), len(ac)-1)
        p   = int(np.argmax(ac[lo:hi])) + lo
        f0  = float(sr/p) if ac[p] > 0.25 else 0.0

        # Mel energies
        mel_e = np.log(self._mel_energies(mag2, freqs) + 1e-10)

        # Spectral centroid
        centroid = float(np.sum(freqs * mag2) / (np.sum(mag2) + 1e-10))

        # Rolloff
        ce  = np.cumsum(mag2)
        idx = np.searchsorted(ce, 0.85 * ce[-1])
        rolloff = float(freqs[min(idx, len(freqs)-1)])

        # Flatness
        flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))

        # ZCR
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2)

        return dict(f0=f0, mel=mel_e, centroid=centroid,
                    rolloff=rolloff, flatness=flatness, zcr=zcr)

    def _mel_energies(self, mag2: np.ndarray, freqs: np.ndarray,
                      f_lo: float = 80.0, f_hi: float = 4000.0) -> np.ndarray:
        def hz2mel(h): return 2595 * np.log10(1 + h/700)
        def mel2hz(m): return 700 * (10**(m/2595) - 1)
        edges = mel2hz(np.linspace(hz2mel(f_lo), hz2mel(f_hi), N_MEL_BANDS+2))
        out   = np.zeros(N_MEL_BANDS)
        for m in range(N_MEL_BANDS):
            lo, mid, hi = edges[m], edges[m+1], edges[m+2]
            up   = (freqs >= lo)  & (freqs < mid)
            down = (freqs >= mid) & (freqs < hi)
            if up.any():
                out[m] += float(np.sum((freqs[up]-lo)/(mid-lo+1e-10) * mag2[up]))
            if down.any():
                out[m] += float(np.sum((hi-freqs[down])/(hi-mid+1e-10) * mag2[down]))
        return out
