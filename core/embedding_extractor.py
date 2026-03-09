"""
Speaker Embedding Extractor — v3
==================================
Two-stream design, based on what actually discriminates speakers
in the absence of a trained neural model:

  Stream 1 — F0 Histogram (primary, 80 bins, 60-400Hz)
    Captures fundamental frequency distribution over voiced frames.
    Same person → histogram peaks in same bins → cosine ~1.0
    Different person (different F0) → very different bins → cosine ~0.1-0.6

  Stream 2 — MFCC Variance Ratio (secondary)
    Per-coefficient variance of MFCC C2-C12 across all voiced frames.
    Similarity = mean(min/max) per coefficient.
    More content-independent than MFCC means; reflects speaking style / 
    vocal tract dynamics rather than phoneme inventory.

Why NOT use MFCC cosine similarity (v2 failure):
    The 12-band mel cosine similarity is 0.98-0.99 for ALL speakers
    in the same language because coarse spectral envelopes are language-
    dependent, not speaker-dependent. It contributed 0.83/0.84 of the total
    dot product while the discriminative F0 contributed only 0.013.
    Removing it as a primary feature was the key fix.

EMBED_DIM is exported for display only.
The actual .extract() method returns (f0_vec, var_vec) — two separate vectors.
"""

from __future__ import annotations
import numpy as np

SAMPLE_RATE    = 16000
HOP_SAMPLES    = 160        # 10ms
FRAME_SAMPLES  = 512        # 32ms

N_F0_BINS      = 80         # F0 histogram  60-400Hz  → 4.25Hz/bin
N_MFCC         = 20         # MFCC order (includes C0)
N_VAR_COEFFS   = 10         # variance of C2-C11
EMBED_DIM      = N_F0_BINS + N_VAR_COEFFS   # 90  (display only)

# Fused score weights  (must sum to 1)
W_F0  = 0.65
W_VAR = 0.35


class EmbeddingExtractor:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sr = sample_rate

    def extract(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (f0_vec, var_vec):
          f0_vec  — L2-normalised 80-bin F0 histogram
          var_vec — raw MFCC variance vector (C2-C11), NOT L2-normalised
                    (ratio similarity used in verifier, not cosine)
        """
        frames = self._process_frames(audio)
        if not frames:
            return np.zeros(N_F0_BINS, np.float32), np.ones(N_VAR_COEFFS, np.float32)

        f0s   = [f["f0"]   for f in frames if f["f0"] > 0]
        mfccs = np.array([f["mfcc"] for f in frames])   # (T, N_MFCC)

        return self._f0_stream(f0s), self._var_stream(mfccs)

    # ── Streams ─────────────────────────────────────────────────────────────

    @staticmethod
    def _f0_stream(f0s: list[float]) -> np.ndarray:
        if not f0s:
            return np.zeros(N_F0_BINS, np.float32)
        hist, _ = np.histogram(f0s, bins=N_F0_BINS, range=(60, 400))
        hist = hist.astype(np.float32)
        # Gaussian smoothing: sigma=2 bins (~8.5 Hz) tolerates natural F0 drift
        # between recording sessions without blurring across speakers (>50 Hz apart)
        sigma = 2.0
        r = int(3 * sigma)
        x = np.arange(-r, r + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        hist = np.convolve(hist, kernel, mode='same')
        n = np.linalg.norm(hist)
        return (hist / n).astype(np.float32) if n > 1e-8 else hist.astype(np.float32)

    @staticmethod
    def _var_stream(mfccs: np.ndarray) -> np.ndarray:
        """Per-coefficient variance of MFCC C2-C11."""
        return mfccs[:, 2:2+N_VAR_COEFFS].var(axis=0).astype(np.float32) + 1e-8

    # ── Frame processing ─────────────────────────────────────────────────────

    def _process_frames(self, audio: np.ndarray) -> list[dict]:
        n = (len(audio) - FRAME_SAMPLES) // HOP_SAMPLES
        if n < 2:
            return []
        rms = np.array([
            float(np.sqrt(np.mean(audio[i*HOP_SAMPLES:i*HOP_SAMPLES+FRAME_SAMPLES]**2)))
            for i in range(n)
        ])
        voiced_rms = rms[rms > 0.005]
        thresh = max(0.012, float(np.percentile(voiced_rms, 15))
                     if len(voiced_rms) >= 10 else 0.012)
        results = []
        for i in range(n):
            if rms[i] >= thresh:
                seg      = audio[i*HOP_SAMPLES : i*HOP_SAMPLES + FRAME_SAMPLES]
                ctx_start = i * HOP_SAMPLES
                ctx      = audio[ctx_start : ctx_start + 2048]
                results.append(self._frame(seg, ctx))
        return results

    def _frame(self, seg: np.ndarray, audio_ctx: np.ndarray) -> dict:
        """
        seg       — 512-sample hop window (for MFCC)
        audio_ctx — 2048-sample window aligned to same start (for F0)
        """
        sr, N = self.sr, len(seg)
        win  = seg * np.hanning(N)
        mag  = np.abs(np.fft.rfft(win, n=512)) + 1e-10
        mag2 = mag ** 2
        freqs = np.fft.rfftfreq(512, 1.0 / sr)

        f0 = self._f0_cmndf(audio_ctx, sr)

        mfcc = self._mel_mfcc(mag2, freqs)
        return dict(f0=f0, mfcc=mfcc)

    @staticmethod
    def _f0_cmndf(frame: np.ndarray, sr: int) -> float:
        """YIN-inspired CMNDF on up to 2048 samples. Returns 0.0 for unvoiced."""
        work = frame[:2048] if len(frame) >= 2048 else frame
        N    = len(work)
        lo   = int(sr / 400)
        hi   = min(int(sr / 60), N // 2)
        if lo >= hi:
            return 0.0
        ac = np.correlate(work, work, mode="full")[N - 1:]
        d  = 2.0 * (ac[0] - ac)
        cmndf = np.ones(len(d))
        running = 0.0
        for tau in range(1, hi + 5):
            running += d[tau]
            cmndf[tau] = d[tau] * tau / (running + 1e-10)
        for tau in range(lo, hi):
            if cmndf[tau] < 0.20:
                while tau + 1 < hi and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                return float(sr / tau)
        tau = int(np.argmin(cmndf[lo:hi])) + lo
        return float(sr / tau) if cmndf[tau] < 0.85 else 0.0

    def _mel_mfcc(self, mag2: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        n_mel = 40; sr = self.sr
        def hz2mel(h): return 2595*np.log10(1+h/700)
        def mel2hz(m): return 700*(10**(m/2595)-1)
        edges = mel2hz(np.linspace(hz2mel(80), hz2mel(sr/2), n_mel+2))
        nf = len(mag2)
        fb = np.zeros((n_mel, nf))
        for m in range(n_mel):
            lo, mid, hi = edges[m], edges[m+1], edges[m+2]
            up = (freqs[:nf]>=lo) & (freqs[:nf]<mid)
            dn = (freqs[:nf]>=mid) & (freqs[:nf]<hi)
            if up.any(): fb[m,up]=(freqs[:nf][up]-lo)/(mid-lo+1e-10)
            if dn.any(): fb[m,dn]=(hi-freqs[:nf][dn])/(hi-mid+1e-10)
        log_mel = np.log(fb @ mag2 + 1e-10)
        dct = np.cos(np.pi*np.outer(np.arange(N_MFCC), 2*np.arange(n_mel)+1)/(2*n_mel))
        return (log_mel @ dct.T).astype(np.float32)