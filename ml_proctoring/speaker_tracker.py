"""
ml_proctoring.speaker_tracker
==============================
Online cosine-clustering of d-vector embeddings.
Assigns speaker IDs in real time and counts distinct speakers.
No batch processing — one embedding at a time.
"""
from __future__ import annotations

import numpy as np

_MERGE_THRESHOLD = 0.75   # cosine sim above which two embeddings = same speaker


class OnlineSpeakerTracker:
    """
    Running k-means-style online speaker tracker.

    Each update() call assigns the incoming embedding to the nearest
    existing cluster centroid (if close enough) or creates a new cluster.
    Centroids are updated with a running mean after each assignment.
    """

    def __init__(self, merge_threshold: float = _MERGE_THRESHOLD):
        self._threshold  = merge_threshold
        self._centroids: list[np.ndarray] = []   # L2-normalised
        self._counts:    list[int]         = []   # frames assigned to each cluster

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, embedding: np.ndarray) -> int:
        """
        Assign embedding to a speaker cluster.

        Returns the assigned speaker_id (0-indexed integer).
        New cluster IDs are allocated incrementally.
        """
        emb = self._normalise(embedding)

        if not self._centroids:
            self._centroids.append(emb)
            self._counts.append(1)
            return 0

        sims    = [float(np.dot(emb, c)) for c in self._centroids]
        best    = int(np.argmax(sims))
        best_s  = sims[best]

        if best_s >= self._threshold:
            # Merge into existing cluster — update centroid with running mean
            n = self._counts[best]
            new_c = (self._centroids[best] * n + emb) / (n + 1)
            self._centroids[best] = self._normalise(new_c)
            self._counts[best]   += 1
            return best
        else:
            # New speaker
            self._centroids.append(emb)
            self._counts.append(1)
            return len(self._centroids) - 1

    # ── Query ─────────────────────────────────────────────────────────────────

    @property
    def n_speakers(self) -> int:
        """Number of distinct speaker clusters seen so far."""
        return len(self._centroids)

    def verify_score(self, enrolled_centroid: np.ndarray) -> float:
        """
        Cosine similarity between the enrolled speaker centroid and the
        dominant cluster centroid (the one with the most assigned frames).

        Returns 0.0 if no clusters exist yet.
        """
        if not self._centroids:
            return 0.0
        dominant = int(np.argmax(self._counts))
        enroll_n = self._normalise(enrolled_centroid)
        return float(np.dot(enroll_n, self._centroids[dominant]))

    def current_embedding(self) -> np.ndarray | None:
        """Most recently updated cluster centroid (dominant speaker)."""
        if not self._centroids:
            return None
        dominant = int(np.argmax(self._counts))
        return self._centroids[dominant].copy()

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._centroids.clear()
        self._counts.clear()

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)
