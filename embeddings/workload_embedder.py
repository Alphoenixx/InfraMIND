"""
InfraMIND v3 — Workload Embedder (Contribution C1)
====================================================

Extracts a fixed 5-dimensional embedding vector z ∈ ℝ⁵ from a workload
trace window, capturing:

    z = [mean_rate, std_rate, burstiness, autocorrelation, peak_to_avg_ratio]

These features condition the surrogate GP model, enabling workload-aware
infrastructure optimization. The embedding captures:

  - Central tendency (mean_rate)
  - Spread / volatility (std_rate)
  - Overdispersion / spikiness (burstiness = σ²/μ, index of dispersion)
  - Temporal structure (lag-1 autocorrelation)
  - Extremal behavior (peak_to_avg_ratio)
"""

import numpy as np
from typing import List, Optional


class WorkloadEmbedder:
    """
    Extracts workload feature embeddings from traffic traces.

    The embedding is designed to be:
      - Low-dimensional (5D) for sample efficiency
      - Statistically meaningful for conditioning the optimizer
      - Fast to compute (no learned parameters)
    """

    EMBEDDING_DIM = 5
    FEATURE_NAMES = [
        "mean_rate",
        "std_rate",
        "burstiness",
        "autocorrelation",
        "peak_to_avg_ratio",
    ]

    def embed(self, trace_window: np.ndarray) -> np.ndarray:
        """
        Compute the 5D embedding from a single trace window.

        Parameters
        ----------
        trace_window : np.ndarray
            1D array of arrival rates, shape (window_size,).

        Returns
        -------
        z : np.ndarray
            Embedding vector of shape (5,).
        """
        if len(trace_window) < 3:
            raise ValueError("Trace window must have at least 3 samples")

        mean_rate = np.mean(trace_window)
        std_rate = np.std(trace_window)

        # Burstiness: index of dispersion (variance-to-mean ratio)
        # High values indicate overdispersed, bursty traffic
        burstiness = (std_rate ** 2) / max(mean_rate, 1e-8)

        # Lag-1 autocorrelation: measures temporal persistence
        autocorrelation = self._lag1_autocorrelation(trace_window)

        # Peak-to-average ratio: captures spike severity
        peak_to_avg = np.max(trace_window) / max(mean_rate, 1e-8)

        return np.array([
            mean_rate,
            std_rate,
            burstiness,
            autocorrelation,
            peak_to_avg,
        ], dtype=np.float64)

    def embed_batch(self, traces: np.ndarray) -> np.ndarray:
        """
        Embed multiple trace windows.

        Parameters
        ----------
        traces : np.ndarray
            2D array of shape (n_traces, window_size).

        Returns
        -------
        embeddings : np.ndarray
            Shape (n_traces, 5).
        """
        return np.array([self.embed(t) for t in traces])

    def compute_volatility(self, trace_window: np.ndarray) -> float:
        """
        Compute a scalar volatility measure used by Adaptive TuRBO
        to scale the trust region.

        This is the burstiness (index of dispersion): σ²/μ

        High volatility → tighter trust region (more conservative search)
        Low volatility  → wider trust region (more exploration)

        Parameters
        ----------
        trace_window : np.ndarray
            1D array of arrival rates.

        Returns
        -------
        volatility : float
            Non-negative scalar. 0 = perfectly steady.
        """
        z = self.embed(trace_window)
        return float(z[2])  # burstiness component

    def normalize_embedding(
        self,
        z: np.ndarray,
        z_mean: Optional[np.ndarray] = None,
        z_std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Normalize embedding for GP input (zero mean, unit variance).

        If z_mean/z_std not provided, normalizes by the embedding's own stats.
        """
        if z_mean is None:
            z_mean = z.mean(axis=0) if z.ndim == 2 else np.zeros(self.EMBEDDING_DIM)
        if z_std is None:
            z_std = z.std(axis=0) if z.ndim == 2 else np.ones(self.EMBEDDING_DIM)
        z_std = np.maximum(z_std, 1e-8)
        return (z - z_mean) / z_std

    @staticmethod
    def _lag1_autocorrelation(x: np.ndarray) -> float:
        """
        Compute lag-1 autocorrelation coefficient.

        Returns value in [-1, 1]:
          +1 = strong positive temporal correlation
           0 = uncorrelated
          -1 = strong negative temporal correlation (alternating)
        """
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        var = np.var(x)
        if var < 1e-12:
            return 0.0
        cov = np.mean((x[:-1] - mean) * (x[1:] - mean))
        return float(np.clip(cov / var, -1.0, 1.0))
