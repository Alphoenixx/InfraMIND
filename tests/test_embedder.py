"""
InfraMIND v3 — Tests for Workload Embedder
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from embeddings.workload_embedder import WorkloadEmbedder


class TestWorkloadEmbedder:

    def setup_method(self):
        self.embedder = WorkloadEmbedder()

    def test_embedding_shape(self):
        trace = np.ones(100) * 50.0
        z = self.embedder.embed(trace)
        assert z.shape == (5,), f"Expected shape (5,), got {z.shape}"

    def test_steady_workload_low_burstiness(self):
        trace = np.ones(100) * 100.0 + np.random.normal(0, 1, 100)
        z = self.embedder.embed(trace)
        # Burstiness should be low for near-constant signal
        assert z[2] < 1.0, f"Expected low burstiness, got {z[2]}"

    def test_bursty_workload_high_burstiness(self):
        trace = np.ones(100) * 50.0
        trace[40:50] = 500.0  # Big spike
        z = self.embedder.embed(trace)
        # Burstiness should be high
        assert z[2] > 5.0, f"Expected high burstiness, got {z[2]}"

    def test_autocorrelation_smooth_signal(self):
        t = np.linspace(0, 4 * np.pi, 200)
        trace = 100 + 30 * np.sin(t)
        z = self.embedder.embed(trace)
        # Smooth sine should have high positive autocorrelation
        assert z[3] > 0.5, f"Expected high autocorrelation, got {z[3]}"

    def test_autocorrelation_random_noise(self):
        trace = np.random.uniform(50, 150, 200)
        z = self.embedder.embed(trace)
        # Random noise should have near-zero autocorrelation
        assert abs(z[3]) < 0.3, f"Expected low autocorrelation, got {z[3]}"

    def test_peak_to_avg_constant(self):
        trace = np.ones(100) * 100.0
        z = self.embedder.embed(trace)
        assert abs(z[4] - 1.0) < 0.01, f"Expected PAR ≈ 1.0, got {z[4]}"

    def test_peak_to_avg_spiky(self):
        trace = np.ones(100) * 50.0
        trace[42] = 500.0
        z = self.embedder.embed(trace)
        assert z[4] > 5.0, f"Expected high PAR, got {z[4]}"

    def test_embed_batch(self):
        traces = np.random.uniform(50, 150, (10, 100))
        embeddings = self.embedder.embed_batch(traces)
        assert embeddings.shape == (10, 5)

    def test_volatility_computation(self):
        trace_steady = np.ones(100) * 100.0
        trace_bursty = np.ones(100) * 50.0
        trace_bursty[30:50] = 500.0

        v_steady = self.embedder.compute_volatility(trace_steady)
        v_bursty = self.embedder.compute_volatility(trace_bursty)

        assert v_bursty > v_steady, f"Bursty should be more volatile: {v_bursty} vs {v_steady}"

    def test_minimum_trace_length(self):
        with pytest.raises(ValueError):
            self.embedder.embed(np.array([1.0, 2.0]))

    def test_normalize_embedding(self):
        traces = np.random.uniform(50, 150, (20, 100))
        embeddings = self.embedder.embed_batch(traces)
        normalized = self.embedder.normalize_embedding(embeddings)
        # Should have approximately zero mean and unit std
        assert np.allclose(normalized.mean(axis=0), 0, atol=0.1)
        assert np.allclose(normalized.std(axis=0), 1, atol=0.2)
