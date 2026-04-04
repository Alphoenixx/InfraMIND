"""
InfraMIND v3 — Tests for Stability Metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from metrics.stability_metrics import StabilityMetrics, ObjectiveValue
from simulator.engine import SimulationResult


class TestStabilityMetrics:

    def setup_method(self):
        self.metrics = StabilityMetrics(
            sla_target_ms=200.0,
            lambda_sla=10.0,
            lambda_variance=2.0,
        )

    def _make_result(self, latencies, cost=1.0, dropped=0, total=100):
        return SimulationResult(
            latencies=np.array(latencies),
            total_cost=cost,
            dropped_requests=dropped,
            completed_requests=total - dropped,
            total_requests=total,
        )

    def test_percentiles(self):
        latencies = list(range(1, 101))  # 1 to 100
        result = self._make_result(latencies)
        obj = self.metrics.compute(result)

        assert abs(obj.p50 - 50.5) < 1.0
        assert abs(obj.p99 - 99.01) < 1.0

    def test_sla_violations(self):
        # All latencies above SLA (200ms)
        latencies = np.full(100, 250.0)
        result = self._make_result(latencies)
        obj = self.metrics.compute(result)

        assert obj.sla_violation_rate > 0.9
        assert obj.sla_penalty > 0

    def test_no_sla_violations(self):
        latencies = np.full(100, 100.0)
        result = self._make_result(latencies)
        obj = self.metrics.compute(result)

        assert obj.sla_violation_rate < 0.01
        assert obj.is_feasible

    def test_variance_affects_objective(self):
        # Same mean but different variance
        latencies_stable = np.full(100, 100.0)
        latencies_volatile = np.concatenate([np.full(50, 50.0), np.full(50, 150.0)])

        obj_stable = self.metrics.compute(self._make_result(latencies_stable))
        obj_volatile = self.metrics.compute(self._make_result(latencies_volatile))

        # Volatile should have higher objective due to variance penalty
        assert obj_volatile.objective > obj_stable.objective

    def test_stability_score(self):
        # Perfect stability
        score_perfect = StabilityMetrics.compute_stability_score(np.full(100, 100.0))
        assert abs(score_perfect - 1.0) < 0.01

        # Low stability — extreme bimodal distribution
        score_volatile = StabilityMetrics.compute_stability_score(
            np.concatenate([np.full(50, 1.0), np.full(50, 1000.0)])
        )
        assert score_volatile < 0.6

    def test_ablated_no_variance(self):
        latencies = np.concatenate([np.full(50, 50.0), np.full(50, 150.0)])
        result = self._make_result(latencies)

        obj_full = self.metrics.compute(result)
        obj_no_var = self.metrics.compute_ablated(result, disable_variance=True)

        assert obj_no_var.objective < obj_full.objective

    def test_degenerate_result(self):
        result = self._make_result([], cost=0)
        obj = self.metrics.compute(result)
        assert obj.objective == 1e6  # Large penalty

    def test_to_dict(self):
        result = self._make_result([100.0, 110.0, 120.0])
        obj = self.metrics.compute(result)
        d = obj.to_dict()
        assert "objective" in d
        assert "p99" in d
        assert "is_feasible" in d
