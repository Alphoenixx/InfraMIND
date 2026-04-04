"""
InfraMIND v3 — Tests for Optimizer Components
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from optimizer.adaptive_turbo import AdaptiveTuRBO


class TestAdaptiveTuRBO:

    def setup_method(self):
        self.turbo = AdaptiveTuRBO(
            dim=10,
            length_init=0.8,
            length_min=0.005,
            length_max=1.6,
            success_tolerance=3,
            failure_tolerance=5,
            volatility_alpha=1.5,
        )

    def test_trust_region_shrinks_with_volatility(self):
        center = np.full(10, 0.5)

        # Low volatility → large TR
        lb_low, ub_low = self.turbo.get_trust_region(center, workload_volatility=0.1)
        tr_size_low = np.mean(ub_low - lb_low)

        # High volatility → small TR
        lb_high, ub_high = self.turbo.get_trust_region(center, workload_volatility=5.0)
        tr_size_high = np.mean(ub_high - lb_high)

        assert tr_size_high < tr_size_low, \
            f"Higher volatility should shrink TR: {tr_size_high} vs {tr_size_low}"

    def test_trust_region_zero_volatility(self):
        center = np.full(10, 0.5)
        lb, ub = self.turbo.get_trust_region(center, workload_volatility=0.0)
        # With zero volatility, adapted length should equal base length
        adapted_range = np.mean(ub - lb)
        assert adapted_range > 0.5, "Zero volatility should give large TR"

    def test_success_expands_tr(self):
        x = np.full(10, 0.5)
        initial_length = self.turbo.length

        # Simulate consecutive improvements
        for i in range(self.turbo.success_tolerance):
            self.turbo.update_state(10.0 - i * 1.0, x)

        assert self.turbo.length > initial_length, "Consecutive successes should expand TR"

    def test_failure_shrinks_tr(self):
        x = np.full(10, 0.5)
        self.turbo.best_value = 5.0
        initial_length = self.turbo.length

        # Simulate consecutive failures
        for i in range(self.turbo.failure_tolerance):
            self.turbo.update_state(5.0 + i * 0.1, x)  # Worse values

        assert self.turbo.length < initial_length, "Consecutive failures should shrink TR"

    def test_restart_on_min_length(self):
        x = np.full(10, 0.5)
        self.turbo.length = 0.004  # Below minimum
        self.turbo.best_value = 5.0
        self.turbo.failure_counter = 0

        # Trigger enough failures to shrink below min, causing restart
        for i in range(self.turbo.failure_tolerance):
            self.turbo.update_state(6.0, x)

        assert self.turbo.n_restarts >= 1, "Should restart when TR too small"
        # After restart, length resets to length_init
        assert self.turbo.length >= self.turbo.length_min, "After restart, length should be above minimum"

    def test_suggest_candidates_shape(self):
        center = np.full(10, 0.5)
        candidates = self.turbo.suggest_candidates(
            center=center,
            workload_volatility=1.0,
            n_candidates=100,
            batch_size=4,
        )
        assert candidates.shape == (4, 10)

    def test_candidates_within_trust_region(self):
        center = np.full(10, 0.5)
        lb, ub = self.turbo.get_trust_region(center, workload_volatility=1.0)
        candidates = self.turbo.suggest_candidates(
            center=center,
            workload_volatility=1.0,
            n_candidates=100,
            batch_size=10,
        )
        assert np.all(candidates >= lb - 1e-6), "Candidates should be within TR lower bound"
        assert np.all(candidates <= ub + 1e-6), "Candidates should be within TR upper bound"

    def test_state_history(self):
        center = np.full(10, 0.5)
        self.turbo.get_trust_region(center, 1.0)
        self.turbo.get_trust_region(center, 2.0)
        trajectory = self.turbo.get_state_trajectory()
        assert len(trajectory) == 2
        assert trajectory[0]['volatility'] == 1.0
        assert trajectory[1]['volatility'] == 2.0
