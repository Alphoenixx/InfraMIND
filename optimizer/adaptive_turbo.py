"""
InfraMIND v3 — Adaptive TuRBO Optimizer (Contribution C3)
==========================================================

NOVEL CONTRIBUTION: Workload-Sensitive Trust Region Adaptation.

Standard TuRBO maintains a trust region that expands on consecutive
improvements and shrinks on consecutive failures. Our extension adds
a workload-conditioned scaling factor:

    L_adapted = L_base × (1 / (1 + α · burstiness))

This creates fundamentally different optimization behavior:

    Stable workloads (burstiness ≈ 0):
        → volatility_factor ≈ 1.0
        → Full trust region
        → Aggressive exploration

    Bursty workloads (burstiness >> 1):
        → volatility_factor << 1.0
        → Tighter trust region
        → Conservative, safe search

WHY THIS MATTERS:
    Under volatile workloads, the objective landscape becomes noisy.
    Standard TuRBO may explore too aggressively, evaluating
    configurations in regions where the GP has high uncertainty
    AND the simulation itself is noisy. This leads to:
      - Wasted evaluations
      - False improvements that don't generalize
      - Oscillation between "good" and "bad" regions

    By tightening the trust region, we:
      - Stay closer to known-good regions
      - Reduce evaluation noise impact
      - Focus search on reliable local improvements

Complexity:
    - Trust region computation: O(d) per iteration
    - Candidate generation: O(C × d) for C Sobol candidates
    - Total per iteration: O(C × n² + n³) (dominated by GP predict/fit)
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger("inframind.optimizer")


@dataclass
class TuRBOState:
    """Tracks the internal state of the TuRBO optimizer for logging/analysis."""
    iteration: int = 0
    length: float = 0.8
    adapted_length: float = 0.8
    volatility: float = 0.0
    volatility_factor: float = 1.0
    success_counter: int = 0
    failure_counter: int = 0
    best_value: float = float("inf")
    center: Optional[np.ndarray] = None
    n_restarts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "length": self.length,
            "adapted_length": self.adapted_length,
            "volatility": self.volatility,
            "volatility_factor": self.volatility_factor,
            "success_counter": self.success_counter,
            "failure_counter": self.failure_counter,
            "best_value": self.best_value,
            "n_restarts": self.n_restarts,
        }


class AdaptiveTuRBO:
    """
    Trust Region Bayesian Optimization with workload-sensitive adaptation.

    Extends TuRBO (Eriksson et al., NeurIPS 2019) with:
    1. Workload volatility-scaled trust regions
    2. History-aware restart with best-known center retention

    Parameters
    ----------
    dim : int
        Dimensionality of the search space (effective dim from param mapper).
    length_init : float
        Initial trust region side length (in unit hypercube).
    length_min : float
        Minimum length triggering restart.
    length_max : float
        Maximum trust region length.
    success_tolerance : int
        Consecutive improvements before expanding TR.
    failure_tolerance : int
        Consecutive non-improvements before shrinking TR.
    volatility_alpha : float
        Sensitivity of TR to workload burstiness.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        dim: int,
        length_init: float = 0.8,
        length_min: float = 0.005,
        length_max: float = 1.6,
        success_tolerance: int = 3,
        failure_tolerance: int = 5,
        volatility_alpha: float = 1.5,
        seed: int = 42,
    ):
        self.dim = dim
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tolerance = success_tolerance
        self.failure_tolerance = failure_tolerance
        self.volatility_alpha = volatility_alpha
        self.rng = np.random.RandomState(seed)

        # State
        self.length = length_init
        self.success_counter = 0
        self.failure_counter = 0
        self.best_value = float("inf")
        self.best_x = None
        self.n_restarts = 0

        # History for analysis
        self.state_history: List[TuRBOState] = []
        self._iteration = 0

    def get_trust_region(
        self,
        center: np.ndarray,
        workload_volatility: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the adapted trust region bounds.

        Parameters
        ----------
        center : np.ndarray, shape (dim,)
            Center of the trust region (best observed x, in [0,1]^d).
        workload_volatility : float
            Burstiness score from workload embedder (σ²/μ).

        Returns
        -------
        lb : np.ndarray — lower bounds in [0,1]^d
        ub : np.ndarray — upper bounds in [0,1]^d
        """
        # Workload-sensitive scaling
        volatility_factor = 1.0 / (1.0 + self.volatility_alpha * workload_volatility)
        adapted_length = self.length * volatility_factor

        # Clamp
        adapted_length = np.clip(adapted_length, self.length_min, self.length_max)

        # Build trust region as hypercube around center
        lb = np.clip(center - adapted_length / 2.0, 0.0, 1.0)
        ub = np.clip(center + adapted_length / 2.0, 0.0, 1.0)

        # Log state
        state = TuRBOState(
            iteration=self._iteration,
            length=self.length,
            adapted_length=adapted_length,
            volatility=workload_volatility,
            volatility_factor=volatility_factor,
            success_counter=self.success_counter,
            failure_counter=self.failure_counter,
            best_value=self.best_value,
            center=center.copy(),
            n_restarts=self.n_restarts,
        )
        self.state_history.append(state)

        return lb, ub

    def suggest_candidates(
        self,
        center: np.ndarray,
        workload_volatility: float,
        n_candidates: int = 5000,
        batch_size: int = 4,
        model=None,
    ) -> np.ndarray:
        """
        Generate candidate points within the trust region.

        Uses Sobol quasi-random sampling for space-filling within TR,
        then optionally ranks by acquisition function if model is provided.

        Parameters
        ----------
        center : np.ndarray
            Current best point.
        workload_volatility : float
            Workload burstiness score.
        n_candidates : int
            Number of Sobol candidates to generate.
        batch_size : int
            Number of points to return.
        model : optional
            GP surrogate for acquisition-based ranking.

        Returns
        -------
        candidates : np.ndarray, shape (batch_size, dim)
        """
        lb, ub = self.get_trust_region(center, workload_volatility)

        # Sobol quasi-random sampling within TR
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=self.dim, scramble=True, seed=self.rng.randint(0, 2**31))
            raw_samples = sampler.random(n_candidates)
        except ImportError:
            # Fallback to uniform random
            raw_samples = self.rng.uniform(0, 1, size=(n_candidates, self.dim))

        # Scale to trust region
        candidates = lb + raw_samples * (ub - lb)

        # Rank by acquisition function if model available
        if model is not None and model.is_fitted:
            candidates = self._rank_by_acquisition(candidates, model, center, workload_volatility)

        return candidates[:batch_size]

    def _rank_by_acquisition(
        self,
        candidates: np.ndarray,
        model,
        center: np.ndarray,
        workload_volatility: float,
    ) -> np.ndarray:
        """Rank candidates by Expected Improvement."""
        from embeddings.workload_embedder import WorkloadEmbedder

        # For acquisition, we need [θ ∥ z] inputs
        # Use the current workload embedding for all candidates
        # (z is constant within a single iteration)
        mean, std = model.predict(candidates)

        # Expected Improvement
        best_f = self.best_value
        with np.errstate(divide="warn"):
            improvement = best_f - mean
            Z = improvement / np.maximum(std, 1e-8)

        # EI = improvement × Φ(Z) + std × φ(Z)
        from scipy.stats import norm
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-8] = 0.0

        # Sort by EI descending
        indices = np.argsort(-ei)
        return candidates[indices]

    def update_state(self, new_value: float, new_x: np.ndarray):
        """
        Update trust region state after evaluating a new point.

        Parameters
        ----------
        new_value : float
            Objective value of the new evaluation.
        new_x : np.ndarray
            The input that produced this value.
        """
        self._iteration += 1

        # Check for improvement (with tolerance)
        if self.best_value == float("inf"):
            is_improvement = True  # Any finite value improves on inf
        else:
            threshold = self.best_value - 1e-3 * abs(self.best_value)
            is_improvement = new_value < threshold
        if is_improvement:
            self.success_counter += 1
            self.failure_counter = 0
            logger.debug(
                f"  TuRBO: improvement {self.best_value:.4f} → {new_value:.4f} "
                f"(success {self.success_counter}/{self.success_tolerance})"
            )
        else:
            self.failure_counter += 1
            self.success_counter = 0
            logger.debug(
                f"  TuRBO: no improvement (failure {self.failure_counter}/{self.failure_tolerance})"
            )

        # Expand on consecutive successes
        if self.success_counter >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
            logger.info(f"  TuRBO: TR expanded → length={self.length:.4f}")

        # Shrink on consecutive failures
        elif self.failure_counter >= self.failure_tolerance:
            self.length = self.length / 2.0
            self.failure_counter = 0
            logger.info(f"  TuRBO: TR shrunk → length={self.length:.4f}")

        # Update best
        if new_value < self.best_value:
            self.best_value = new_value
            self.best_x = new_x.copy()

        # Check for restart
        if self.length < self.length_min:
            self._restart()

    def _restart(self):
        """Restart TuRBO with fresh trust region (but keep best point)."""
        self.n_restarts += 1
        self.length = self.length_init
        self.success_counter = 0
        self.failure_counter = 0
        logger.info(
            f"  TuRBO: RESTART #{self.n_restarts} "
            f"(best_value={self.best_value:.4f})"
        )

    def get_center(self) -> np.ndarray:
        """Get current best point as TR center."""
        if self.best_x is not None:
            return self.best_x.copy()
        # Random center if no observations yet
        return self.rng.uniform(0, 1, size=self.dim)

    def get_state_trajectory(self) -> List[Dict]:
        """Return full state history for visualization."""
        return [s.to_dict() for s in self.state_history]

    @property
    def is_converged(self) -> bool:
        """Check if optimizer has effectively converged."""
        return self.length < self.length_min and self.n_restarts >= 3
