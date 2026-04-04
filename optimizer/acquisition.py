"""
InfraMIND v3 — Trust Region Acquisition Function
==================================================

Wraps acquisition function evaluation to respect the adaptive
trust region bounds. Supports both BoTorch and scipy-based backends.

The acquisition function ranks candidate points within the trust
region by Expected Improvement (EI):

    EI(x) = E[max(f_best - f(x), 0)]
           = (f_best - μ(x)) Φ(Z) + σ(x) φ(Z)

    where Z = (f_best - μ(x)) / σ(x)

This is the standard analytical EI, computed using the GP posterior.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from scipy.stats import norm

logger = logging.getLogger("inframind.optimizer")


class TrustRegionAcquisition:
    """
    Expected Improvement acquisition within an adaptive trust region.

    Generates candidates via Sobol sampling within the trust region,
    evaluates EI for each, and returns the top-k candidates.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def optimize(
        self,
        model,
        best_f: float,
        lb: np.ndarray,
        ub: np.ndarray,
        n_candidates: int = 5000,
        batch_size: int = 4,
        workload_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Find the best candidates within the trust region.

        Parameters
        ----------
        model : WorkloadConditionedGP
            Fitted surrogate model.
        best_f : float
            Best objective value observed so far.
        lb, ub : np.ndarray
            Trust region bounds in [0,1]^d (from AdaptiveTuRBO).
        n_candidates : int
            Number of Sobol candidates to evaluate.
        batch_size : int
            Number of top candidates to return.
        workload_embedding : np.ndarray, optional
            If provided, appended to each candidate for GP evaluation.

        Returns
        -------
        best_candidates : np.ndarray, shape (batch_size, theta_dim)
            Top candidates by EI value (in θ space, without z).
        """
        theta_dim = len(lb)

        # Generate Sobol candidates in trust region
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=theta_dim, scramble=True, seed=self.rng.randint(0, 2**31))
            raw = sampler.random(n_candidates)
        except ImportError:
            raw = self.rng.uniform(0, 1, size=(n_candidates, theta_dim))

        # Scale to trust region
        candidates_theta = lb + raw * (ub - lb)

        # Build GP inputs: [θ ∥ z]
        if workload_embedding is not None:
            z_tiled = np.tile(workload_embedding, (n_candidates, 1))
            candidates_gp = np.hstack([candidates_theta, z_tiled])
        else:
            candidates_gp = candidates_theta

        # Evaluate GP posterior
        mean, std = model.predict(candidates_gp)

        # Compute Expected Improvement
        ei = self._expected_improvement(mean, std, best_f)

        # Select top-k by EI
        top_indices = np.argsort(-ei)[:batch_size]

        logger.debug(
            f"Acquisition: top EI = {ei[top_indices[0]]:.6f}, "
            f"mean(EI) = {np.mean(ei):.6f}, "
            f"TR vol = {np.prod(ub - lb):.6f}"
        )

        return candidates_theta[top_indices]

    @staticmethod
    def _expected_improvement(
        mean: np.ndarray,
        std: np.ndarray,
        best_f: float,
    ) -> np.ndarray:
        """
        Compute analytical Expected Improvement.

        EI(x) = (f_best - μ(x)) Φ(Z) + σ(x) φ(Z)
        where Z = (f_best - μ(x)) / σ(x)

        We are MINIMIZING, so improvement = best_f - mean (lower is better).
        """
        improvement = best_f - mean
        Z = np.zeros_like(mean)
        mask = std > 1e-8
        Z[mask] = improvement[mask] / std[mask]

        ei = np.zeros_like(mean)
        ei[mask] = improvement[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])
        ei[~mask] = np.maximum(improvement[~mask], 0.0)

        return ei

    def compute_ei_surface(
        self,
        model,
        best_f: float,
        lb: np.ndarray,
        ub: np.ndarray,
        resolution: int = 50,
        fixed_dims: Optional[dict] = None,
        workload_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute EI over a 2D grid for visualization.

        Parameters
        ----------
        model : fitted GP model
        best_f : best observed value
        lb, ub : bounds
        resolution : grid points per dimension
        fixed_dims : dict mapping dim_index → fixed_value (for slicing)
        workload_embedding : optional z vector

        Returns
        -------
        X1, X2 : np.ndarray — meshgrid coordinates
        EI : np.ndarray — EI values on grid
        """
        dim = len(lb)
        if fixed_dims is None:
            fixed_dims = {}

        # Find two free dimensions
        free_dims = [d for d in range(dim) if d not in fixed_dims]
        if len(free_dims) < 2:
            raise ValueError("Need at least 2 free dimensions for surface")

        d1, d2 = free_dims[0], free_dims[1]

        x1 = np.linspace(lb[d1], ub[d1], resolution)
        x2 = np.linspace(lb[d2], ub[d2], resolution)
        X1, X2 = np.meshgrid(x1, x2)

        # Build query points
        grid_points = np.zeros((resolution * resolution, dim))
        for d in range(dim):
            if d == d1:
                grid_points[:, d] = X1.ravel()
            elif d == d2:
                grid_points[:, d] = X2.ravel()
            elif d in fixed_dims:
                grid_points[:, d] = fixed_dims[d]
            else:
                grid_points[:, d] = (lb[d] + ub[d]) / 2

        # Add workload embedding
        if workload_embedding is not None:
            z_tiled = np.tile(workload_embedding, (grid_points.shape[0], 1))
            grid_gp = np.hstack([grid_points, z_tiled])
        else:
            grid_gp = grid_points

        mean, std = model.predict(grid_gp)
        ei = self._expected_improvement(mean, std, best_f)

        return X1, X2, ei.reshape(resolution, resolution)
