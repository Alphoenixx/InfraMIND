"""
InfraMIND v3 — Workload-Conditioned GP Surrogate (Contribution C1)
===================================================================

Implements the workload-conditioned Gaussian Process surrogate model:

    f(θ, z) → objective_value

where:
    θ ∈ ℝ^d = infrastructure parameters (from param mapper)
    z ∈ ℝ^5 = workload embedding (from workload embedder)

The GP is trained on (x, y) pairs where x = [θ ∥ z] and y = objective.

Key design decisions:
  - Uses Matérn-5/2 kernel with automatic relevance determination (ARD)
  - ARD learns separate lengthscales for θ dims and z dims
  - This allows the model to discover which workload features matter most
  - Fitted via marginal likelihood maximization (Type-II ML)

Complexity:
  - GP fitting: O(n³) where n = number of observations
  - GP prediction: O(n²) per query point
  - Practical limit: ~1000-2000 observations before switching to sparse GP

This module wraps BoTorch's SingleTaskGP for maximum research credibility
and falls back to a simpler sklearn-based GP if BoTorch is unavailable.
"""

import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger("inframind.optimizer")

# Try BoTorch (preferred), fall back to sklearn
_USE_BOTORCH = False
try:
    import torch
    from torch import Tensor
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.means import ConstantMean
    _USE_BOTORCH = True
    logger.info("Using BoTorch GP backend")
except ImportError:
    logger.info("BoTorch not available, using sklearn GP fallback")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class WorkloadConditionedGP:
    """
    Workload-conditioned Gaussian Process surrogate.

    Learns the mapping:
        f: [θ_reduced ∥ z_workload] → objective_value

    The ARD kernel automatically learns which dimensions
    (infrastructure params vs workload features) are most relevant.
    """

    def __init__(
        self,
        theta_dim: int,
        z_dim: int = 5,
        noise_variance: float = 1e-4,
    ):
        """
        Parameters
        ----------
        theta_dim : int
            Dimensionality of infrastructure parameter vector.
        z_dim : int
            Dimensionality of workload embedding (default: 5).
        noise_variance : float
            Observation noise variance.
        """
        self.theta_dim = theta_dim
        self.z_dim = z_dim
        self.input_dim = theta_dim + z_dim
        self.noise_variance = noise_variance
        self.model = None
        self._X_train = None
        self._Y_train = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit the GP to observed data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
            Input matrix: each row is [θ ∥ z].
        Y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            Objective values.
        """
        assert X.shape[1] == self.input_dim, (
            f"Expected input dim {self.input_dim}, got {X.shape[1]}"
        )

        self._X_train = X.copy()
        self._Y_train = Y.copy().reshape(-1)

        if _USE_BOTORCH:
            self._fit_botorch(X, Y)
        else:
            self._fit_sklearn(X, Y)

        self._is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict posterior mean and standard deviation.

        Parameters
        ----------
        X : np.ndarray, shape (n_queries, input_dim)

        Returns
        -------
        mean : np.ndarray, shape (n_queries,)
        std : np.ndarray, shape (n_queries,)
        """
        if not self._is_fitted:
            raise RuntimeError("GP model not fitted yet")

        if _USE_BOTORCH:
            return self._predict_botorch(X)
        else:
            return self._predict_sklearn(X)

    def get_best_observed(self) -> Tuple[np.ndarray, float]:
        """Return the best observed input and value."""
        if self._X_train is None:
            raise RuntimeError("No training data")
        best_idx = np.argmin(self._Y_train)
        return self._X_train[best_idx].copy(), float(self._Y_train[best_idx])

    # ── BoTorch Backend ─────────────────────────────────────────

    def _fit_botorch(self, X: np.ndarray, Y: np.ndarray):
        """Fit using BoTorch SingleTaskGP with ARD Matérn kernel."""
        X_torch = torch.tensor(X, dtype=torch.float64)
        Y_torch = torch.tensor(Y.reshape(-1, 1), dtype=torch.float64)

        # Standardize Y for better GP fitting
        self._y_mean = Y_torch.mean()
        self._y_std = Y_torch.std().clamp(min=1e-6)
        Y_standardized = (Y_torch - self._y_mean) / self._y_std

        self.model = SingleTaskGP(X_torch, Y_standardized)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            logger.warning(f"GP fitting warning: {e}")

    def _predict_botorch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using BoTorch model."""
        X_torch = torch.tensor(X, dtype=torch.float64)

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_torch)
            mean_standardized = posterior.mean.squeeze(-1)
            var_standardized = posterior.variance.squeeze(-1)

        # Un-standardize
        mean = (mean_standardized * self._y_std + self._y_mean).numpy()
        std = (var_standardized.sqrt() * self._y_std).numpy()

        return mean, std

    def get_botorch_model(self):
        """Return the underlying BoTorch model (for acquisition functions)."""
        if not _USE_BOTORCH:
            raise RuntimeError("BoTorch not available")
        return self.model

    # ── Sklearn Fallback ────────────────────────────────────────

    def _fit_sklearn(self, X: np.ndarray, Y: np.ndarray):
        """Fit using sklearn GaussianProcessRegressor."""
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(self.input_dim),
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.noise_variance,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        self.model.fit(X, Y.reshape(-1))

    def _predict_sklearn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using sklearn model."""
        mean, std = self.model.predict(X, return_std=True)
        return mean, std
