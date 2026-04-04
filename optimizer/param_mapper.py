"""
InfraMIND v3 — Hierarchical Parameter Mapper
==============================================

Maps a low-dimensional optimization vector θ_reduced to the full
per-service configuration. Services within the same cluster
share parameter values.

This is the mechanism by which structure learning (C2) reduces
the effective dimensionality of the optimization problem:

    Full space:     n_services × n_per_service_params + n_global_params
    Reduced space:  n_clusters × n_per_service_params + n_global_params

Example:
    7 services × 4 params + 1 global = 29 dimensions (full)
    3 clusters × 4 params + 1 global = 13 dimensions (reduced)

    → 55% dimensionality reduction

The mapper operates on the unit hypercube [0,1]^d and maps to
physical parameter values using the per-param bounds.
"""

import numpy as np
import logging
from typing import Dict, List, Set, Any, Tuple, Optional
from config.settings import Settings, ParameterBound

logger = logging.getLogger("inframind.optimizer")


class HierarchicalParamMapper:
    """
    Maps optimizer's θ ∈ [0,1]^d to full service configuration.

    Workflow:
    1. Structure learner produces clusters: [{svc_a, svc_b}, {svc_c}, ...]
    2. Mapper creates d = n_clusters × n_per_params + n_global_params vars
    3. Optimizer proposes θ ∈ [0,1]^d
    4. Mapper decodes θ → per-service config dicts
    """

    def __init__(
        self,
        clusters: List[Set[str]],
        settings: Settings,
    ):
        """
        Parameters
        ----------
        clusters : list of sets
            Service clusters from structure learner.
        settings : Settings
            Global settings with parameter bounds.
        """
        self.clusters = clusters
        self.settings = settings
        self.param_names = list(settings.per_service_params.keys())
        self.global_param_names = list(settings.global_params.keys())

        self._n_per_service_params = len(self.param_names)
        self._n_global_params = len(self.global_param_names)
        self._n_clusters = len(clusters)

        logger.info(
            f"ParamMapper initialized: {self._n_clusters} clusters × "
            f"{self._n_per_service_params} params + "
            f"{self._n_global_params} global = {self.effective_dim}D"
        )

    @property
    def effective_dim(self) -> int:
        """Effective optimization dimensionality."""
        return self._n_clusters * self._n_per_service_params + self._n_global_params

    @property
    def full_dim(self) -> int:
        """Full (unstructured) dimensionality."""
        return self.settings.flat_dim

    @property
    def reduction_ratio(self) -> float:
        """Dimensionality reduction ratio."""
        return self.effective_dim / max(self.full_dim, 1)

    def decode(self, theta: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Map optimizer's θ ∈ [0,1]^d → per-service configuration.

        Parameters
        ----------
        theta : np.ndarray, shape (effective_dim,)
            Normalized parameter vector from optimizer.

        Returns
        -------
        config : dict[str, dict]
            Per-service config, e.g.:
            {
                "api_gateway": {"replicas": 4, "cpu_millicores": 2000, ...},
                "auth": {"replicas": 4, "cpu_millicores": 2000, ...},  # same cluster
                ...
            }
        """
        assert len(theta) == self.effective_dim, (
            f"Expected θ of dim {self.effective_dim}, got {len(theta)}"
        )

        config = {}
        idx = 0

        # Per-cluster parameters
        for cluster in self.clusters:
            cluster_params_raw = theta[idx:idx + self._n_per_service_params]
            idx += self._n_per_service_params

            # Map [0,1] → physical values
            physical_params = {}
            for j, pname in enumerate(self.param_names):
                bounds = self.settings.per_service_params[pname]
                val_01 = np.clip(cluster_params_raw[j], 0.0, 1.0)
                physical_val = bounds.min + val_01 * (bounds.max - bounds.min)

                if bounds.type == "int":
                    physical_val = int(round(physical_val))

                physical_params[pname] = physical_val

            # Apply same params to all services in cluster
            for service in cluster:
                config[service] = dict(physical_params)

        # Global parameters
        global_config = {}
        for k, gname in enumerate(self.global_param_names):
            bounds = self.settings.global_params[gname]
            val_01 = np.clip(theta[idx + k], 0.0, 1.0)
            physical_val = bounds.min + val_01 * (bounds.max - bounds.min)

            if bounds.type == "int":
                physical_val = int(round(physical_val))

            global_config[gname] = physical_val

        config["_global"] = global_config

        return config

    def encode(self, config: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Reverse map: full config → θ ∈ [0,1]^d.

        Uses the first service in each cluster as representative.
        """
        theta = np.zeros(self.effective_dim)
        idx = 0

        for cluster in self.clusters:
            representative = sorted(cluster)[0]
            svc_config = config.get(representative, {})

            for j, pname in enumerate(self.param_names):
                bounds = self.settings.per_service_params[pname]
                physical_val = svc_config.get(pname, (bounds.min + bounds.max) / 2)
                theta[idx + j] = (physical_val - bounds.min) / (bounds.max - bounds.min)

            idx += self._n_per_service_params

        # Global params
        global_config = config.get("_global", {})
        for k, gname in enumerate(self.global_param_names):
            bounds = self.settings.global_params[gname]
            physical_val = global_config.get(gname, (bounds.min + bounds.max) / 2)
            theta[idx + k] = (physical_val - bounds.min) / (bounds.max - bounds.min)

        return np.clip(theta, 0.0, 1.0)

    def get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Generate midpoint configuration (for sensitivity analysis baseline)."""
        theta_mid = np.full(self.effective_dim, 0.5)
        return self.decode(theta_mid)

    def get_random_config(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, Dict[str, Any]]:
        """Generate a random configuration within bounds."""
        rng = rng or np.random.RandomState()
        theta = rng.uniform(0, 1, size=self.effective_dim)
        return self.decode(theta)

    def get_param_labels(self) -> List[str]:
        """Human-readable labels for each dimension of θ."""
        labels = []
        for i, cluster in enumerate(self.clusters):
            cluster_name = f"C{i}({','.join(sorted(cluster)[:2])}{'...' if len(cluster) > 2 else ''})"
            for pname in self.param_names:
                labels.append(f"{cluster_name}.{pname}")
        for gname in self.global_param_names:
            labels.append(f"global.{gname}")
        return labels
