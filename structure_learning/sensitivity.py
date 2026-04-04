"""
InfraMIND v3 — Sensitivity Analyzer (Contribution C2, Part 1)
==============================================================

Computes the parameter sensitivity matrix S ∈ ℝ^(n_services × n_params)
via finite difference approximation:

    S[i,j] = ∂Latency_P99 / ∂θ_ij ≈ ( f(θ + δe_ij) - f(θ - δe_ij) ) / (2δ)

This matrix captures how sensitive end-to-end latency is to each
service's parameter, enabling data-driven service clustering.

Complexity:
  - O(n_services × n_params × 2) simulation evaluations
  - For 7 services × 4 params = 56 simulations
  - Each simulation is O(n_requests × path_length)
  - Total: O(56 × N × L) ≈ manageable

Scaling Note:
  - For >20 services, computation grows linearly but may become
    expensive. In practice, structure is learned once and cached.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional

from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine, SimulationResult
from config.settings import Settings

logger = logging.getLogger("inframind.structure_learning")


class SensitivityAnalyzer:
    """
    Computes parameter–latency sensitivity matrix for structure learning.

    For each (service, parameter) pair, perturbs the parameter by ±δ
    and measures the change in P99 end-to-end latency.
    """

    def __init__(
        self,
        dag: ServiceDAG,
        engine: SimulationEngine,
        settings: Settings,
        delta: float = 0.05,
    ):
        """
        Parameters
        ----------
        dag : ServiceDAG
            Service topology.
        engine : SimulationEngine
            Simulation engine for evaluating configs.
        settings : Settings
            Global settings.
        delta : float
            Fractional perturbation size (e.g., 0.05 = 5%).
        """
        self.dag = dag
        self.engine = engine
        self.settings = settings
        self.delta = delta
        self.param_names = list(settings.per_service_params.keys())

    def compute_sensitivity_matrix(
        self,
        base_config: Dict[str, Dict[str, Any]],
        workload_trace: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the sensitivity matrix S.

        Parameters
        ----------
        base_config : dict
            Per-service configuration at which to compute sensitivities.
        workload_trace : np.ndarray
            Workload trace for simulation.

        Returns
        -------
        S : np.ndarray, shape (n_services, n_params)
            Sensitivity matrix where S[i,j] = ∂P99/∂θ_ij.
        """
        n_services = len(self.dag.service_names)
        n_params = len(self.param_names)
        S = np.zeros((n_services, n_params))

        base_seed = seed or self.settings.optimization.seed

        logger.info(
            f"Computing sensitivity matrix: {n_services} services × "
            f"{n_params} params = {n_services * n_params * 2} simulations"
        )

        # Baseline P99
        base_result = self.engine.run(workload_trace, base_config, seed=base_seed)
        base_p99 = float(np.percentile(base_result.latencies, 99))
        logger.info(f"Baseline P99: {base_p99:.2f}ms")

        for i, svc_name in enumerate(self.dag.service_names):
            for j, param_name in enumerate(self.param_names):
                # Get bounds for this parameter
                bounds = self.settings.per_service_params[param_name]
                base_val = base_config.get(svc_name, {}).get(
                    param_name, (bounds.min + bounds.max) / 2
                )

                # Compute perturbation
                param_range = bounds.max - bounds.min
                delta_val = self.delta * param_range

                # Forward perturbation
                config_plus = self._perturb(base_config, svc_name, param_name, base_val + delta_val, bounds)
                result_plus = self.engine.run(workload_trace, config_plus, seed=base_seed)
                p99_plus = float(np.percentile(result_plus.latencies, 99))

                # Backward perturbation
                config_minus = self._perturb(base_config, svc_name, param_name, base_val - delta_val, bounds)
                result_minus = self.engine.run(workload_trace, config_minus, seed=base_seed)
                p99_minus = float(np.percentile(result_minus.latencies, 99))

                # Central difference
                sensitivity = (p99_plus - p99_minus) / (2 * delta_val) if delta_val > 0 else 0.0
                S[i, j] = sensitivity

                logger.debug(
                    f"  {svc_name}.{param_name}: "
                    f"P99-={p99_minus:.1f} P99+={p99_plus:.1f} → S={sensitivity:.4f}"
                )

        return S

    def compute_impact_scores(
        self,
        sensitivity_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-service impact scores from the sensitivity matrix.

        Impact score = L2 norm of a service's sensitivity vector.
        Higher impact services affect latency more strongly.

        Returns
        -------
        scores : np.ndarray, shape (n_services,)
        """
        return np.linalg.norm(sensitivity_matrix, axis=1)

    def _perturb(
        self,
        base_config: Dict[str, Dict],
        service: str,
        param: str,
        new_value: float,
        bounds: Any,
    ) -> Dict[str, Dict]:
        """Create a perturbed copy of the config."""
        import copy
        config = copy.deepcopy(base_config)

        # Clamp to bounds
        clamped = np.clip(new_value, bounds.min, bounds.max)

        # Apply type
        if bounds.type == "int":
            clamped = int(round(clamped))

        if service not in config:
            config[service] = {}
        config[service][param] = clamped

        return config

    def get_sensitivity_report(
        self,
        S: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate a human-readable sensitivity report.

        Returns dict with:
          - most_sensitive_service
          - most_sensitive_param
          - sensitivity_ranking
          - matrix (for serialization)
        """
        impact_scores = self.compute_impact_scores(S)
        service_ranking = sorted(
            zip(self.dag.service_names, impact_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Most sensitive (service, param) pair
        flat_idx = np.argmax(np.abs(S))
        max_i, max_j = np.unravel_index(flat_idx, S.shape)

        return {
            "most_sensitive_service": self.dag.service_names[max_i],
            "most_sensitive_param": self.param_names[max_j],
            "max_sensitivity": float(np.abs(S[max_i, max_j])),
            "service_ranking": [(name, float(score)) for name, score in service_ranking],
            "matrix": S.tolist(),
            "service_names": self.dag.service_names,
            "param_names": self.param_names,
        }
