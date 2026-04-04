"""
InfraMIND v3 — Baseline Optimization Methods
==============================================

Implements all baselines for experimental comparison:

  B1: Static Provisioning — fixed config, no adaptation
  B2: Reactive Auto-Scaling — threshold-based HPA-style scaling
  B3: Vanilla Bayesian Optimization — GP-EI, no structure, no workload conditioning
  B4: Standard TuRBO — trust region BO without workload adaptation

Each baseline implements the same interface:
  .optimize(workload_trace, n_iterations) → OptimizationResult
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from scipy.stats import norm

from simulator.engine import SimulationEngine, SimulationResult
from simulator.dag import ServiceDAG
from metrics.stability_metrics import StabilityMetrics, ObjectiveValue
from config.settings import Settings

logger = logging.getLogger("inframind.baselines")


@dataclass
class OptimizationResult:
    """Stores optimization trajectory and final result."""
    method_name: str = ""
    best_config: Dict = field(default_factory=dict)
    best_objective: ObjectiveValue = field(default_factory=ObjectiveValue)
    trajectory: List[Dict] = field(default_factory=list)
    configs_evaluated: int = 0
    workload_name: str = ""
    trial: int = 0

    def to_dict(self) -> Dict:
        return {
            "method": self.method_name,
            "best_objective": self.best_objective.to_dict(),
            "configs_evaluated": self.configs_evaluated,
            "workload": self.workload_name,
            "trial": self.trial,
            "trajectory": self.trajectory,
        }


class BaseOptimizer:
    """Base class for all optimization methods."""

    name = "base"

    def __init__(
        self,
        dag: ServiceDAG,
        engine: SimulationEngine,
        metrics: StabilityMetrics,
        settings: Settings,
    ):
        self.dag = dag
        self.engine = engine
        self.metrics = metrics
        self.settings = settings
        self.rng = np.random.RandomState(settings.optimization.seed)

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        raise NotImplementedError

    def _random_config(self) -> Dict[str, Dict]:
        """Generate a random per-service configuration."""
        config = {}
        for svc in self.dag.services.values():
            config[svc.name] = {}
            for pname, bounds in self.settings.per_service_params.items():
                val = self.rng.uniform(bounds.min, bounds.max)
                if bounds.type == "int":
                    val = int(round(val))
                config[svc.name][pname] = val
        # Global params
        global_cfg = {}
        for pname, bounds in self.settings.global_params.items():
            val = self.rng.uniform(bounds.min, bounds.max)
            if bounds.type == "int":
                val = int(round(val))
            global_cfg[pname] = val
        config["_global"] = global_cfg
        return config

    def _midpoint_config(self) -> Dict[str, Dict]:
        """Generate midpoint configuration."""
        config = {}
        for svc in self.dag.services.values():
            config[svc.name] = {}
            for pname, bounds in self.settings.per_service_params.items():
                val = (bounds.min + bounds.max) / 2
                if bounds.type == "int":
                    val = int(round(val))
                config[svc.name][pname] = val
        global_cfg = {}
        for pname, bounds in self.settings.global_params.items():
            val = (bounds.min + bounds.max) / 2
            if bounds.type == "int":
                val = int(round(val))
            global_cfg[pname] = val
        config["_global"] = global_cfg
        return config

    def _evaluate(
        self,
        config: Dict,
        workload_trace: np.ndarray,
        seed: Optional[int] = None,
    ) -> ObjectiveValue:
        """Run simulation and compute objective."""
        result = self.engine.run(workload_trace, config, seed=seed)
        return self.metrics.compute(result)


# ═══════════════════════════════════════════════════════════════
# B1: Static Provisioning
# ═══════════════════════════════════════════════════════════════


class StaticBaseline(BaseOptimizer):
    """
    B1: Fixed midpoint provisioning — no adaptation.

    Simply uses the median of all parameter ranges.
    This is the naive baseline that most production systems start with.
    """

    name = "B1_Static"

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        config = self._midpoint_config()
        obj = self._evaluate(config, workload_trace)

        return OptimizationResult(
            method_name=self.name,
            best_config=config,
            best_objective=obj,
            trajectory=[{"iteration": 0, **obj.to_dict()}],
            configs_evaluated=1,
            workload_name=workload_name,
            trial=trial,
        )


# ═══════════════════════════════════════════════════════════════
# B2: Reactive Auto-Scaling (HPA Simulation)
# ═══════════════════════════════════════════════════════════════


class ReactiveBaseline(BaseOptimizer):
    """
    B2: Threshold-based reactive scaling — mimics Kubernetes HPA.

    Algorithm:
      1. Start with midpoint config
      2. Every N timesteps, check if P99 > target
      3. If violating: scale up replicas by 50%
      4. If significantly under-utilized: scale down by 25%
      5. Apply cooldown between scaling decisions
    """

    name = "B2_Reactive"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_up_factor = 1.5
        self.scale_down_factor = 0.75
        self.check_interval = 10  # Check every 10 iterations
        self.cooldown = 3         # Min iterations between scalings

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        config = self._midpoint_config()
        trajectory = []
        best_obj = None
        best_config = None
        cooldown_counter = 0

        for iteration in range(n_iterations):
            obj = self._evaluate(config, workload_trace, seed=self.settings.optimization.seed + iteration)

            trajectory.append({"iteration": iteration, **obj.to_dict()})

            if best_obj is None or obj.objective < best_obj.objective:
                best_obj = obj
                best_config = {k: dict(v) if isinstance(v, dict) else v for k, v in config.items()}

            cooldown_counter = max(0, cooldown_counter - 1)

            # Reactive scaling logic
            if cooldown_counter == 0 and iteration % self.check_interval == 0 and iteration > 0:
                if obj.p99 > self.settings.objectives.sla_target_p99_ms:
                    # Scale UP
                    self._scale_replicas(config, self.scale_up_factor)
                    cooldown_counter = self.cooldown
                    logger.debug(f"  Reactive: scale UP at iter {iteration}")
                elif obj.p99 < self.settings.objectives.sla_target_p99_ms * 0.5:
                    # Scale DOWN (under-utilized)
                    self._scale_replicas(config, self.scale_down_factor)
                    cooldown_counter = self.cooldown
                    logger.debug(f"  Reactive: scale DOWN at iter {iteration}")

        return OptimizationResult(
            method_name=self.name,
            best_config=best_config or config,
            best_objective=best_obj or ObjectiveValue(),
            trajectory=trajectory,
            configs_evaluated=n_iterations,
            workload_name=workload_name,
            trial=trial,
        )

    def _scale_replicas(self, config: Dict, factor: float):
        """Scale all service replicas by a factor."""
        for svc_name in self.dag.service_names:
            if svc_name in config:
                current = config[svc_name].get("replicas", 2)
                bounds = self.settings.per_service_params["replicas"]
                new_val = int(np.clip(current * factor, bounds.min, bounds.max))
                config[svc_name]["replicas"] = new_val


# ═══════════════════════════════════════════════════════════════
# B3: Vanilla Bayesian Optimization
# ═══════════════════════════════════════════════════════════════


class VanillaBOBaseline(BaseOptimizer):
    """
    B3: Standard Bayesian Optimization — GP-EI, flat parameter space.

    No workload conditioning (z not used).
    No structure learning (all params independent).
    No trust region adaptation.

    This shows the impact of our novel contributions.
    """

    name = "B3_VanillaBO"

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        from optimizer.surrogate import WorkloadConditionedGP

        dim = self.settings.flat_dim
        n_initial = self.settings.optimization.n_initial

        # Initialize with Sobol samples
        X_observed = []
        Y_observed = []
        trajectory = []
        best_obj = None
        best_config = None
        best_x = None

        # Initial random evaluations
        for i in range(min(n_initial, n_iterations)):
            config = self._random_config()
            theta = self._config_to_flat(config)
            obj = self._evaluate(config, workload_trace, seed=self.settings.optimization.seed + i)

            X_observed.append(theta)
            Y_observed.append(obj.objective)
            trajectory.append({"iteration": i, **obj.to_dict()})

            if best_obj is None or obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config
                best_x = theta

        # GP-EI loop
        gp = WorkloadConditionedGP(theta_dim=dim, z_dim=0)

        for i in range(n_initial, n_iterations):
            X = np.array(X_observed)
            Y = np.array(Y_observed)
            gp.fit(X, Y)

            # Generate candidates
            candidates = self.rng.uniform(0, 1, size=(5000, dim))
            mean, std = gp.predict(candidates)

            # EI
            best_f = np.min(Y)
            improvement = best_f - mean
            Z = np.zeros_like(mean)
            mask = std > 1e-8
            Z[mask] = improvement[mask] / std[mask]
            ei = np.zeros_like(mean)
            ei[mask] = improvement[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])

            best_idx = np.argmax(ei)
            next_x = candidates[best_idx]

            config = self._flat_to_config(next_x)
            obj = self._evaluate(config, workload_trace, seed=self.settings.optimization.seed + i)

            X_observed.append(next_x)
            Y_observed.append(obj.objective)
            trajectory.append({"iteration": i, **obj.to_dict()})

            if obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config
                best_x = next_x

        return OptimizationResult(
            method_name=self.name,
            best_config=best_config,
            best_objective=best_obj or ObjectiveValue(),
            trajectory=trajectory,
            configs_evaluated=n_iterations,
            workload_name=workload_name,
            trial=trial,
        )

    def _config_to_flat(self, config: Dict) -> np.ndarray:
        """Convert config to flat [0,1]^d vector."""
        theta = []
        for svc_name in self.dag.service_names:
            svc_cfg = config.get(svc_name, {})
            for pname, bounds in self.settings.per_service_params.items():
                val = svc_cfg.get(pname, (bounds.min + bounds.max) / 2)
                theta.append((val - bounds.min) / (bounds.max - bounds.min))
        for pname, bounds in self.settings.global_params.items():
            val = config.get("_global", {}).get(pname, (bounds.min + bounds.max) / 2)
            theta.append((val - bounds.min) / (bounds.max - bounds.min))
        return np.clip(np.array(theta), 0, 1)

    def _flat_to_config(self, theta: np.ndarray) -> Dict:
        """Convert flat [0,1]^d vector to config."""
        config = {}
        idx = 0
        for svc_name in self.dag.service_names:
            config[svc_name] = {}
            for pname, bounds in self.settings.per_service_params.items():
                val = bounds.min + theta[idx] * (bounds.max - bounds.min)
                if bounds.type == "int":
                    val = int(round(val))
                config[svc_name][pname] = val
                idx += 1
        config["_global"] = {}
        for pname, bounds in self.settings.global_params.items():
            val = bounds.min + theta[idx] * (bounds.max - bounds.min)
            if bounds.type == "int":
                val = int(round(val))
            config["_global"][pname] = val
            idx += 1
        return config


# ═══════════════════════════════════════════════════════════════
# B4: Standard TuRBO (No Workload Adaptation)
# ═══════════════════════════════════════════════════════════════


class StandardTuRBOBaseline(BaseOptimizer):
    """
    B4: Standard TuRBO — trust region BO without our extensions.

    Has trust region expansion/shrinkage, but:
      - No workload conditioning (z not used)
      - No structure learning (flat parameter space)
      - No workload-adaptive trust region scaling

    This isolates the impact of our novel contributions C1-C3.
    """

    name = "B4_StandardTuRBO"

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        from optimizer.surrogate import WorkloadConditionedGP
        from optimizer.adaptive_turbo import AdaptiveTuRBO

        dim = self.settings.flat_dim
        n_initial = self.settings.optimization.n_initial
        turbo_cfg = self.settings.optimization.turbo

        # Standard TuRBO (volatility_alpha=0 disables workload adaptation)
        turbo = AdaptiveTuRBO(
            dim=dim,
            length_init=turbo_cfg.length_init,
            length_min=turbo_cfg.length_min,
            length_max=turbo_cfg.length_max,
            success_tolerance=turbo_cfg.success_tolerance,
            failure_tolerance=turbo_cfg.failure_tolerance,
            volatility_alpha=0.0,  # ← KEY: disables workload adaptation
            seed=self.settings.optimization.seed,
        )

        gp = WorkloadConditionedGP(theta_dim=dim, z_dim=0)

        X_observed = []
        Y_observed = []
        trajectory = []
        best_obj = None
        best_config = None

        vanilla_bo = VanillaBOBaseline(self.dag, self.engine, self.metrics, self.settings)

        # Initial evaluations
        for i in range(min(n_initial, n_iterations)):
            config = self._random_config()
            theta = vanilla_bo._config_to_flat(config)
            obj = self._evaluate(config, workload_trace, seed=self.settings.optimization.seed + i)

            X_observed.append(theta)
            Y_observed.append(obj.objective)
            trajectory.append({"iteration": i, **obj.to_dict()})
            turbo.update_state(obj.objective, theta)

            if best_obj is None or obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

        # TuRBO loop
        for i in range(n_initial, n_iterations):
            X = np.array(X_observed)
            Y = np.array(Y_observed)
            gp.fit(X, Y)

            center = turbo.get_center()
            candidates = turbo.suggest_candidates(
                center=center,
                workload_volatility=0.0,  # No workload adaptation
                n_candidates=5000,
                batch_size=1,
                model=gp,
            )

            next_x = candidates[0]
            config = vanilla_bo._flat_to_config(next_x)
            obj = self._evaluate(config, workload_trace, seed=self.settings.optimization.seed + i)

            X_observed.append(next_x)
            Y_observed.append(obj.objective)
            trajectory.append({"iteration": i, **obj.to_dict()})
            turbo.update_state(obj.objective, next_x)

            if obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

        return OptimizationResult(
            method_name=self.name,
            best_config=best_config,
            best_objective=best_obj or ObjectiveValue(),
            trajectory=trajectory,
            configs_evaluated=n_iterations,
            workload_name=workload_name,
            trial=trial,
        )
