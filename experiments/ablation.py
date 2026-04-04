"""
InfraMIND v3 — Ablation Study Driver
======================================

Systematically removes one contribution at a time to measure
its marginal impact on optimization quality:

  -embedding:     Remove workload conditioning (z = 0)
  -structure:     Remove structure learning (flat parameter space)
  -adaptive_tr:   Remove workload-adaptive trust region (α = 0)
  -stability:     Remove variance penalty (λ₂ = 0)

Each ablation runs InfraMIND v3 with one component disabled,
keeping everything else identical. This proves that each
contribution is independently valuable (ablation monotonicity).
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from config.settings import Settings, get_default_config, set_global_seed
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics
from workloads.generator import WorkloadGenerator
from workloads.profiles import get_profile
from embeddings.workload_embedder import WorkloadEmbedder
from optimizer.param_mapper import HierarchicalParamMapper
from optimizer.surrogate import WorkloadConditionedGP
from optimizer.adaptive_turbo import AdaptiveTuRBO
from optimizer.acquisition import TrustRegionAcquisition
from optimizer.baselines import OptimizationResult, BaseOptimizer

logger = logging.getLogger("inframind.ablation")


@dataclass
class AblationConfig:
    """Configuration for an ablation variant."""
    name: str
    disable_embedding: bool = False
    disable_structure: bool = False
    disable_adaptive_tr: bool = False
    disable_stability: bool = False


# Define all ablation variants
ABLATION_VARIANTS = [
    AblationConfig(name="full", disable_embedding=False, disable_structure=False,
                   disable_adaptive_tr=False, disable_stability=False),
    AblationConfig(name="-embedding", disable_embedding=True),
    AblationConfig(name="-structure", disable_structure=True),
    AblationConfig(name="-adaptive_tr", disable_adaptive_tr=True),
    AblationConfig(name="-stability", disable_stability=True),
]


class AblatedInfraMIND(BaseOptimizer):
    """
    InfraMIND v3 with selectively disabled components for ablation.
    """

    def __init__(
        self,
        dag: ServiceDAG,
        engine: SimulationEngine,
        metrics: StabilityMetrics,
        settings: Settings,
        ablation: AblationConfig,
        clusters: Optional[List] = None,
    ):
        super().__init__(dag, engine, metrics, settings)
        self.ablation = ablation
        self.clusters = clusters
        self.embedder = WorkloadEmbedder()
        self.name = f"ablation_{ablation.name}"

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        # ── Structure ablation ──────────────────────────────
        if self.ablation.disable_structure or self.clusters is None:
            clusters = [{name} for name in self.dag.service_names]
        else:
            clusters = self.clusters

        mapper = HierarchicalParamMapper(clusters, self.settings)
        dim = mapper.effective_dim
        turbo_cfg = self.settings.optimization.turbo
        n_initial = self.settings.optimization.n_initial

        # ── Embedding ablation ──────────────────────────────
        if self.ablation.disable_embedding:
            z = np.zeros(WorkloadEmbedder.EMBEDDING_DIM)
            volatility = 0.0
            z_dim = WorkloadEmbedder.EMBEDDING_DIM  # Still need same GP input dim
        else:
            z = self.embedder.embed(workload_trace)
            volatility = self.embedder.compute_volatility(workload_trace)
            z_dim = WorkloadEmbedder.EMBEDDING_DIM

        # ── Adaptive TR ablation ────────────────────────────
        alpha = 0.0 if self.ablation.disable_adaptive_tr else turbo_cfg.volatility_alpha

        # ── Stability ablation (λ₂ = 0) ────────────────────
        if self.ablation.disable_stability:
            metrics = StabilityMetrics(
                sla_target_ms=self.settings.objectives.sla_target_p99_ms,
                lambda_sla=self.settings.objectives.lambda_sla,
                lambda_variance=0.0,  # ← ablated
            )
        else:
            metrics = self.metrics

        # ── Optimizer setup ─────────────────────────────────
        turbo = AdaptiveTuRBO(
            dim=dim,
            length_init=turbo_cfg.length_init,
            length_min=turbo_cfg.length_min,
            length_max=turbo_cfg.length_max,
            success_tolerance=turbo_cfg.success_tolerance,
            failure_tolerance=turbo_cfg.failure_tolerance,
            volatility_alpha=alpha,
            seed=self.settings.optimization.seed + trial,
        )

        gp = WorkloadConditionedGP(theta_dim=dim, z_dim=z_dim)
        acq = TrustRegionAcquisition(seed=self.settings.optimization.seed + trial)

        X_observed = []
        Y_observed = []
        trajectory = []
        best_obj = None
        best_config = None

        # ── Initial evaluations ─────────────────────────────
        for i in range(min(n_initial, n_iterations)):
            theta = self.rng.uniform(0, 1, size=dim)
            config = mapper.decode(theta)

            result = self.engine.run(
                workload_trace, config,
                seed=self.settings.optimization.seed + i,
            )
            obj = metrics.compute(result)

            x_gp = np.concatenate([theta, z])
            X_observed.append(x_gp)
            Y_observed.append(obj.objective)
            turbo.update_state(obj.objective, theta)

            trajectory.append({"iteration": i, **obj.to_dict()})

            if best_obj is None or obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

        # ── Optimization loop ───────────────────────────────
        for i in range(n_initial, n_iterations):
            X = np.array(X_observed)
            Y = np.array(Y_observed)
            gp.fit(X, Y)

            center = turbo.get_center()
            lb, ub = turbo.get_trust_region(center, volatility)

            candidates = acq.optimize(
                model=gp,
                best_f=turbo.best_value,
                lb=lb, ub=ub,
                n_candidates=5000,
                batch_size=1,
                workload_embedding=z,
            )

            next_theta = candidates[0]
            config = mapper.decode(next_theta)

            sim_result = self.engine.run(
                workload_trace, config,
                seed=self.settings.optimization.seed + i,
            )
            obj = metrics.compute(sim_result)

            x_gp = np.concatenate([next_theta, z])
            X_observed.append(x_gp)
            Y_observed.append(obj.objective)
            turbo.update_state(obj.objective, next_theta)

            trajectory.append({"iteration": i, **obj.to_dict()})

            if obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

        from metrics.stability_metrics import ObjectiveValue
        return OptimizationResult(
            method_name=self.name,
            best_config=best_config,
            best_objective=best_obj or ObjectiveValue(),
            trajectory=trajectory,
            configs_evaluated=n_iterations,
            workload_name=workload_name,
            trial=trial,
        )


def run_ablation_study(
    settings: Settings,
    clusters: List,
    workload_name: str = "bursty",
    n_iterations: int = 50,
    n_trials: int = 3,
) -> List[OptimizationResult]:
    """
    Run full ablation study across all variants.

    Returns results for: full, -embedding, -structure, -adaptive_tr, -stability
    """
    dag = ServiceDAG(settings)
    engine = SimulationEngine(dag, settings)
    metrics = StabilityMetrics(
        sla_target_ms=settings.objectives.sla_target_p99_ms,
        lambda_sla=settings.objectives.lambda_sla,
        lambda_variance=settings.objectives.lambda_variance,
    )
    generator = WorkloadGenerator(seed=settings.optimization.seed)
    profile = get_profile(workload_name)

    all_results = []

    for variant in ABLATION_VARIANTS:
        for trial_idx in range(n_trials):
            trace = generator.generate(
                profile,
                duration_s=settings.simulation.duration_s,
                resolution_s=settings.simulation.resolution_s,
            )

            optimizer = AblatedInfraMIND(
                dag, engine, metrics, settings,
                ablation=variant,
                clusters=clusters,
            )

            logger.info(f"═══ Ablation: {variant.name} | {workload_name} | trial {trial_idx} ═══")
            result = optimizer.optimize(
                trace,
                n_iterations=n_iterations,
                workload_name=workload_name,
                trial=trial_idx,
            )

            logger.info(f"  → obj={result.best_objective.objective:.4f}")
            all_results.append(result)

    return all_results
