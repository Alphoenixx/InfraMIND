"""
InfraMIND v3 — Cross-Workload Generalization Test (Contribution C6)
====================================================================

Tests the ability of the workload-conditioned model to generalize
to unseen traffic distributions:

    Train: Steady + Diurnal workloads
    Test:  Bursty workload (never seen during training)

This evaluates zero-shot transfer capability:
  - The GP surrogate was trained with z embeddings from steady/diurnal
  - At test time, the bursty embedding z_bursty is far from training data
  - If the workload conditioning is effective, the model should still
    propose configurations that work well for bursty traffic
  - If it fails, the model overfits to the training workload distribution

Success criteria:
  - Performance gap between train and test workloads < 20%
  - InfraMIND v3 on unseen workload > VanillaBO on same workload
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config.settings import Settings, get_default_config, set_global_seed
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics, ObjectiveValue
from workloads.generator import WorkloadGenerator
from workloads.profiles import get_profile
from embeddings.workload_embedder import WorkloadEmbedder
from optimizer.param_mapper import HierarchicalParamMapper
from optimizer.surrogate import WorkloadConditionedGP
from optimizer.adaptive_turbo import AdaptiveTuRBO
from optimizer.acquisition import TrustRegionAcquisition
from optimizer.baselines import OptimizationResult

logger = logging.getLogger("inframind.generalization")


@dataclass
class GeneralizationResult:
    """Results of the cross-workload generalization test."""
    train_workloads: List[str]
    test_workload: str
    train_results: List[OptimizationResult]
    test_result: OptimizationResult
    baseline_test_result: OptimizationResult  # VanillaBO on test workload

    @property
    def train_best_obj(self) -> float:
        return min(r.best_objective.objective for r in self.train_results)

    @property
    def test_obj(self) -> float:
        return self.test_result.best_objective.objective

    @property
    def baseline_test_obj(self) -> float:
        return self.baseline_test_result.best_objective.objective

    @property
    def generalization_gap(self) -> float:
        """Relative gap between train and test performance."""
        if self.train_best_obj == 0:
            return 0.0
        return abs(self.test_obj - self.train_best_obj) / abs(self.train_best_obj)

    @property
    def improvement_over_baseline(self) -> float:
        """Relative improvement over vanilla BO on test workload."""
        if self.baseline_test_obj == 0:
            return 0.0
        return (self.baseline_test_obj - self.test_obj) / abs(self.baseline_test_obj)

    def to_dict(self) -> Dict:
        return {
            "train_workloads": self.train_workloads,
            "test_workload": self.test_workload,
            "train_best_objective": self.train_best_obj,
            "test_objective": self.test_obj,
            "baseline_test_objective": self.baseline_test_obj,
            "generalization_gap": self.generalization_gap,
            "improvement_over_baseline": self.improvement_over_baseline,
            "train_results": [r.to_dict() for r in self.train_results],
            "test_result": self.test_result.to_dict(),
            "baseline_test_result": self.baseline_test_result.to_dict(),
        }


def run_generalization_test(
    settings: Settings,
    clusters: List,
    train_workloads: Optional[List[str]] = None,
    test_workload: str = "bursty",
    n_iterations: int = 50,
) -> GeneralizationResult:
    """
    Run the cross-workload generalization experiment.

    Phase 1: Train InfraMIND v3 on steady + diurnal
    Phase 2: Evaluate on bursty (zero-shot)
    Phase 3: Compare with VanillaBO trained directly on bursty
    """
    if train_workloads is None:
        train_workloads = ["steady", "diurnal"]

    dag = ServiceDAG(settings)
    engine = SimulationEngine(dag, settings)
    metrics = StabilityMetrics(
        sla_target_ms=settings.objectives.sla_target_p99_ms,
        lambda_sla=settings.objectives.lambda_sla,
        lambda_variance=settings.objectives.lambda_variance,
    )
    generator = WorkloadGenerator(seed=settings.optimization.seed)
    embedder = WorkloadEmbedder()

    mapper = HierarchicalParamMapper(clusters, settings)
    dim = mapper.effective_dim
    turbo_cfg = settings.optimization.turbo

    # ═══════════════════════════════════════════════════════
    # Phase 1: Train on steady + diurnal
    # ═══════════════════════════════════════════════════════
    logger.info("Phase 1: Training on steady + diurnal workloads")

    # Collect training data across workloads
    X_train = []
    Y_train = []
    train_results = []

    turbo = AdaptiveTuRBO(
        dim=dim, length_init=turbo_cfg.length_init,
        length_min=turbo_cfg.length_min, length_max=turbo_cfg.length_max,
        success_tolerance=turbo_cfg.success_tolerance,
        failure_tolerance=turbo_cfg.failure_tolerance,
        volatility_alpha=turbo_cfg.volatility_alpha,
        seed=settings.optimization.seed,
    )
    gp = WorkloadConditionedGP(theta_dim=dim, z_dim=WorkloadEmbedder.EMBEDDING_DIM)
    acq = TrustRegionAcquisition(seed=settings.optimization.seed)

    rng = np.random.RandomState(settings.optimization.seed)

    for wl_name in train_workloads:
        profile = get_profile(wl_name)
        trace = generator.generate(
            profile,
            duration_s=settings.simulation.duration_s,
            resolution_s=settings.simulation.resolution_s,
        )
        z = embedder.embed(trace)
        volatility = embedder.compute_volatility(trace)

        trajectory = []
        best_obj = None
        best_config = None
        n_per_wl = n_iterations // len(train_workloads)

        for i in range(n_per_wl):
            if len(X_train) < settings.optimization.n_initial:
                theta = rng.uniform(0, 1, size=dim)
            else:
                X = np.array(X_train)
                Y = np.array(Y_train)
                gp.fit(X, Y)
                center = turbo.get_center()
                lb, ub = turbo.get_trust_region(center, volatility)
                candidates = acq.optimize(
                    model=gp, best_f=turbo.best_value,
                    lb=lb, ub=ub, n_candidates=5000, batch_size=1,
                    workload_embedding=z,
                )
                theta = candidates[0]

            config = mapper.decode(theta)
            result = engine.run(trace, config, seed=settings.optimization.seed + i)
            obj = metrics.compute(result)

            x_gp = np.concatenate([theta, z])
            X_train.append(x_gp)
            Y_train.append(obj.objective)
            turbo.update_state(obj.objective, theta)

            trajectory.append({"iteration": i, **obj.to_dict()})

            if best_obj is None or obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

        train_results.append(OptimizationResult(
            method_name=f"InfraMINDv3_train_{wl_name}",
            best_config=best_config,
            best_objective=best_obj or ObjectiveValue(),
            trajectory=trajectory,
            configs_evaluated=n_per_wl,
            workload_name=wl_name,
        ))
        logger.info(f"  Train {wl_name}: best_obj={best_obj.objective:.4f}")

    # ═══════════════════════════════════════════════════════
    # Phase 2: Zero-shot test on bursty
    # ═══════════════════════════════════════════════════════
    logger.info(f"Phase 2: Zero-shot evaluation on {test_workload}")

    test_profile = get_profile(test_workload)
    test_trace = generator.generate(
        test_profile,
        duration_s=settings.simulation.duration_s,
        resolution_s=settings.simulation.resolution_s,
    )
    z_test = embedder.embed(test_trace)
    volatility_test = embedder.compute_volatility(test_trace)

    # Refit GP on all training data
    X = np.array(X_train)
    Y = np.array(Y_train)
    gp.fit(X, Y)

    # Use the trained model to suggest configs for the unseen workload
    test_trajectory = []
    test_best_obj = None
    test_best_config = None
    n_test_iter = n_iterations // 2

    for i in range(n_test_iter):
        center = turbo.get_center()
        lb, ub = turbo.get_trust_region(center, volatility_test)
        candidates = acq.optimize(
            model=gp, best_f=turbo.best_value,
            lb=lb, ub=ub, n_candidates=5000, batch_size=1,
            workload_embedding=z_test,
        )

        theta = candidates[0]
        config = mapper.decode(theta)
        result = engine.run(test_trace, config, seed=settings.optimization.seed + 1000 + i)
        obj = metrics.compute(result)

        x_gp = np.concatenate([theta, z_test])
        X_train.append(x_gp)
        Y_train.append(obj.objective)
        turbo.update_state(obj.objective, theta)

        # Refit periodically
        if i % 5 == 0 and len(X_train) > 10:
            gp.fit(np.array(X_train), np.array(Y_train))

        test_trajectory.append({"iteration": i, **obj.to_dict()})

        if test_best_obj is None or obj.objective < test_best_obj.objective:
            test_best_obj = obj
            test_best_config = config

    test_result = OptimizationResult(
        method_name="InfraMINDv3_test_zeroshot",
        best_config=test_best_config,
        best_objective=test_best_obj or ObjectiveValue(),
        trajectory=test_trajectory,
        configs_evaluated=n_test_iter,
        workload_name=test_workload,
    )

    logger.info(f"  Test {test_workload}: best_obj={test_best_obj.objective:.4f}")

    # ═══════════════════════════════════════════════════════
    # Phase 3: Baseline comparison (VanillaBO on test workload)
    # ═══════════════════════════════════════════════════════
    logger.info(f"Phase 3: VanillaBO baseline on {test_workload}")

    from optimizer.baselines import VanillaBOBaseline
    vanilla = VanillaBOBaseline(dag, engine, metrics, settings)
    baseline_result = vanilla.optimize(
        test_trace,
        n_iterations=n_test_iter,
        workload_name=test_workload,
    )

    logger.info(f"  Baseline {test_workload}: best_obj={baseline_result.best_objective.objective:.4f}")

    # ═══════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════
    gen_result = GeneralizationResult(
        train_workloads=train_workloads,
        test_workload=test_workload,
        train_results=train_results,
        test_result=test_result,
        baseline_test_result=baseline_result,
    )

    logger.info(f"Generalization gap: {gen_result.generalization_gap:.2%}")
    logger.info(f"Improvement over baseline: {gen_result.improvement_over_baseline:.2%}")

    return gen_result
