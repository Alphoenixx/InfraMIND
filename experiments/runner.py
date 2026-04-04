"""
InfraMIND v3 — Experiment Runner
==================================

Main orchestrator for running full comparison experiments:
  - All baselines × all workloads × multiple trials
  - InfraMIND v3 full system
  - Saves results as structured JSON

Pipeline per method/workload/trial:
  1. Generate workload trace
  2. Initialize optimizer
  3. Run optimization loop
  4. Log trajectory + best config + metrics
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from config.settings import Settings, get_default_config, set_global_seed, RESULTS_DIR
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics
from workloads.generator import WorkloadGenerator
from workloads.profiles import WorkloadProfile, ALL_PROFILES, get_profile
from embeddings.workload_embedder import WorkloadEmbedder
from structure_learning.sensitivity import SensitivityAnalyzer
from structure_learning.cluster import ServiceClusterer
from optimizer.param_mapper import HierarchicalParamMapper
from optimizer.surrogate import WorkloadConditionedGP
from optimizer.adaptive_turbo import AdaptiveTuRBO
from optimizer.acquisition import TrustRegionAcquisition
from optimizer.baselines import (
    StaticBaseline, ReactiveBaseline,
    VanillaBOBaseline, StandardTuRBOBaseline,
    OptimizationResult, BaseOptimizer,
)

logger = logging.getLogger("inframind.experiments")


class InfraMINDv3Optimizer(BaseOptimizer):
    """
    B5: InfraMIND v3 — Full system with all contributions.

    Combines:
      C1: Workload-conditioned GP surrogate
      C2: Structure-learned parameter mapping
      C3: Adaptive trust region scaling
      C4: Stability-aware objective
      C5: DAG simulation
      C6: Cross-workload generalization
    """

    name = "B5_InfraMINDv3"

    def __init__(
        self,
        dag: ServiceDAG,
        engine: SimulationEngine,
        metrics: StabilityMetrics,
        settings: Settings,
        clusters: Optional[List] = None,
    ):
        super().__init__(dag, engine, metrics, settings)
        self.clusters = clusters
        self.embedder = WorkloadEmbedder()

    def optimize(
        self,
        workload_trace: np.ndarray,
        n_iterations: int = 50,
        workload_name: str = "",
        trial: int = 0,
    ) -> OptimizationResult:
        # ── Setup param mapper ──────────────────────────────
        if self.clusters is None:
            # Default: one cluster per service
            clusters = [{name} for name in self.dag.service_names]
        else:
            clusters = self.clusters

        mapper = HierarchicalParamMapper(clusters, self.settings)
        dim = mapper.effective_dim
        turbo_cfg = self.settings.optimization.turbo
        n_initial = self.settings.optimization.n_initial

        # ── Workload embedding ──────────────────────────────
        z = self.embedder.embed(workload_trace)
        volatility = self.embedder.compute_volatility(workload_trace)
        logger.info(f"Workload: {workload_name}, embedding={z}, volatility={volatility:.4f}")

        # ── Initialize optimizer ────────────────────────────
        turbo = AdaptiveTuRBO(
            dim=dim,
            length_init=turbo_cfg.length_init,
            length_min=turbo_cfg.length_min,
            length_max=turbo_cfg.length_max,
            success_tolerance=turbo_cfg.success_tolerance,
            failure_tolerance=turbo_cfg.failure_tolerance,
            volatility_alpha=turbo_cfg.volatility_alpha,
            seed=self.settings.optimization.seed + trial,
        )

        gp = WorkloadConditionedGP(theta_dim=dim, z_dim=WorkloadEmbedder.EMBEDDING_DIM)
        acq = TrustRegionAcquisition(seed=self.settings.optimization.seed + trial)

        X_observed = []  # [θ ∥ z] inputs
        X_theta = []     # θ only (for TR centering)
        Y_observed = []
        trajectory = []
        best_obj = None
        best_config = None

        # ── Initial evaluations (Sobol) ─────────────────────
        for i in range(min(n_initial, n_iterations)):
            theta = self.rng.uniform(0, 1, size=dim)
            config = mapper.decode(theta)
            obj = self._evaluate(config, workload_trace,
                                seed=self.settings.optimization.seed + i)

            x_gp = np.concatenate([theta, z])
            X_observed.append(x_gp)
            X_theta.append(theta)
            Y_observed.append(obj.objective)
            turbo.update_state(obj.objective, theta)

            trajectory.append({"iteration": i, **obj.to_dict()})

            if best_obj is None or obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

            logger.debug(f"  Init {i}/{n_initial}: obj={obj.objective:.4f}")

        # ── Adaptive TuRBO loop ─────────────────────────────
        for i in range(n_initial, n_iterations):
            X = np.array(X_observed)
            Y = np.array(Y_observed)
            gp.fit(X, Y)

            center = turbo.get_center()
            lb, ub = turbo.get_trust_region(center, volatility)

            # Generate candidates via acquisition function
            candidates = acq.optimize(
                model=gp,
                best_f=turbo.best_value,
                lb=lb,
                ub=ub,
                n_candidates=5000,
                batch_size=1,
                workload_embedding=z,
            )

            next_theta = candidates[0]
            config = mapper.decode(next_theta)
            obj = self._evaluate(config, workload_trace,
                                seed=self.settings.optimization.seed + i)

            x_gp = np.concatenate([next_theta, z])
            X_observed.append(x_gp)
            X_theta.append(next_theta)
            Y_observed.append(obj.objective)
            turbo.update_state(obj.objective, next_theta)

            trajectory.append({"iteration": i, **obj.to_dict()})

            if obj.objective < best_obj.objective:
                best_obj = obj
                best_config = config

            if i % 10 == 0:
                logger.info(
                    f"  Iter {i}/{n_iterations}: obj={obj.objective:.4f}, "
                    f"best={best_obj.objective:.4f}, TR={turbo.length:.4f}"
                )

        return OptimizationResult(
            method_name=self.name,
            best_config=best_config,
            best_objective=best_obj or ObjectiveValue(),
            trajectory=trajectory,
            configs_evaluated=n_iterations,
            workload_name=workload_name,
            trial=trial,
        )


class ExperimentRunner:
    """
    Orchestrates full experiments: methods × workloads × trials.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.settings = get_default_config(config_path)
        self.dag = ServiceDAG(self.settings)
        self.engine = SimulationEngine(self.dag, self.settings)
        self.metrics = StabilityMetrics(
            sla_target_ms=self.settings.objectives.sla_target_p99_ms,
            lambda_sla=self.settings.objectives.lambda_sla,
            lambda_variance=self.settings.objectives.lambda_variance,
        )
        self.generator = WorkloadGenerator(seed=self.settings.optimization.seed)
        self.all_results: List[OptimizationResult] = []

    def _learn_structure(self, workload_trace: np.ndarray):
        """Run structure learning to get service clusters."""
        logger.info("Running structure learning...")
        mapper_default = HierarchicalParamMapper(
            [{name} for name in self.dag.service_names],
            self.settings,
        )
        base_config = mapper_default.get_default_config()

        analyzer = SensitivityAnalyzer(
            self.dag, self.engine, self.settings,
            delta=self.settings.structure_learning.sensitivity_delta,
        )
        S = analyzer.compute_sensitivity_matrix(base_config, workload_trace)

        clusterer = ServiceClusterer(
            self.dag.service_names,
            method=self.settings.structure_learning.method,
            min_clusters=self.settings.structure_learning.min_clusters,
            max_clusters=self.settings.structure_learning.max_clusters,
        )
        clusters = clusterer.cluster(
            S,
            n_clusters=self.settings.structure_learning.n_clusters,
        )

        report = analyzer.get_sensitivity_report(S)
        logger.info(f"Service ranking: {report['service_ranking']}")
        logger.info(f"Clusters: {[sorted(c) for c in clusters]}")

        return clusters, S

    def run_full_comparison(
        self,
        workload_names: Optional[List[str]] = None,
        n_iterations: int = 50,
        n_trials: int = 3,
    ):
        """Run all methods across all workloads."""
        if workload_names is None:
            workload_names = ["steady", "diurnal", "bursty"]

        set_global_seed(self.settings.optimization.seed)

        # Generate a representative workload for structure learning
        steady_trace = self.generator.generate(
            get_profile("steady"),
            duration_s=self.settings.simulation.duration_s,
            resolution_s=self.settings.simulation.resolution_s,
        )
        clusters, sensitivity_matrix = self._learn_structure(steady_trace)

        # Initialize all methods
        methods = {
            "B1_Static": StaticBaseline(self.dag, self.engine, self.metrics, self.settings),
            "B2_Reactive": ReactiveBaseline(self.dag, self.engine, self.metrics, self.settings),
            "B3_VanillaBO": VanillaBOBaseline(self.dag, self.engine, self.metrics, self.settings),
            "B4_StandardTuRBO": StandardTuRBOBaseline(self.dag, self.engine, self.metrics, self.settings),
            "B5_InfraMINDv3": InfraMINDv3Optimizer(
                self.dag, self.engine, self.metrics, self.settings, clusters=clusters
            ),
        }

        for wl_name in workload_names:
            profile = get_profile(wl_name)

            for trial in range(n_trials):
                trace = self.generator.generate(
                    profile,
                    duration_s=self.settings.simulation.duration_s,
                    resolution_s=self.settings.simulation.resolution_s,
                )

                for method_name, method in methods.items():
                    logger.info(f"═══ {method_name} | {wl_name} | trial {trial} ═══")
                    start = time.time()

                    result = method.optimize(
                        trace,
                        n_iterations=n_iterations,
                        workload_name=wl_name,
                        trial=trial,
                    )

                    elapsed = time.time() - start
                    logger.info(
                        f"  → obj={result.best_objective.objective:.4f}, "
                        f"cost={result.best_objective.cost:.4f}, "
                        f"p99={result.best_objective.p99:.1f}ms, "
                        f"time={elapsed:.1f}s"
                    )

                    self.all_results.append(result)

        return self.all_results

    def save_results(self, output_dir: Optional[str] = None):
        """Save all results as JSON."""
        out_dir = Path(output_dir) if output_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out_dir / f"experiment_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "settings": {
                "n_services": self.settings.n_services,
                "service_names": self.settings.service_names,
                "sla_target_p99_ms": self.settings.objectives.sla_target_p99_ms,
                "lambda_sla": self.settings.objectives.lambda_sla,
                "lambda_variance": self.settings.objectives.lambda_variance,
            },
            "results": [r.to_dict() for r in self.all_results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath
