"""
InfraMIND v3 — Run Single Optimization Demo
=============================================

Quick demo: runs a single InfraMIND v3 optimization loop with
live console output. Good for smoke testing.

Usage:
    python scripts/run_single_optimization.py --n-iter 10
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_default_config, set_global_seed
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics
from workloads.generator import WorkloadGenerator
from workloads.profiles import get_profile
from embeddings.workload_embedder import WorkloadEmbedder
from structure_learning.sensitivity import SensitivityAnalyzer
from structure_learning.cluster import ServiceClusterer
from optimizer.param_mapper import HierarchicalParamMapper
from optimizer.surrogate import WorkloadConditionedGP
from optimizer.adaptive_turbo import AdaptiveTuRBO
from optimizer.acquisition import TrustRegionAcquisition

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="InfraMIND v3 — Single Optimization Demo")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of BO iterations")
    parser.add_argument("--workload", type=str, default="bursty", choices=["steady", "diurnal", "bursty"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-structure", action="store_true", help="Skip structure learning")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    )
    logger = logging.getLogger("inframind.demo")

    print()
    print("=" * 70)
    print("  🧠 InfraMIND v3 — Single Optimization Demo")
    print("=" * 70)
    print()

    # 1. Load config
    settings = get_default_config()
    set_global_seed(args.seed)
    logger.info(f"Configuration loaded: {settings.n_services} services, {settings.flat_dim}D flat space")

    # 2. Build DAG
    dag = ServiceDAG(settings)
    logger.info(f"DAG built:\n{dag}")
    crit_path, crit_time = dag.critical_path()
    logger.info(f"Critical path: {' → '.join(crit_path)} = {crit_time:.0f}ms")

    # 3. Generate workload
    generator = WorkloadGenerator(seed=args.seed)
    profile = get_profile(args.workload)
    trace = generator.generate(profile, duration_s=settings.simulation.duration_s, resolution_s=settings.simulation.resolution_s)
    logger.info(f"Workload generated: {args.workload}, {len(trace)} timesteps")

    # 4. Embed workload
    embedder = WorkloadEmbedder()
    z = embedder.embed(trace)
    volatility = embedder.compute_volatility(trace)
    print(f"\n📊 Workload Embedding:")
    for name, val in zip(WorkloadEmbedder.FEATURE_NAMES, z):
        print(f"  {name:>20s} = {val:.4f}")
    print(f"  {'volatility':>20s} = {volatility:.4f}")

    # 5. Structure Learning
    engine = SimulationEngine(dag, settings)
    metrics = StabilityMetrics(
        sla_target_ms=settings.objectives.sla_target_p99_ms,
        lambda_sla=settings.objectives.lambda_sla,
        lambda_variance=settings.objectives.lambda_variance,
    )

    if args.skip_structure:
        clusters = [{name} for name in dag.service_names]
        logger.info("Structure learning SKIPPED (one cluster per service)")
    else:
        print(f"\n🔬 Running Structure Learning...")
        mapper_default = HierarchicalParamMapper([{name} for name in dag.service_names], settings)
        base_config = mapper_default.get_default_config()
        analyzer = SensitivityAnalyzer(dag, engine, settings, delta=settings.structure_learning.sensitivity_delta)
        S = analyzer.compute_sensitivity_matrix(base_config, trace)

        report = analyzer.get_sensitivity_report(S)
        print(f"\n📋 Sensitivity Report:")
        print(f"  Most sensitive: {report['most_sensitive_service']}.{report['most_sensitive_param']}")
        for name, score in report['service_ranking']:
            print(f"  {name:>20s}: impact = {score:.4f}")

        clusterer = ServiceClusterer(dag.service_names, method=settings.structure_learning.method)
        clusters = clusterer.cluster(S)
        print(f"\n🏗️ Service Clusters:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: {sorted(cluster)}")

    # 6. Setup Optimization
    mapper = HierarchicalParamMapper(clusters, settings)
    dim = mapper.effective_dim
    turbo_cfg = settings.optimization.turbo

    turbo = AdaptiveTuRBO(
        dim=dim, length_init=turbo_cfg.length_init,
        length_min=turbo_cfg.length_min, length_max=turbo_cfg.length_max,
        success_tolerance=turbo_cfg.success_tolerance,
        failure_tolerance=turbo_cfg.failure_tolerance,
        volatility_alpha=turbo_cfg.volatility_alpha,
        seed=args.seed,
    )
    gp = WorkloadConditionedGP(theta_dim=dim, z_dim=WorkloadEmbedder.EMBEDDING_DIM)
    acq = TrustRegionAcquisition(seed=args.seed)

    print(f"\n⚙️ Optimization Setup:")
    print(f"  Effective dim: {dim} (reduced from {settings.flat_dim})")
    print(f"  Reduction: {mapper.reduction_ratio:.1%}")
    print(f"  Initial samples: {settings.optimization.n_initial}")
    print(f"  BO iterations: {args.n_iter}")

    # 7. Run Optimization
    rng = np.random.RandomState(args.seed)
    X_all = []
    Y_all = []
    best_obj = None
    best_config = None
    n_initial = min(settings.optimization.n_initial, args.n_iter)

    print(f"\n🚀 Starting Optimization Loop...")
    print("-" * 70)

    for i in range(args.n_iter):
        if i < n_initial:
            theta = rng.uniform(0, 1, size=dim)
            phase = "INIT"
        else:
            X = np.array(X_all)
            Y = np.array(Y_all)
            gp.fit(X, Y)
            center = turbo.get_center()
            lb, ub = turbo.get_trust_region(center, volatility)
            candidates = acq.optimize(
                model=gp, best_f=turbo.best_value,
                lb=lb, ub=ub, n_candidates=5000, batch_size=1,
                workload_embedding=z,
            )
            theta = candidates[0]
            phase = "BO  "

        config = mapper.decode(theta)
        result = engine.run(trace, config, seed=args.seed + i)
        obj = metrics.compute(result)

        x_gp = np.concatenate([theta, z])
        X_all.append(x_gp)
        Y_all.append(obj.objective)
        turbo.update_state(obj.objective, theta)

        improved = "★" if best_obj is None or obj.objective < best_obj.objective else " "
        if best_obj is None or obj.objective < best_obj.objective:
            best_obj = obj
            best_config = config

        print(
            f"  {improved} [{phase}] iter {i:3d} | "
            f"obj={obj.objective:8.3f} | cost={obj.cost:.4f} | "
            f"p99={obj.p99:6.1f}ms | var={obj.latency_variance:8.1f} | "
            f"TR={turbo.length:.4f} | best={best_obj.objective:.3f}"
        )

    # 8. Summary
    print()
    print("=" * 70)
    print("  📊 OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\n  Best Objective:     {best_obj.objective:.4f}")
    print(f"  Best Cost:          ${best_obj.cost:.4f}")
    print(f"  Best P99 Latency:   {best_obj.p99:.1f} ms")
    print(f"  Latency Variance:   {best_obj.latency_variance:.1f}")
    print(f"  SLA Violations:     {best_obj.sla_violation_rate:.2%}")
    print(f"  Feasible:           {'✅' if best_obj.is_feasible else '❌'}")
    print(f"  Stability Score:    {StabilityMetrics.compute_stability_score(np.array([best_obj.p50, best_obj.p99])):.3f}")
    print(f"  TuRBO Restarts:     {turbo.n_restarts}")
    print(f"  Configs Evaluated:  {args.n_iter}")
    print()


if __name__ == "__main__":
    main()
