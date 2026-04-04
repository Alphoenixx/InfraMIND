"""
InfraMIND v3 — Full Experiment Runner Script
==============================================

End-to-end: runs all baselines × all workloads, ablation study,
generalization test, and saves results.

Usage:
    python scripts/run_full_experiment.py --n-iter 30 --n-trials 2
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_default_config, set_global_seed, RESULTS_DIR
from experiments.runner import ExperimentRunner
from experiments.ablation import run_ablation_study
from experiments.generalization import run_generalization_test
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics
from workloads.generator import WorkloadGenerator
from workloads.profiles import get_profile
from structure_learning.sensitivity import SensitivityAnalyzer
from structure_learning.cluster import ServiceClusterer
from optimizer.param_mapper import HierarchicalParamMapper


def main():
    parser = argparse.ArgumentParser(description="InfraMIND v3 — Full Experiment Suite")
    parser.add_argument("--n-iter", type=int, default=30, help="BO iterations per method")
    parser.add_argument("--n-trials", type=int, default=2, help="Trials per method/workload")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-generalization", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    )
    logger = logging.getLogger("inframind.experiment")

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  🧠 InfraMIND v3 — Full Experiment Suite" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    settings = get_default_config()
    settings.optimization.seed = args.seed
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # ═══════════════════════════════════════════
    # Phase 1: Structure Learning
    # ═══════════════════════════════════════════
    print("━" * 70)
    print("  Phase 1: Structure Learning")
    print("━" * 70)

    dag = ServiceDAG(settings)
    engine = SimulationEngine(dag, settings)
    generator = WorkloadGenerator(seed=args.seed)
    steady_trace = generator.generate(get_profile("steady"),
        duration_s=settings.simulation.duration_s,
        resolution_s=settings.simulation.resolution_s)

    mapper_flat = HierarchicalParamMapper([{n} for n in dag.service_names], settings)
    base_config = mapper_flat.get_default_config()

    analyzer = SensitivityAnalyzer(dag, engine, settings)
    S = analyzer.compute_sensitivity_matrix(base_config, steady_trace)
    clusterer = ServiceClusterer(dag.service_names, method=settings.structure_learning.method)
    clusters = clusterer.cluster(S)

    cluster_report = clusterer.get_cluster_report(clusters, S)
    print(f"  Clusters: {[sorted(c) for c in clusters]}")
    print(f"  Dimensionality: {settings.flat_dim}D → {HierarchicalParamMapper(clusters, settings).effective_dim}D")

    # ═══════════════════════════════════════════
    # Phase 2: Baseline Comparison
    # ═══════════════════════════════════════════
    print("\n" + "━" * 70)
    print("  Phase 2: Full Baseline Comparison")
    print("━" * 70)

    runner = ExperimentRunner()
    comparison_results = runner.run_full_comparison(
        n_iterations=args.n_iter,
        n_trials=args.n_trials,
    )

    # ═══════════════════════════════════════════
    # Phase 3: Ablation Study
    # ═══════════════════════════════════════════
    ablation_results = []
    if not args.skip_ablation:
        print("\n" + "━" * 70)
        print("  Phase 3: Ablation Study")
        print("━" * 70)

        ablation_results = run_ablation_study(
            settings, clusters,
            workload_name="bursty",
            n_iterations=args.n_iter,
            n_trials=1,
        )

    # ═══════════════════════════════════════════
    # Phase 4: Generalization Test
    # ═══════════════════════════════════════════
    gen_result = None
    if not args.skip_generalization:
        print("\n" + "━" * 70)
        print("  Phase 4: Cross-Workload Generalization")
        print("━" * 70)

        gen_result = run_generalization_test(
            settings, clusters,
            n_iterations=args.n_iter,
        )

    # ═══════════════════════════════════════════
    # Save All Results
    # ═══════════════════════════════════════════
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = runner.save_results(str(output_dir))

    # Save ablation results separately
    if ablation_results:
        abl_path = output_dir / f"ablation_{timestamp}.json"
        abl_data = {"results": [r.to_dict() for r in ablation_results]}
        with open(abl_path, "w") as f:
            json.dump(abl_data, f, indent=2, default=str)
        print(f"  Ablation results saved to {abl_path}")

    # Save generalization results
    if gen_result:
        gen_path = output_dir / f"generalization_{timestamp}.json"
        with open(gen_path, "w") as f:
            json.dump(gen_result.to_dict(), f, indent=2, default=str)
        print(f"  Generalization results saved to {gen_path}")

    total_time = time.time() - total_start

    print("\n" + "═" * 70)
    print(f"  ✅ ALL EXPERIMENTS COMPLETE — {total_time:.0f}s total")
    print("═" * 70)
    print(f"  Results: {filepath}")
    print()


if __name__ == "__main__":
    main()
