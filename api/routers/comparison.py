from fastapi import APIRouter, HTTPException
from api.models.schemas import CompareRequest
from config.settings import get_default_config
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics
from workloads.generator import WorkloadGenerator
from workloads.profiles import get_profile
from optimizer.param_mapper import HierarchicalParamMapper
import numpy as np
from api.services.utils import clean_config_for_json

router = APIRouter(prefix="/compare", tags=["comparison"])

settings = get_default_config()
dag = ServiceDAG(settings)
metrics_engine = StabilityMetrics(
    sla_target_ms=settings.objectives.sla_target_p99_ms,
    lambda_sla=settings.objectives.lambda_sla,
    lambda_variance=settings.objectives.lambda_variance
)
workload_gen = WorkloadGenerator(seed=42)


def _get_default_config():
    """Generate the DAG midpoint default config (no optimization)."""
    clusters = [{name} for name in dag.service_names]
    mapper = HierarchicalParamMapper(clusters, settings)
    config = mapper.get_default_config()
    return clean_config_for_json(config)


def _run_sim(config: dict, workload_type: str, duration_s: float):
    profile = get_profile(workload_type)
    trace = workload_gen.generate(profile, duration_s=duration_s, resolution_s=1.0)
    engine = SimulationEngine(dag, settings)
    sim_result = engine.run(trace, config)
    obj_value = metrics_engine.compute(sim_result)
    return {
        "p50_latency_ms": obj_value.p50,
        "p90_latency_ms": obj_value.p90,
        "p99_latency_ms": obj_value.p99,
        "sla_violations_pct": sim_result.drop_rate * 100,
        "cost_estimate": obj_value.cost,
        "dropped_requests": sim_result.dropped_requests,
    }


@router.post("")
async def compare_configs(request: CompareRequest):
    try:
        # If baseline is empty, use DAG midpoint defaults
        baseline_config = request.baseline_config if request.baseline_config else _get_default_config()

        baseline = _run_sim(baseline_config, request.workload_type, request.duration_s)
        optimized = _run_sim(request.optimized_config, request.workload_type, request.duration_s)

        def pct_delta(old, new):
            if old == 0:
                return 0 if new == 0 else 100.0
            return round((new - old) / abs(old) * 100, 1)

        deltas = {
            "p99_change_pct": pct_delta(baseline["p99_latency_ms"], optimized["p99_latency_ms"]),
            "cost_change_pct": pct_delta(baseline["cost_estimate"], optimized["cost_estimate"]),
            "sla_change_pct": pct_delta(baseline["sla_violations_pct"], optimized["sla_violations_pct"]),
        }

        # Auto-generate insight
        parts = []
        if deltas["p99_change_pct"] < 0:
            parts.append(f"P99 latency improved by {abs(deltas['p99_change_pct'])}%")
        elif deltas["p99_change_pct"] > 0:
            parts.append(f"P99 latency increased by {deltas['p99_change_pct']}%")

        if deltas["cost_change_pct"] > 0:
            parts.append(f"cost increased by {deltas['cost_change_pct']}%")
        elif deltas["cost_change_pct"] < 0:
            parts.append(f"cost decreased by {abs(deltas['cost_change_pct'])}%")

        insight = "The optimized configuration " + (", ".join(parts) if parts else "is similar to baseline") + ". This represents the Pareto-optimal trade-off for your current weight preferences."

        return {
            "baseline": baseline,
            "optimized": optimized,
            "baseline_config": baseline_config,
            "deltas": deltas,
            "insight": insight,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

