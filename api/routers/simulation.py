from fastapi import APIRouter, HTTPException
import time
from api.models.schemas import SimulateRequest
from config.settings import get_default_config
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from metrics.stability_metrics import StabilityMetrics
from workloads.generator import WorkloadGenerator
from workloads.profiles import get_profile

router = APIRouter(prefix="/simulate", tags=["simulation"])

# Pre-load shared resources
settings = get_default_config()
dag = ServiceDAG(settings)
metrics_engine = StabilityMetrics(
    sla_target_ms=settings.objectives.sla_target_p99_ms,
    lambda_sla=settings.objectives.lambda_sla,
    lambda_variance=settings.objectives.lambda_variance
)
workload_gen = WorkloadGenerator(seed=42)

@router.post("")
async def run_simulation(request: SimulateRequest):
    try:
        # Generate workload trace
        profile = get_profile(request.workload_type)
        trace = workload_gen.generate(
            profile,
            duration_s=request.duration_s,
            resolution_s=1.0  # seconds
        )
        
        # Initialize simulator
        engine = SimulationEngine(dag, settings)
        
        # Run
        sim_result = engine.run(trace, request.config)
        
        # Compute metrics
        obj_value = metrics_engine.compute(sim_result)
        
        return {
            "p50_latency_ms": obj_value.p50,
            "p90_latency_ms": obj_value.p90,
            "p99_latency_ms": obj_value.p99,
            "sla_violations_pct": sim_result.drop_rate * 100,
            "cost_estimate": obj_value.cost,
            "dropped_requests": sim_result.dropped_requests
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
