"""
InfraMIND v4 — Training WebSocket (Real Bayesian Optimization)
===============================================================

Runs the actual AdaptiveTuRBO + WorkloadConditionedGP optimization loop
over the SimPy simulation engine. Each epoch:

  1. GP fits on all observed (θ, z, objective) data
  2. TuRBO proposes candidates within an adapted trust region
  3. Acquisition function (Expected Improvement) ranks candidates
  4. Best candidate is simulated through the full DAG engine
  5. Results streamed to the frontend in real-time
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import numpy as np
import logging
from api.models.schemas import TrainingConfig

# Real InfraMIND engine components
from config.settings import get_default_config
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
from api.services.utils import clean_config_for_json

logger = logging.getLogger("inframind.training_ws")

# Training constants
MAX_WS_EPOCHS = 50  # Cap per WebSocket session for responsiveness
REWARD_SCALE_FACTOR = 3.0  # Maps objective (0.1-20) to reward (0-100)

router = APIRouter(prefix="/ws", tags=["websockets"])
active_training = False
current_config = TrainingConfig()


@router.websocket("/training")
async def training_ws(websocket: WebSocket):
    global active_training, current_config
    await websocket.accept()

    try:
        while True:
            # Check for commands from client
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                if data.get("type") == "START":
                    active_training = True
                    current_config = TrainingConfig(**data.get("config", {}))
                elif data.get("type") == "STOP":
                    active_training = False
                elif data.get("type") == "UPDATE_WEIGHTS":
                    current_config = TrainingConfig(**data.get("config", {}))
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            if active_training:
                await _run_real_optimization(websocket)

    except WebSocketDisconnect:
        logger.info("Training WS Disconnected")


async def _run_real_optimization(websocket: WebSocket):
    """Run the actual AdaptiveTuRBO Bayesian Optimization loop."""
    global active_training, current_config

    # ── 1. Initialize all engine components ──────────────────────
    settings = get_default_config()
    dag = ServiceDAG(settings)

    metrics = StabilityMetrics(
        sla_target_ms=settings.objectives.sla_target_p99_ms,
        lambda_sla=settings.objectives.lambda_sla,
        lambda_variance=settings.objectives.lambda_variance,
    )

    # Generate workload trace
    workload_gen = WorkloadGenerator(seed=42)
    profile = get_profile("bursty")
    trace = workload_gen.generate(
        profile,
        duration_s=settings.simulation.duration_s,
        resolution_s=settings.simulation.resolution_s,
    )

    # Workload embedding (conditions the GP)
    embedder = WorkloadEmbedder()
    z = embedder.embed(trace)
    volatility = embedder.compute_volatility(trace)

    # Parameter mapper (one cluster per service for now)
    clusters = [{name} for name in dag.service_names]
    mapper = HierarchicalParamMapper(clusters, settings)
    dim = mapper.effective_dim

    # ── 2. Initialize optimizer components ───────────────────────
    turbo_cfg = settings.optimization.turbo
    turbo = AdaptiveTuRBO(
        dim=dim,
        length_init=turbo_cfg.length_init,
        length_min=turbo_cfg.length_min,
        length_max=turbo_cfg.length_max,
        success_tolerance=turbo_cfg.success_tolerance,
        failure_tolerance=turbo_cfg.failure_tolerance,
        volatility_alpha=turbo_cfg.volatility_alpha,
        seed=42,
    )
    gp = WorkloadConditionedGP(
        theta_dim=dim,
        z_dim=WorkloadEmbedder.EMBEDDING_DIM,
    )
    acq = TrustRegionAcquisition(seed=42)

    n_initial = settings.optimization.n_initial
    epochs = min(current_config.epochs, MAX_WS_EPOCHS)
    rng = np.random.RandomState(42)

    # Storage
    X_all = []  # GP inputs: [θ ∥ z]
    Y_all = []  # Objective values
    reward_history = []
    best_reward = -1e9
    best_obj_value = float("inf")
    best_config = mapper.get_default_config()

    # ── 3. Optimization loop ─────────────────────────────────────
    for epoch in range(1, epochs + 1):
        if not active_training:
            break

        # Phase selection: initial random exploration vs. BO
        if epoch <= n_initial:
            theta = rng.uniform(0, 1, size=dim)
            phase = "INIT"
        else:
            # Fit GP on accumulated data
            X = np.array(X_all)
            Y = np.array(Y_all)
            gp.fit(X, Y)

            # Get trust region from TuRBO
            center = turbo.get_center()
            lb, ub = turbo.get_trust_region(center, volatility)

            # Acquisition-guided candidate selection
            candidates = acq.optimize(
                model=gp,
                best_f=turbo.best_value,
                lb=lb,
                ub=ub,
                n_candidates=2000,  # Reduced from 5000 for speed
                batch_size=1,
                workload_embedding=z,
            )
            theta = candidates[0]
            phase = "BO"

        # Decode θ → physical service config
        config = mapper.decode(theta)

        # Run real simulation
        engine = SimulationEngine(dag, settings)
        result = engine.run(trace, config, seed=42 + epoch)

        # Compute real objective
        obj = metrics.compute(result)

        # Store for GP
        x_gp = np.concatenate([theta, z])
        X_all.append(x_gp)
        Y_all.append(obj.objective)

        # Update TuRBO trust region state
        turbo.update_state(obj.objective, theta)

        # Track best
        if obj.objective < best_obj_value:
            best_obj_value = obj.objective
            best_config = config

        # Map objective (usually 0.1 to 20.0) to a 0-100 score
        # A perfect score 100 means objective = 0
        reward = max(0.0, 100.0 - (obj.objective * REWARD_SCALE_FACTOR))

        if reward > best_reward:
            best_reward = reward

        reward_history.append(reward)

        # Stream to frontend
        await websocket.send_json({
            "epoch": epoch,
            "total_epochs": epochs,
            "reward": round(reward, 2),
            "loss": round(obj.objective, 4),
            "best_reward": round(best_reward, 2),
            "reward_history": [round(r, 2) for r in reward_history],
            "status": "running",
            "phase": phase,
            "turbo_length": round(turbo.length, 4),
            "p99_ms": round(obj.p99, 1),
            "cost": round(obj.cost, 4),
        })

        # Yield to event loop
        await asyncio.sleep(0.01)

        # Listen for weight updates or stop
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
            if data.get("type") == "STOP":
                active_training = False
            elif data.get("type") == "UPDATE_WEIGHTS":
                current_config = TrainingConfig(**data.get("config", {}))
        except asyncio.TimeoutError:
            pass

    # ── 4. Training complete ─────────────────────────────────────
    active_training = False

    clean_config = clean_config_for_json(best_config)

    await websocket.send_json({
        "status": "completed",
        "best_reward": round(best_reward, 2),
        "total_epochs": epochs,
        "optimized_config": clean_config,
    })
