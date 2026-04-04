"""
InfraMIND v3 — Simulation Engine (Contribution C5)
====================================================

Orchestrates the full microservice DAG simulation:
  1. Builds service nodes from configuration
  2. Wires up queues according to DAG topology
  3. Generates requests following a workload trace
  4. Runs the SimPy simulation
  5. Collects end-to-end latency + cost metrics

This is the central simulation loop that the optimizer calls
to evaluate candidate configurations.
"""

import simpy
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .dag import ServiceDAG
from .service_node import ServiceNode
from .request import Request
from config.settings import Settings

logger = logging.getLogger("inframind.simulator")


@dataclass
class SimulationResult:
    """
    Aggregated results from a single simulation run.

    Contains all information needed by the metrics engine to
    compute the stability-aware objective.
    """
    # End-to-end latencies (ms) for all completed requests
    latencies: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-service latency arrays
    per_service_latencies: Dict[str, np.ndarray] = field(default_factory=dict)
    # Per-service stats
    per_service_stats: Dict[str, Dict] = field(default_factory=dict)
    # Request disposition
    total_requests: int = 0
    completed_requests: int = 0
    dropped_requests: int = 0
    # Cost
    total_cost: float = 0.0
    # Simulation time actually used
    simulation_time_s: float = 0.0

    @property
    def completion_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.completed_requests / self.total_requests

    @property
    def drop_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.dropped_requests / self.total_requests


class SimulationEngine:
    """
    Orchestrates SimPy-based microservice DAG simulation.

    Usage
    -----
    >>> engine = SimulationEngine(dag, settings)
    >>> result = engine.run(workload_trace, service_config)
    """

    def __init__(self, dag: ServiceDAG, settings: Settings):
        """
        Parameters
        ----------
        dag : ServiceDAG
            DAG topology.
        settings : Settings
            Global settings.
        """
        self.dag = dag
        self.settings = settings

    def run(
        self,
        workload_trace: np.ndarray,
        service_config: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Execute a full simulation run.

        Parameters
        ----------
        workload_trace : np.ndarray
            Arrival rates λ(t) in req/s, shape (n_timesteps,).
        service_config : dict
            Per-service configuration, e.g.:
            {
                "api_gateway": {"replicas": 4, "cpu_millicores": 2000, ...},
                "auth": {"replicas": 2, ...},
                ...
            }

        Returns
        -------
        result : SimulationResult
        """
        rng = np.random.RandomState(seed or self.settings.optimization.seed)
        env = simpy.Environment()

        # Build service nodes with wired queues
        nodes, entry_queues = self._build_nodes(env, service_config, rng)

        # Track all requests
        all_requests: List[Request] = []

        # Start request generator
        env.process(self._request_generator(
            env, workload_trace, entry_queues, all_requests, rng
        ))

        # Run simulation
        duration = self.settings.simulation.duration_s
        env.run(until=duration)

        # Collect results
        return self._collect_results(
            nodes, all_requests, service_config, duration
        )

    def _build_nodes(
        self,
        env: simpy.Environment,
        service_config: Dict[str, Dict],
        rng: np.random.RandomState,
    ):
        """
        Instantiate ServiceNode objects and wire their queues
        according to the DAG topology.
        """
        # First pass: create input queues for each service
        input_queues: Dict[str, simpy.Store] = {}
        for name in self.dag.service_names:
            cfg = service_config.get(name, {})
            capacity = int(cfg.get("queue_capacity", 100))
            input_queues[name] = simpy.Store(env, capacity=capacity)

        # Second pass: create nodes with downstream queue references
        nodes: Dict[str, ServiceNode] = {}
        for name in self.dag.service_names:
            cfg = service_config.get(name, {})
            svc_def = self.dag.services[name]

            downstream_queues = {}
            for ds_name in self.dag.get_downstream(name):
                downstream_queues[ds_name] = input_queues[ds_name]

            node = ServiceNode(
                env=env,
                name=name,
                replicas=int(cfg.get("replicas", svc_def.base_replicas)),
                base_service_time=svc_def.base_service_time,
                cpu_millicores=int(cfg.get("cpu_millicores", 1000)),
                queue_capacity=int(cfg.get("queue_capacity", 100)),
                downstream_queues=downstream_queues,
                rng=np.random.RandomState(rng.randint(0, 2**31)),
            )
            # Override the node's input queue with the shared one
            node.input_queue = input_queues[name]
            nodes[name] = node

        # Entry queues (roots of the DAG)
        entry_queues = {name: input_queues[name] for name in self.dag.entry_points}

        return nodes, entry_queues

    def _request_generator(
        self,
        env: simpy.Environment,
        workload_trace: np.ndarray,
        entry_queues: Dict[str, simpy.Store],
        all_requests: List[Request],
        rng: np.random.RandomState,
    ):
        """
        SimPy process that generates requests following the workload trace.

        Uses Poisson process: inter-arrival time ~ Exponential(1/λ(t))
        """
        resolution = self.settings.simulation.resolution_s
        warmup = self.settings.simulation.warmup_s
        request_id = 0

        for step, rate in enumerate(workload_trace):
            step_start = step * resolution
            step_end = step_start + resolution

            if rate <= 0:
                yield env.timeout(resolution)
                continue

            # Generate Poisson arrivals within this time step
            n_arrivals = rng.poisson(rate * resolution)

            if n_arrivals == 0:
                yield env.timeout(resolution)
                continue

            # Distribute arrivals uniformly within the step
            arrival_offsets = np.sort(rng.uniform(0, resolution, size=n_arrivals))

            prev_offset = 0.0
            for offset in arrival_offsets:
                # Wait until this arrival time
                wait = offset - prev_offset
                if wait > 0:
                    yield env.timeout(wait)
                prev_offset = offset

                # Create request
                req = Request(request_id=request_id, created_at=env.now)
                request_id += 1

                # Only track requests after warmup period
                if env.now >= warmup:
                    all_requests.append(req)

                # Submit to all entry points (fan-out at gateway)
                for entry_name, queue in entry_queues.items():
                    queue.put(req)

            # Wait for remainder of this step
            remaining = resolution - prev_offset
            if remaining > 0:
                yield env.timeout(remaining)

    def _collect_results(
        self,
        nodes: Dict[str, ServiceNode],
        all_requests: List[Request],
        service_config: Dict[str, Dict],
        duration_s: float,
    ) -> SimulationResult:
        """Aggregate simulation results."""
        # End-to-end latencies
        completed = [r for r in all_requests if not r.dropped and r.completed_at is not None]
        dropped = [r for r in all_requests if r.dropped]

        latencies = np.array([r.end_to_end_latency for r in completed]) if completed else np.array([0.0])

        # Per-service latencies
        per_service_latencies = {}
        per_service_stats = {}
        for name, node in nodes.items():
            per_service_latencies[name] = np.array(node.latencies) if node.latencies else np.array([0.0])
            per_service_stats[name] = node.get_stats()

        # Cost computation
        cost_per_unit = self.settings.objectives.cost_per_replica_per_hour
        hours = duration_s / 3600.0
        total_cost = 0.0
        for name, node in nodes.items():
            cpu_factor = node.cpu_millicores / 1000.0  # Normalize to cores
            total_cost += node.replicas * cpu_factor * cost_per_unit * hours

        return SimulationResult(
            latencies=latencies,
            per_service_latencies=per_service_latencies,
            per_service_stats=per_service_stats,
            total_requests=len(all_requests),
            completed_requests=len(completed),
            dropped_requests=len(dropped),
            total_cost=total_cost,
            simulation_time_s=duration_s,
        )
