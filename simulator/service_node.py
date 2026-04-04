"""
InfraMIND v3 — Service Node (SimPy Process)
============================================

Models a single microservice as a SimPy process with:
  - Resource pool (replicas) modeled as simpy.Resource
  - Bounded request queue modeled as simpy.Store
  - Stochastic service time (exponential distribution)
  - Service time scaling with CPU allocation
  - Queue overflow detection (request dropping)
"""

import simpy
import numpy as np
from typing import Dict, Optional, List, Callable
from .request import Request


class ServiceNode:
    """
    A single microservice node in the DAG simulation.

    Parameters
    ----------
    env : simpy.Environment
        Shared simulation environment.
    name : str
        Service name (must match DAG node name).
    replicas : int
        Number of concurrent workers (simpy.Resource capacity).
    base_service_time : float
        Mean service time in milliseconds.
    cpu_millicores : int
        CPU allocation. Higher CPU → lower effective service time.
    queue_capacity : int
        Max queue depth before requests are dropped.
    downstream_queues : dict
        Maps downstream service name → simpy.Store to push requests into.
    """

    # CPU scaling: service_time = base_time × (1000 / cpu_millicores)
    # At 1000mc (1 full core), service time equals base.
    # At 2000mc, service time halved. At 500mc, doubled.
    CPU_REFERENCE = 1000.0

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        replicas: int = 2,
        base_service_time: float = 10.0,
        cpu_millicores: int = 1000,
        queue_capacity: int = 100,
        downstream_queues: Optional[Dict[str, simpy.Store]] = None,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.env = env
        self.name = name
        self.replicas = replicas
        self.base_service_time = base_service_time
        self.cpu_millicores = cpu_millicores
        self.queue_capacity = queue_capacity
        self.downstream_queues = downstream_queues or {}
        self.rng = rng or np.random.RandomState()

        # SimPy resources
        self.resource = simpy.Resource(env, capacity=replicas)
        self.input_queue = simpy.Store(env, capacity=queue_capacity)

        # Metrics tracking
        self.processed_count = 0
        self.dropped_count = 0
        self.latencies: List[float] = []
        self.queue_lengths: List[float] = []

        # Start the worker process
        self.env.process(self._worker_loop())

    @property
    def effective_service_time(self) -> float:
        """Compute effective service time based on CPU allocation."""
        return self.base_service_time * (self.CPU_REFERENCE / max(self.cpu_millicores, 1))

    def submit_request(self, request: Request) -> bool:
        """
        Submit a request to this service's input queue.

        Returns True if enqueued, False if dropped (queue full).
        """
        if len(self.input_queue.items) >= self.queue_capacity:
            request.mark_dropped(self.name, self.env.now)
            self.dropped_count += 1
            return False

        self.input_queue.put(request)
        self.queue_lengths.append(len(self.input_queue.items))
        return True

    def _worker_loop(self):
        """Main service worker — pulls from queue and processes."""
        while True:
            # Wait for a request
            request = yield self.input_queue.get()

            if request.dropped:
                continue

            # Acquire a replica (worker thread)
            with self.resource.request() as req:
                yield req

                # Sample service time from exponential distribution
                service_time_ms = self.rng.exponential(self.effective_service_time)
                service_time_s = service_time_ms / 1000.0

                yield self.env.timeout(service_time_s)

                # Record hop
                request.record_hop(self.name, service_time_ms)
                self.latencies.append(service_time_ms)
                self.processed_count += 1

                # Propagate to downstream services
                if self.downstream_queues:
                    for ds_name, ds_queue in self.downstream_queues.items():
                        # Create a forwarding — the downstream service will
                        # pick it up from its own queue
                        ds_queue.put(request)
                else:
                    # Leaf service — mark request complete
                    request.mark_completed(self.env.now)

    def get_stats(self) -> Dict:
        """Return service-level statistics."""
        latencies = np.array(self.latencies) if self.latencies else np.array([0.0])
        return {
            "name": self.name,
            "processed": self.processed_count,
            "dropped": self.dropped_count,
            "replicas": self.replicas,
            "cpu_millicores": self.cpu_millicores,
            "effective_service_time_ms": self.effective_service_time,
            "mean_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "avg_queue_length": float(np.mean(self.queue_lengths)) if self.queue_lengths else 0.0,
        }

    def reset(self):
        """Reset metrics for a new simulation run."""
        self.processed_count = 0
        self.dropped_count = 0
        self.latencies.clear()
        self.queue_lengths.clear()
