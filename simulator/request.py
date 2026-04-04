"""
InfraMIND v3 — Request Data Class
===================================

Tracks a single request as it propagates through the microservice DAG,
recording per-hop latencies and final disposition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Request:
    """
    A single request traversing the service DAG.

    Attributes
    ----------
    request_id : int
        Unique identifier for this request.
    created_at : float
        Simulation time when the request was generated.
    completed_at : float or None
        Simulation time when the request finished (all services visited).
    path : list of str
        Ordered list of services visited by this request.
    per_hop_latencies : dict
        Maps service_name → latency (ms) experienced at that hop.
    dropped : bool
        True if the request was dropped due to queue overflow.
    drop_service : str or None
        Name of the service where the request was dropped.
    """
    request_id: int = 0
    created_at: float = 0.0
    completed_at: Optional[float] = None
    path: List[str] = field(default_factory=list)
    per_hop_latencies: Dict[str, float] = field(default_factory=dict)
    dropped: bool = False
    drop_service: Optional[str] = None

    @property
    def end_to_end_latency(self) -> Optional[float]:
        """Total latency from creation to completion (ms)."""
        if self.completed_at is None:
            return None
        return (self.completed_at - self.created_at) * 1000.0  # convert to ms

    @property
    def total_service_time(self) -> float:
        """Sum of per-hop latencies (ms)."""
        return sum(self.per_hop_latencies.values())

    @property
    def total_queue_time(self) -> Optional[float]:
        """End-to-end latency minus service times = total queuing time."""
        e2e = self.end_to_end_latency
        if e2e is None:
            return None
        return max(0.0, e2e - self.total_service_time)

    def record_hop(self, service_name: str, latency_ms: float):
        """Record a completed hop."""
        self.path.append(service_name)
        self.per_hop_latencies[service_name] = latency_ms

    def mark_dropped(self, service_name: str, time: float):
        """Mark this request as dropped at a service."""
        self.dropped = True
        self.drop_service = service_name
        self.completed_at = time

    def mark_completed(self, time: float):
        """Mark this request as successfully completed."""
        self.completed_at = time
