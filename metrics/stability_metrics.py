"""
InfraMIND v3 — Stability-Aware Metrics Engine (Contribution C4)
================================================================

Computes the stability-aware multi-component objective:

    obj = Cost + λ₁·SLA_violations + λ₂·Var(Latency)

This is the KEY DIFFERENTIATOR from threshold-only optimization:

    Standard approach:  minimize Cost  s.t.  P99 ≤ target
    Our approach:       minimize Cost + penalty(violations) + penalty(variance)

Why variance matters:
  - Two configs can both satisfy P99 < 200ms
  - But one might oscillate between 50ms and 199ms
  - While the other stays consistently at 120ms ± 10ms
  - The stable one is operationally superior
  - Variance penalty captures this distinction

Complexity: O(n) where n = number of completed requests
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from simulator.engine import SimulationResult


@dataclass
class ObjectiveValue:
    """
    Complete objective decomposition — every metric needed
    for analysis, ablation, and Pareto frontier visualization.
    """
    # Composite objective (what the optimizer minimizes)
    objective: float = 0.0

    # Individual components
    cost: float = 0.0
    sla_violation_rate: float = 0.0
    latency_variance: float = 0.0

    # Latency percentiles (for analysis, not directly optimized)
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    mean_latency: float = 0.0
    max_latency: float = 0.0

    # Request disposition
    completion_rate: float = 0.0
    drop_rate: float = 0.0
    total_requests: int = 0

    # Penalty decomposition (for ablation)
    sla_penalty: float = 0.0
    variance_penalty: float = 0.0

    # Feasibility
    is_feasible: bool = True  # True if P99 ≤ target

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "cost": self.cost,
            "sla_violation_rate": self.sla_violation_rate,
            "latency_variance": self.latency_variance,
            "p50": self.p50,
            "p90": self.p90,
            "p99": self.p99,
            "p999": self.p999,
            "mean_latency": self.mean_latency,
            "max_latency": self.max_latency,
            "completion_rate": self.completion_rate,
            "drop_rate": self.drop_rate,
            "total_requests": self.total_requests,
            "sla_penalty": self.sla_penalty,
            "variance_penalty": self.variance_penalty,
            "is_feasible": self.is_feasible,
        }


class StabilityMetrics:
    """
    Computes stability-aware objective from simulation results.

    The objective is decomposed into three components:
      1. Cost          — direct infrastructure cost
      2. SLA penalty   — fraction of requests violating P99 target
      3. Variance penalty — latency variance penalizing oscillations

    Parameters
    ----------
    sla_target_ms : float
        P99 latency SLA target in milliseconds.
    lambda_sla : float
        Weight for SLA violation penalty.
    lambda_variance : float
        Weight for latency variance penalty.
    """

    def __init__(
        self,
        sla_target_ms: float = 200.0,
        lambda_sla: float = 10.0,
        lambda_variance: float = 2.0,
    ):
        self.sla_target_ms = sla_target_ms
        self.lambda_sla = lambda_sla
        self.lambda_variance = lambda_variance

    def compute(self, result: SimulationResult) -> ObjectiveValue:
        """
        Compute the full stability-aware objective.

        Parameters
        ----------
        result : SimulationResult
            Output from simulation engine.

        Returns
        -------
        ObjectiveValue with all metrics computed.
        """
        latencies = result.latencies

        # Handle edge case: no valid latencies
        if len(latencies) == 0 or (len(latencies) == 1 and latencies[0] == 0.0):
            return ObjectiveValue(
                objective=1e6,  # Large penalty for degenerate configs
                cost=result.total_cost,
                is_feasible=False,
                total_requests=result.total_requests,
                drop_rate=result.drop_rate,
                completion_rate=result.completion_rate,
            )

        # ── Latency percentiles ──────────────────────────────────
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p99 = float(np.percentile(latencies, 99))
        p999 = float(np.percentile(latencies, 99.9))
        mean_lat = float(np.mean(latencies))
        max_lat = float(np.max(latencies))

        # ── SLA violations ───────────────────────────────────────
        # Fraction of requests exceeding the P99 SLA target
        sla_violations = float(np.mean(latencies > self.sla_target_ms))

        # Also penalize dropped requests (treated as SLA violations)
        drop_penalty = result.drop_rate
        effective_sla_violation = sla_violations + drop_penalty

        # ── Latency variance ─────────────────────────────────────
        # Normalized variance (divide by mean² to make scale-invariant)
        raw_variance = float(np.var(latencies))
        # Coefficient of variation squared — scale-free stability measure
        cv_squared = raw_variance / max(mean_lat ** 2, 1e-8)

        # ── Objective composition ────────────────────────────────
        cost = result.total_cost
        sla_penalty = self.lambda_sla * effective_sla_violation
        variance_penalty = self.lambda_variance * cv_squared

        objective = cost + sla_penalty + variance_penalty

        # ── Feasibility ──────────────────────────────────────────
        is_feasible = p99 <= self.sla_target_ms and result.drop_rate < 0.05

        return ObjectiveValue(
            objective=objective,
            cost=cost,
            sla_violation_rate=effective_sla_violation,
            latency_variance=raw_variance,
            p50=p50,
            p90=p90,
            p99=p99,
            p999=p999,
            mean_latency=mean_lat,
            max_latency=max_lat,
            completion_rate=result.completion_rate,
            drop_rate=result.drop_rate,
            total_requests=result.total_requests,
            sla_penalty=sla_penalty,
            variance_penalty=variance_penalty,
            is_feasible=is_feasible,
        )

    def compute_ablated(
        self,
        result: SimulationResult,
        disable_sla: bool = False,
        disable_variance: bool = False,
    ) -> ObjectiveValue:
        """
        Compute objective with selective ablation of penalty terms.

        Used for ablation studies to measure the contribution of
        each component to overall optimization quality.
        """
        original_sla = self.lambda_sla
        original_var = self.lambda_variance

        if disable_sla:
            self.lambda_sla = 0.0
        if disable_variance:
            self.lambda_variance = 0.0

        result_obj = self.compute(result)

        # Restore
        self.lambda_sla = original_sla
        self.lambda_variance = original_var

        return result_obj

    @staticmethod
    def compute_stability_score(latencies: np.ndarray) -> float:
        """
        Compute a 0-1 stability score (higher = more stable).

        Based on coefficient of variation:
          stability = 1 / (1 + CV)

        Where CV = std/mean.

        A perfectly stable system (all same latency) scores 1.0.
        A highly volatile system approaches 0.0.
        """
        if len(latencies) == 0:
            return 0.0
        mean = np.mean(latencies)
        std = np.std(latencies)
        cv = std / max(mean, 1e-8)
        return 1.0 / (1.0 + cv)
