"""
InfraMIND v3 — Workload Trace Generator
========================================

Generates synthetic workload traces (arrival rate time-series) from
WorkloadProfile definitions. Supports composition of:
  - Base constant rate
  - Gaussian noise
  - Diurnal (sinusoidal) patterns
  - Burst (spike) injection

Output: np.ndarray of shape (n_timesteps,) representing λ(t) in req/s.
"""

import numpy as np
from typing import Optional
from .profiles import WorkloadProfile


class WorkloadGenerator:
    """Generates synthetic workload arrival rate traces."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        profile: WorkloadProfile,
        duration_s: float = 300.0,
        resolution_s: float = 1.0,
    ) -> np.ndarray:
        """
        Generate a workload trace from a profile.

        Parameters
        ----------
        profile : WorkloadProfile
            Traffic pattern definition.
        duration_s : float
            Total trace duration in seconds.
        resolution_s : float
            Time step between samples in seconds.

        Returns
        -------
        trace : np.ndarray
            Arrival rates λ(t) of shape (n_steps,), in requests/second.
        """
        n_steps = int(duration_s / resolution_s)
        t = np.linspace(0, duration_s, n_steps, endpoint=False)

        # Base rate
        trace = np.full(n_steps, profile.base_rate, dtype=np.float64)

        # Gaussian noise
        if profile.noise_level > 0:
            noise = self.rng.normal(0, profile.noise_level * profile.base_rate, n_steps)
            trace += noise

        # Diurnal component
        if profile.diurnal_amplitude > 0:
            amplitude = profile.diurnal_amplitude * profile.base_rate
            diurnal = amplitude * np.sin(
                2 * np.pi * t / profile.diurnal_period_s + profile.diurnal_phase
            )
            trace += diurnal

        # Burst injection
        if profile.burst_count > 0:
            trace = self._inject_bursts(
                trace,
                t,
                duration_s,
                profile.burst_count,
                profile.burst_magnitude,
                profile.burst_duration_s,
                profile.base_rate,
            )

        # Clamp to non-negative
        trace = np.maximum(trace, 1.0)

        return trace

    def _inject_bursts(
        self,
        trace: np.ndarray,
        t: np.ndarray,
        duration_s: float,
        n_bursts: int,
        magnitude: float,
        burst_duration_s: float,
        base_rate: float,
    ) -> np.ndarray:
        """Inject Poisson-triggered burst events."""
        # Random burst start times (avoiding first/last 10% of trace)
        margin = 0.1 * duration_s
        burst_starts = self.rng.uniform(margin, duration_s - margin, size=n_bursts)

        for start in burst_starts:
            end = start + burst_duration_s
            # Smooth burst envelope (raised cosine)
            mask = (t >= start) & (t <= end)
            if not np.any(mask):
                continue
            t_burst = t[mask]
            # Raised cosine for smooth onset/offset
            phase = (t_burst - start) / burst_duration_s * np.pi
            envelope = 0.5 * (1 - np.cos(2 * phase))
            trace[mask] += magnitude * base_rate * envelope

        return trace

    def generate_batch(
        self,
        profile: WorkloadProfile,
        n_traces: int,
        duration_s: float = 300.0,
        resolution_s: float = 1.0,
    ) -> np.ndarray:
        """
        Generate multiple traces.

        Returns
        -------
        traces : np.ndarray of shape (n_traces, n_steps)
        """
        traces = []
        for _ in range(n_traces):
            traces.append(self.generate(profile, duration_s, resolution_s))
        return np.array(traces)

    def generate_sliding_windows(
        self,
        trace: np.ndarray,
        window_size: int = 30,
        stride: int = 10,
    ) -> np.ndarray:
        """
        Extract sliding windows from a trace for embedding.

        Returns
        -------
        windows : np.ndarray of shape (n_windows, window_size)
        """
        n = len(trace)
        windows = []
        for start in range(0, n - window_size + 1, stride):
            windows.append(trace[start:start + window_size])
        return np.array(windows)
