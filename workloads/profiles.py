"""
InfraMIND v3 — Workload Profiles
=================================

Defines three canonical workload profiles:
  1. Steady   — constant rate + Gaussian noise
  2. Diurnal  — sinusoidal 24h pattern
  3. Bursty   — Poisson-triggered spike events
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkloadProfile:
    """Definition of a workload traffic pattern."""
    name: str
    base_rate: float = 100.0
    noise_level: float = 0.05

    # Diurnal parameters
    diurnal_amplitude: float = 0.0      # 0 = no diurnal component
    diurnal_period_s: float = 86400.0   # 24 hours

    # Burst parameters
    burst_count: int = 0                # 0 = no bursts
    burst_magnitude: float = 3.0        # multiplier over base_rate
    burst_duration_s: float = 10.0      # seconds per burst event

    # Phase shift for diurnal (simulate different times of day)
    diurnal_phase: float = 0.0


# -----------------------------------------------------------------------
# Pre-defined canonical profiles
# -----------------------------------------------------------------------

STEADY = WorkloadProfile(
    name="steady",
    base_rate=100.0,
    noise_level=0.05,
    diurnal_amplitude=0.0,
    burst_count=0,
)

DIURNAL = WorkloadProfile(
    name="diurnal",
    base_rate=100.0,
    noise_level=0.05,
    diurnal_amplitude=0.6,
    diurnal_period_s=300.0,   # Compressed to 300s for simulation
    burst_count=0,
)

BURSTY = WorkloadProfile(
    name="bursty",
    base_rate=100.0,
    noise_level=0.08,
    diurnal_amplitude=0.0,
    burst_count=8,
    burst_magnitude=4.0,
    burst_duration_s=8.0,
)


# Convenience list
ALL_PROFILES = [STEADY, DIURNAL, BURSTY]

PROFILE_MAP = {p.name: p for p in ALL_PROFILES}


def get_profile(name: str) -> WorkloadProfile:
    """Retrieve a profile by name."""
    if name not in PROFILE_MAP:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILE_MAP.keys())}")
    return PROFILE_MAP[name]
