"""
InfraMIND v3 — Global Settings & Configuration Loader
=====================================================

Loads default_config.yaml and provides typed access to all settings.
"""

import os
import yaml
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("inframind")


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------
@dataclass
class TuRBOSettings:
    length_init: float = 0.8
    length_min: float = 0.005
    length_max: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int = 5
    volatility_alpha: float = 1.5


@dataclass
class OptimizationSettings:
    n_initial: int = 20
    n_iterations: int = 50
    batch_size: int = 4
    seed: int = 42
    turbo: TuRBOSettings = field(default_factory=TuRBOSettings)


@dataclass
class ObjectiveSettings:
    sla_target_p99_ms: float = 200.0
    lambda_sla: float = 10.0
    lambda_variance: float = 2.0
    cost_per_replica_per_hour: float = 0.05


@dataclass
class SimulationSettings:
    duration_s: float = 300.0
    warmup_s: float = 30.0
    resolution_s: float = 1.0


@dataclass
class WorkloadSettings:
    base_rate: float = 100.0
    noise_level: float = 0.05
    diurnal_amplitude: float = 0.6
    diurnal_period_s: float = 86400.0
    burst_count: int = 5
    burst_magnitude: float = 3.0
    burst_duration_s: float = 10.0


@dataclass
class StructureLearningSettings:
    method: str = "spectral"
    n_clusters: Optional[int] = None
    sensitivity_delta: float = 0.05
    min_clusters: int = 2
    max_clusters: int = 5


@dataclass
class ServiceConfig:
    name: str = ""
    base_service_time: float = 10.0
    base_replicas: int = 2
    downstream: List[str] = field(default_factory=list)


@dataclass
class ParameterBound:
    min: float = 0.0
    max: float = 1.0
    type: str = "int"


@dataclass
class Settings:
    """Master settings container — loaded from YAML."""
    services: List[ServiceConfig] = field(default_factory=list)
    per_service_params: Dict[str, ParameterBound] = field(default_factory=dict)
    global_params: Dict[str, ParameterBound] = field(default_factory=dict)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    objectives: ObjectiveSettings = field(default_factory=ObjectiveSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    workload: WorkloadSettings = field(default_factory=WorkloadSettings)
    structure_learning: StructureLearningSettings = field(default_factory=StructureLearningSettings)

    @property
    def service_names(self) -> List[str]:
        return [s.name for s in self.services]

    @property
    def n_services(self) -> int:
        return len(self.services)

    @property
    def n_per_service_params(self) -> int:
        return len(self.per_service_params)

    @property
    def flat_dim(self) -> int:
        """Total dimensionality without structure learning."""
        return self.n_services * self.n_per_service_params + len(self.global_params)


def _parse_settings(raw: Dict[str, Any]) -> Settings:
    """Parse raw YAML dict into typed Settings."""
    services = []
    for svc in raw.get("dag", {}).get("services", []):
        services.append(ServiceConfig(
            name=svc["name"],
            base_service_time=float(svc["base_service_time"]),
            base_replicas=int(svc["base_replicas"]),
            downstream=svc.get("downstream", []),
        ))

    per_service_params = {}
    for pname, pdef in raw.get("parameters", {}).get("per_service", {}).items():
        per_service_params[pname] = ParameterBound(
            min=float(pdef["min"]),
            max=float(pdef["max"]),
            type=pdef.get("type", "int"),
        )

    global_params = {}
    for pname, pdef in raw.get("parameters", {}).get("global", {}).items():
        global_params[pname] = ParameterBound(
            min=float(pdef["min"]),
            max=float(pdef["max"]),
            type=pdef.get("type", "int"),
        )

    opt_raw = raw.get("optimization", {})
    turbo_raw = opt_raw.get("turbo", {})
    turbo = TuRBOSettings(
        length_init=turbo_raw.get("length_init", 0.8),
        length_min=turbo_raw.get("length_min", 0.005),
        length_max=turbo_raw.get("length_max", 1.6),
        success_tolerance=turbo_raw.get("success_tolerance", 3),
        failure_tolerance=turbo_raw.get("failure_tolerance", 5),
        volatility_alpha=turbo_raw.get("volatility_alpha", 1.5),
    )
    optimization = OptimizationSettings(
        n_initial=opt_raw.get("n_initial", 20),
        n_iterations=opt_raw.get("n_iterations", 50),
        batch_size=opt_raw.get("batch_size", 4),
        seed=opt_raw.get("seed", 42),
        turbo=turbo,
    )

    obj_raw = raw.get("objectives", {})
    objectives = ObjectiveSettings(
        sla_target_p99_ms=obj_raw.get("sla_target_p99_ms", 200.0),
        lambda_sla=obj_raw.get("lambda_sla", 10.0),
        lambda_variance=obj_raw.get("lambda_variance", 2.0),
        cost_per_replica_per_hour=obj_raw.get("cost_per_replica_per_hour", 0.05),
    )

    sim_raw = raw.get("simulation", {})
    simulation = SimulationSettings(
        duration_s=sim_raw.get("duration_s", 300.0),
        warmup_s=sim_raw.get("warmup_s", 30.0),
        resolution_s=sim_raw.get("resolution_s", 1.0),
    )

    wl_raw = raw.get("workload", {})
    workload_settings = WorkloadSettings(
        base_rate=wl_raw.get("base_rate", 100.0),
        noise_level=wl_raw.get("noise_level", 0.05),
        diurnal_amplitude=wl_raw.get("diurnal_amplitude", 0.6),
        diurnal_period_s=wl_raw.get("diurnal_period_s", 86400.0),
        burst_count=wl_raw.get("burst_count", 5),
        burst_magnitude=wl_raw.get("burst_magnitude", 3.0),
        burst_duration_s=wl_raw.get("burst_duration_s", 10.0),
    )

    sl_raw = raw.get("structure_learning", {})
    structure_learning = StructureLearningSettings(
        method=sl_raw.get("method", "spectral"),
        n_clusters=sl_raw.get("n_clusters", None),
        sensitivity_delta=sl_raw.get("sensitivity_delta", 0.05),
        min_clusters=sl_raw.get("min_clusters", 2),
        max_clusters=sl_raw.get("max_clusters", 5),
    )

    return Settings(
        services=services,
        per_service_params=per_service_params,
        global_params=global_params,
        optimization=optimization,
        objectives=objectives,
        simulation=simulation,
        workload=workload_settings,
        structure_learning=structure_learning,
    )


def get_default_config(config_path: Optional[str] = None) -> Settings:
    """Load and parse the default configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning(f"Config not found at {path}, using defaults")
        return Settings()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _parse_settings(raw)


def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
