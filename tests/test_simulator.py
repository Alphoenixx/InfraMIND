"""
InfraMIND v3 — Tests for DAG Simulator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from config.settings import get_default_config
from simulator.dag import ServiceDAG
from simulator.engine import SimulationEngine
from simulator.request import Request


class TestServiceDAG:

    def setup_method(self):
        self.settings = get_default_config()
        self.dag = ServiceDAG(self.settings)

    def test_service_count(self):
        assert self.dag.n_services == 7

    def test_entry_points(self):
        entries = self.dag.entry_points
        assert "api_gateway" in entries

    def test_leaf_services(self):
        leaves = self.dag.leaf_services
        assert "user_db" in leaves
        assert "product_db" in leaves
        assert "ml_engine" in leaves

    def test_topological_sort(self):
        topo = self.dag.topological_sort()
        assert len(topo) == 7
        # api_gateway should come before auth
        assert topo.index("api_gateway") < topo.index("auth")
        # catalog should come before product_db
        assert topo.index("catalog") < topo.index("product_db")

    def test_critical_path(self):
        path, total = self.dag.critical_path()
        assert len(path) >= 2
        assert total > 0

    def test_downstream(self):
        ds = self.dag.get_downstream("api_gateway")
        assert "auth" in ds
        assert "catalog" in ds

    def test_upstream(self):
        us = self.dag.get_upstream("auth")
        assert "api_gateway" in us

    def test_all_paths(self):
        paths = self.dag.all_paths_from("api_gateway")
        assert len(paths) >= 2  # At least through auth and catalog


class TestRequest:

    def test_end_to_end_latency(self):
        req = Request(request_id=1, created_at=0.0)
        req.mark_completed(0.15)
        assert abs(req.end_to_end_latency - 150.0) < 0.01  # 150ms

    def test_dropped_request(self):
        req = Request(request_id=2, created_at=0.0)
        req.mark_dropped("auth", 0.01)
        assert req.dropped is True
        assert req.drop_service == "auth"

    def test_hop_recording(self):
        req = Request(request_id=3, created_at=0.0)
        req.record_hop("api_gateway", 5.0)
        req.record_hop("auth", 10.0)
        assert len(req.path) == 2
        assert req.total_service_time == 15.0


class TestSimulationEngine:

    def setup_method(self):
        self.settings = get_default_config()
        self.dag = ServiceDAG(self.settings)
        self.engine = SimulationEngine(self.dag, self.settings)

    def test_basic_simulation(self):
        # Short simulation
        trace = np.full(50, 20.0)  # 20 req/s for 50 seconds
        config = {}
        for svc in self.dag.services.values():
            config[svc.name] = {
                "replicas": 4,
                "cpu_millicores": 1000,
                "queue_capacity": 200,
            }
        config["_global"] = {"connection_pool_size": 50}

        result = self.engine.run(trace, config, seed=42)

        assert result.total_requests > 0
        assert result.completed_requests >= 0
        assert len(result.latencies) > 0
        assert result.total_cost > 0

    def test_cost_increases_with_replicas(self):
        trace = np.full(30, 10.0)

        # Low replicas
        config_low = {}
        for svc in self.dag.services.values():
            config_low[svc.name] = {"replicas": 1, "cpu_millicores": 500, "queue_capacity": 50}
        config_low["_global"] = {"connection_pool_size": 10}

        # High replicas
        config_high = {}
        for svc in self.dag.services.values():
            config_high[svc.name] = {"replicas": 8, "cpu_millicores": 2000, "queue_capacity": 200}
        config_high["_global"] = {"connection_pool_size": 50}

        result_low = self.engine.run(trace, config_low, seed=42)
        result_high = self.engine.run(trace, config_high, seed=42)

        assert result_high.total_cost > result_low.total_cost
