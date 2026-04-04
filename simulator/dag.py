"""
InfraMIND v3 — Service DAG Topology (Contribution C5)
======================================================

Defines the directed acyclic graph of microservice dependencies.
Provides topology analysis: adjacency lists, topological ordering,
critical path identification.
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
from config.settings import Settings, ServiceConfig


class ServiceDAG:
    """
    Directed acyclic graph representing microservice dependencies.

    Each node is a service; edges represent call dependencies
    (parent calls child during request processing).
    """

    def __init__(self, settings: Settings):
        """
        Build DAG from settings configuration.

        Parameters
        ----------
        settings : Settings
            Parsed configuration containing service definitions.
        """
        self.settings = settings
        self.services: Dict[str, ServiceConfig] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adj: Dict[str, List[str]] = defaultdict(list)
        self._build(settings)

    def _build(self, settings: Settings):
        """Construct adjacency lists from service configs."""
        for svc in settings.services:
            self.services[svc.name] = svc
            self.adjacency[svc.name] = list(svc.downstream)
            for child in svc.downstream:
                self.reverse_adj[child].append(svc.name)

    @property
    def service_names(self) -> List[str]:
        return list(self.services.keys())

    @property
    def n_services(self) -> int:
        return len(self.services)

    @property
    def entry_points(self) -> List[str]:
        """Services with no incoming edges (DAG roots)."""
        all_children = set()
        for children in self.adjacency.values():
            all_children.update(children)
        return [name for name in self.services if name not in all_children]

    @property
    def leaf_services(self) -> List[str]:
        """Services with no outgoing edges (DAG sinks)."""
        return [name for name, children in self.adjacency.items() if len(children) == 0]

    def get_downstream(self, service: str) -> List[str]:
        """Get immediate downstream dependencies."""
        return self.adjacency.get(service, [])

    def get_upstream(self, service: str) -> List[str]:
        """Get immediate upstream callers."""
        return self.reverse_adj.get(service, [])

    def topological_sort(self) -> List[str]:
        """
        Kahn's algorithm for topological ordering.
        Returns services in dependency-respecting order.
        """
        in_degree = defaultdict(int)
        for name in self.services:
            in_degree[name] = 0
        for name, children in self.adjacency.items():
            for child in children:
                in_degree[child] += 1

        queue = deque([n for n in self.services if in_degree[n] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.adjacency[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.services):
            raise ValueError("DAG contains a cycle!")

        return order

    def critical_path(self) -> Tuple[List[str], float]:
        """
        Find the critical path through the DAG based on base service times.
        Returns (path, total_time_ms).
        """
        topo = self.topological_sort()
        dist = {name: 0.0 for name in self.services}
        pred = {name: None for name in self.services}

        for node in topo:
            node_time = self.services[node].base_service_time
            for child in self.adjacency[node]:
                new_dist = dist[node] + node_time
                if new_dist > dist[child]:
                    dist[child] = new_dist
                    pred[child] = node

        # Find the leaf with maximum distance
        max_leaf = max(self.leaf_services, key=lambda n: dist[n] + self.services[n].base_service_time)
        total = dist[max_leaf] + self.services[max_leaf].base_service_time

        # Reconstruct path
        path = [max_leaf]
        node = max_leaf
        while pred[node] is not None:
            node = pred[node]
            path.append(node)
        path.reverse()

        return path, total

    def all_paths_from(self, source: str) -> List[List[str]]:
        """
        Get all paths from a source service to leaf nodes.
        Used for computing end-to-end latency distributions.
        """
        if source not in self.services:
            raise ValueError(f"Unknown service: {source}")

        paths = []
        self._dfs_paths(source, [source], paths)
        return paths

    def _dfs_paths(self, node: str, current_path: List[str], all_paths: List[List[str]]):
        children = self.adjacency[node]
        if not children:
            all_paths.append(list(current_path))
            return
        for child in children:
            current_path.append(child)
            self._dfs_paths(child, current_path, all_paths)
            current_path.pop()

    def depth(self, service: str) -> int:
        """Compute the depth (longest path from any root to this service)."""
        if service in self.entry_points:
            return 0
        return 1 + max(self.depth(p) for p in self.get_upstream(service))

    def __repr__(self) -> str:
        lines = ["ServiceDAG:"]
        for name, children in self.adjacency.items():
            if children:
                lines.append(f"  {name} → {', '.join(children)}")
            else:
                lines.append(f"  {name} (leaf)")
        return "\n".join(lines)
