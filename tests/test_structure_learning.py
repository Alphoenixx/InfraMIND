"""
InfraMIND v3 — Tests for Structure Learning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from structure_learning.cluster import ServiceClusterer


class TestServiceClusterer:

    def test_two_clear_clusters(self):
        """Services with very different sensitivity profiles should cluster apart."""
        service_names = ["svc_a", "svc_b", "svc_c", "svc_d"]
        # Two groups: (a,b) with high sensitivity on param 0, (c,d) on param 1
        S = np.array([
            [10.0, 0.1],   # svc_a
            [9.5, 0.2],    # svc_b
            [0.1, 10.0],   # svc_c
            [0.2, 9.0],    # svc_d
        ])

        clusterer = ServiceClusterer(service_names, method="kmeans", min_clusters=2, max_clusters=3)
        clusters = clusterer.cluster(S, n_clusters=2)

        assert len(clusters) == 2
        # a,b should be in same cluster; c,d in same cluster
        for cluster in clusters:
            if "svc_a" in cluster:
                assert "svc_b" in cluster
            if "svc_c" in cluster:
                assert "svc_d" in cluster

    def test_auto_k_selection(self):
        service_names = ["s1", "s2", "s3", "s4", "s5", "s6"]
        # Three clear groups
        S = np.array([
            [10, 0, 0],
            [9,  0, 0],
            [0, 10, 0],
            [0,  9, 0],
            [0,  0, 10],
            [0,  0, 9],
        ], dtype=float)

        clusterer = ServiceClusterer(service_names, method="kmeans", min_clusters=2, max_clusters=5)
        clusters = clusterer.cluster(S, n_clusters=None)  # Auto
        assert 2 <= len(clusters) <= 4  # Should find ~3 clusters

    def test_single_service(self):
        clusterer = ServiceClusterer(["only_one"], method="kmeans")
        S = np.array([[5.0, 3.0]])
        clusters = clusterer.cluster(S, n_clusters=1)
        assert len(clusters) == 1
        assert "only_one" in clusters[0]

    def test_cluster_report(self):
        service_names = ["a", "b", "c"]
        S = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 0.5]])
        clusterer = ServiceClusterer(service_names, method="kmeans")
        clusters = clusterer.cluster(S, n_clusters=2)
        report = clusterer.get_cluster_report(clusters, S)
        
        assert "n_clusters" in report
        assert "dimensionality_reduction" in report
        assert report["dimensionality_reduction"]["original"] == 3
        assert report["dimensionality_reduction"]["reduced"] == 2
