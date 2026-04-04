"""
InfraMIND v3 — Service Clusterer (Contribution C2, Part 2)
===========================================================

Clusters services based on their sensitivity profiles to enable
hierarchical parameter sharing. Services with similar sensitivity
patterns should share configurations, reducing the effective
optimization dimensionality.

Methods:
  - Spectral Clustering: Uses RBF affinity on sensitivity vectors
  - K-Means: Direct clustering on normalized sensitivity vectors
  - Auto k: Selects optimal k via silhouette score

Result: groups of services that share parameters in the optimizer.

Complexity:
  - Spectral: O(n³) for eigendecomposition (n = n_services, typically small)
  - K-Means: O(n × k × iterations)
  - Silhouette scan: O(max_k × clustering_cost)
"""

import numpy as np
import logging
from typing import List, Set, Dict, Optional, Tuple
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

logger = logging.getLogger("inframind.structure_learning")


class ServiceClusterer:
    """
    Clusters services based on sensitivity profiles for parameter sharing.

    When services have similar sensitivity patterns (i.e., they respond
    similarly to parameter changes), they should share configurations.
    This dramatically reduces the optimization dimensionality:

        Full dim:  n_services × n_params  (e.g., 7 × 4 = 28)
        Reduced:   n_clusters × n_params  (e.g., 3 × 4 = 12)

    This reduction is critical for sample-efficient Bayesian optimization.
    """

    def __init__(
        self,
        service_names: List[str],
        method: str = "spectral",
        min_clusters: int = 2,
        max_clusters: int = 5,
    ):
        """
        Parameters
        ----------
        service_names : list of str
            Names of all services in the DAG.
        method : str
            Clustering method: 'spectral' or 'kmeans'.
        min_clusters, max_clusters : int
            Range for automatic k selection.
        """
        self.service_names = service_names
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def cluster(
        self,
        sensitivity_matrix: np.ndarray,
        n_clusters: Optional[int] = None,
        seed: int = 42,
    ) -> List[Set[str]]:
        """
        Cluster services based on sensitivity profiles.

        Parameters
        ----------
        sensitivity_matrix : np.ndarray, shape (n_services, n_params)
            Sensitivity matrix from SensitivityAnalyzer.
        n_clusters : int or None
            Number of clusters. None = auto-detect.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        clusters : list of sets
            Each set contains service names that should share parameters.
        """
        n_services = len(self.service_names)

        # Trivial case: fewer services than min_clusters
        if n_services <= self.min_clusters:
            return [set(self.service_names)]

        # Normalize sensitivity vectors
        scaler = StandardScaler()
        S_normalized = scaler.fit_transform(sensitivity_matrix)

        # Determine k
        if n_clusters is None:
            n_clusters = self._auto_select_k(S_normalized, seed)
            logger.info(f"Auto-selected k={n_clusters} clusters")

        # Clamp k
        n_clusters = min(n_clusters, n_services)
        n_clusters = max(n_clusters, 1)

        # Cluster
        if self.method == "spectral":
            labels = self._spectral_cluster(S_normalized, n_clusters, seed)
        elif self.method == "kmeans":
            labels = self._kmeans_cluster(S_normalized, n_clusters, seed)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Convert labels to sets
        clusters = self._labels_to_clusters(labels)

        logger.info(f"Service clusters ({self.method}, k={n_clusters}):")
        for i, cluster in enumerate(clusters):
            logger.info(f"  Cluster {i}: {cluster}")

        return clusters

    def _spectral_cluster(
        self,
        S: np.ndarray,
        k: int,
        seed: int,
    ) -> np.ndarray:
        """Spectral clustering with RBF affinity."""
        model = SpectralClustering(
            n_clusters=k,
            affinity="rbf",
            gamma=1.0,
            random_state=seed,
            assign_labels="kmeans",
        )
        return model.fit_predict(S)

    def _kmeans_cluster(
        self,
        S: np.ndarray,
        k: int,
        seed: int,
    ) -> np.ndarray:
        """K-Means clustering on normalized sensitivity vectors."""
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=10,
        )
        return model.fit_predict(S)

    def _auto_select_k(
        self,
        S: np.ndarray,
        seed: int,
    ) -> int:
        """
        Automatically select optimal number of clusters
        via silhouette score maximization.
        """
        n_services = S.shape[0]
        max_k = min(self.max_clusters, n_services - 1)
        min_k = self.min_clusters

        if max_k < min_k:
            return min_k

        best_k = min_k
        best_score = -1.0

        for k in range(min_k, max_k + 1):
            try:
                if self.method == "spectral":
                    labels = self._spectral_cluster(S, k, seed)
                else:
                    labels = self._kmeans_cluster(S, k, seed)

                # Silhouette score requires at least 2 unique labels
                if len(set(labels)) < 2:
                    continue

                score = silhouette_score(S, labels)
                logger.debug(f"  k={k}: silhouette={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.warning(f"  k={k}: clustering failed: {e}")
                continue

        return best_k

    def _labels_to_clusters(self, labels: np.ndarray) -> List[Set[str]]:
        """Convert label array to list of service name sets."""
        clusters_dict: Dict[int, Set[str]] = {}
        for i, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = set()
            clusters_dict[label].add(self.service_names[i])

        # Sort by cluster label for deterministic ordering
        return [clusters_dict[k] for k in sorted(clusters_dict.keys())]

    def get_cluster_report(
        self,
        clusters: List[Set[str]],
        sensitivity_matrix: np.ndarray,
    ) -> Dict:
        """Generate a report on cluster quality."""
        report = {
            "n_clusters": len(clusters),
            "method": self.method,
            "clusters": [],
            "dimensionality_reduction": {
                "original": len(self.service_names),
                "reduced": len(clusters),
                "reduction_ratio": len(clusters) / max(len(self.service_names), 1),
            },
        }

        for i, cluster in enumerate(clusters):
            svc_indices = [self.service_names.index(s) for s in cluster]
            cluster_S = sensitivity_matrix[svc_indices]

            report["clusters"].append({
                "id": i,
                "services": sorted(cluster),
                "size": len(cluster),
                "mean_sensitivity": float(np.mean(np.abs(cluster_S))),
                "max_sensitivity": float(np.max(np.abs(cluster_S))),
                "intra_cluster_variance": float(np.var(cluster_S)),
            })

        return report
