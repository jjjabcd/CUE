from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from cue.ltau.descriptor import pdfs_to_uncertainty_entropy


@dataclass
class KNNResult:
    """kNN search output."""
    neighbor_indices: np.ndarray   # (Nq, k)
    distances: np.ndarray # (Nq, k)


class LTAUEstimator:
    """
    LTAU-style estimator:
      - Fit on (ref_emb, reference_pdfs)
      - Predict uncertainty as entropy of the mean neighbor PDF (Algorithm 2)
    """

    def __init__(self, k: int = 10):
        """
        Args:
            k (int): Number of nearest neighbors.
        """
        self.k = int(k)

        self.ref_emb: Optional[np.ndarray] = None
        self.reference_pdfs: Optional[np.ndarray] = None
        self.shared_bin_edges: Optional[np.ndarray] = None

        self._sk_index: Optional[NearestNeighbors] = None

    def fit(self, ref_emb: np.ndarray, reference_pdfs: np.ndarray, shared_bin_edges: np.ndarray) -> None:
        """
        Args:
            ref_emb (np.ndarray): (N, D) reference embedding vectors (hidden embeddings).
            reference_pdfs (np.ndarray): (N, B) density per bin.
            shared_bin_edges (np.ndarray): (B+1,)
        """
        ref_emb = np.asarray(ref_emb, dtype=np.float32)
        reference_pdfs = np.asarray(reference_pdfs, dtype=np.float32)
        shared_bin_edges = np.asarray(shared_bin_edges, dtype=float)

        if ref_emb.ndim != 2:
            raise ValueError("ref_emb must be (N, D)")
        if reference_pdfs.ndim != 2:
            raise ValueError("reference_pdfs must be (N, B)")
        if ref_emb.shape[0] != reference_pdfs.shape[0]:
            raise ValueError("ref_emb and reference_pdfs must share N")
        if shared_bin_edges.ndim != 1 or shared_bin_edges.shape[0] != reference_pdfs.shape[1] + 1:
            raise ValueError("shared_bin_edges must be (B+1,)")

        self.ref_emb = ref_emb
        self.reference_pdfs = reference_pdfs
        self.shared_bin_edges = shared_bin_edges

        nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        nn.fit(ref_emb)
        self._sk_index = nn

    def kneighbors(self, query_desc: np.ndarray, k: Optional[int] = None) -> KNNResult:
        """
        Args:
            query_desc (np.ndarray): (Nq, D)
            k (Optional[int]): If None, use self.k

        Returns:
            KNNResult: indices/distances, both shape (Nq, k)
        """
        if self._sk_index is None:
            raise RuntimeError("Call fit() first.")

        query_desc = np.asarray(query_desc, dtype=np.float32)
        k_eff = int(self.k if k is None else k)

        distances, neighbor_indices = self._sk_index.kneighbors(query_desc, n_neighbors=k_eff, return_distance=True)
        return KNNResult(neighbor_indices=neighbor_indices, distances=distances)

    def predict_pdfs(self, query_desc: np.ndarray) -> np.ndarray:
        """
        Args:
            query_desc (np.ndarray): (Nq, D)

        Returns:
            np.ndarray: Estimated query_pdf for queries, (Nq, B)
        """
        if self.reference_pdfs is None:
            raise RuntimeError("Call fit() first.")

        knn = self.kneighbors(query_desc, k=self.k)
        neighbor_ref_pdf = self.reference_pdfs[knn.neighbor_indices]  # (Nq, k, B)
        return neighbor_ref_pdf.mean(axis=1)             # (Nq, B)

    def predict_uncertainty(self, query_desc: np.ndarray) -> np.ndarray:
        """
        Args:
            query_desc (np.ndarray): (Nq, D)

        Returns:
            np.ndarray: Normalized entropy uncertainty, shape (Nq,)
        """
        if self.shared_bin_edges is None:
            raise RuntimeError("Call fit() first.")

        query_pdf = self.predict_pdfs(query_desc)
        return pdfs_to_uncertainty_entropy(query_pdf, self.shared_bin_edges)
