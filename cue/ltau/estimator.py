from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cue.ltau.descriptor import (
    pdfs_to_expected_error,
    pdfs_to_uncertainty_entropy,
)


@dataclass
class KNNResult:
    """kNN search output."""
    indices: np.ndarray   # (Nq, k)
    distances: np.ndarray # (Nq, k)


class LTAUEstimator:
    """
    LTAU-style estimator:
      - Fit on (train_descriptors, train_pdfs)
      - Predict test pdf as mean of neighbor train pdfs (Algorithm 2)
    """

    def __init__(self, k: int = 10, use_faiss: bool = True):
        """
        Args:
            k (int): Number of nearest neighbors.
            use_faiss (bool): If True, try FAISS; fallback to sklearn if unavailable.
        """
        self.k = int(k)
        self.use_faiss = bool(use_faiss)

        self.train_desc: Optional[np.ndarray] = None
        self.train_pdfs: Optional[np.ndarray] = None
        self.bin_edges: Optional[np.ndarray] = None

        self._faiss_index = None
        self._sk_index = None

    def fit(self, train_desc: np.ndarray, train_pdfs: np.ndarray, bin_edges: np.ndarray) -> None:
        """
        Args:
            train_desc (np.ndarray): (N, D) descriptor vectors (hidden embeddings).
            train_pdfs (np.ndarray): (N, B) density per bin.
            bin_edges (np.ndarray): (B+1,)

        Returns:
            None
        """
        train_desc = np.asarray(train_desc, dtype=np.float32)
        train_pdfs = np.asarray(train_pdfs, dtype=np.float32)
        bin_edges = np.asarray(bin_edges, dtype=float)

        if train_desc.ndim != 2:
            raise ValueError("train_desc must be (N, D)")
        if train_pdfs.ndim != 2:
            raise ValueError("train_pdfs must be (N, B)")
        if train_desc.shape[0] != train_pdfs.shape[0]:
            raise ValueError("train_desc and train_pdfs must share N")
        if bin_edges.ndim != 1 or bin_edges.shape[0] != train_pdfs.shape[1] + 1:
            raise ValueError("bin_edges must be (B+1,)")

        self.train_desc = train_desc
        self.train_pdfs = train_pdfs
        self.bin_edges = bin_edges

        # Build kNN index in descriptor space
        if self.use_faiss:
            try:
                import faiss  # type: ignore

                d = train_desc.shape[1]
                index = faiss.IndexFlatL2(d)
                index.add(train_desc)
                self._faiss_index = index
                self._sk_index = None
                return
            except Exception:
                pass

        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        nn.fit(train_desc)
        self._sk_index = nn
        self._faiss_index = None

    def kneighbors(self, query_desc: np.ndarray, k: Optional[int] = None) -> KNNResult:
        """
        Args:
            query_desc (np.ndarray): (Nq, D)
            k (Optional[int]): If None, use self.k

        Returns:
            KNNResult: indices/distances, both shape (Nq, k)
        """
        if self.train_desc is None:
            raise RuntimeError("Call fit() first.")

        query_desc = np.asarray(query_desc, dtype=np.float32)
        k_eff = int(self.k if k is None else k)

        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(query_desc, k_eff)
            return KNNResult(indices=indices, distances=distances)

        if self._sk_index is None:
            raise RuntimeError("No kNN backend available.")

        distances, indices = self._sk_index.kneighbors(query_desc, n_neighbors=k_eff, return_distance=True)
        return KNNResult(indices=indices, distances=distances)

    def predict_pdfs(self, query_desc: np.ndarray) -> np.ndarray:
        """
        Args:
            query_desc (np.ndarray): (Nq, D)

        Returns:
            np.ndarray: Estimated pdfs for queries, (Nq, B)
        """
        if self.train_pdfs is None:
            raise RuntimeError("Call fit() first.")

        knn = self.kneighbors(query_desc, k=self.k)
        neigh_pdfs = self.train_pdfs[knn.indices]  # (Nq, k, B)
        return neigh_pdfs.mean(axis=1)             # (Nq, B)

    def predict_uncertainty(
        self,
        query_desc: np.ndarray,
        mode: str = "entropy",
        return_pdfs: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Args:
            query_desc (np.ndarray): (Nq, D)
            mode (str): 'entropy' | 'expected_error'
            return_pdfs (bool)

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - u: (Nq,)
                - pdfs: (Nq, B) if return_pdfs else None
        """
        if self.bin_edges is None:
            raise RuntimeError("Call fit() first.")

        pdfs = self.predict_pdfs(query_desc)

        if mode == "entropy":
            u = pdfs_to_uncertainty_entropy(pdfs, self.bin_edges)
        elif mode == "expected_error":
            u = pdfs_to_expected_error(pdfs, self.bin_edges)
        else:
            raise ValueError("mode must be 'entropy' or 'expected_error'")

        return (u, pdfs) if return_pdfs else (u, None)
