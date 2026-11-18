import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Torch Dataset wrapper with dtype safety.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        do_normalize: bool = True,
    ):
        """
        Args:
            data (np.ndarray): Feature array (N, D)
            labels (np.ndarray): Label array (N, 1)
            do_normalize (bool)
        """
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

class NumpyRegressionDataset(Dataset):
    """
    Simple torch Dataset for regression with numpy inputs.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X (np.ndarray): Feature array, shape (N, D)
            y (np.ndarray): Label array, shape (N, 1) or (N,)
        """
        self.X = X.astype(np.float32)
        y_ = y.reshape(-1, 1) if y.ndim == 1 else y
        self.y = y_.astype(np.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]