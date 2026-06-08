from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from .fusion import compute_fusion_weights
from .train_mlp import MLPTrainConfigs, train_mlp
from .mlp import MLP
from .trainer import Trainer
from cue.utils.FeatureExtraction import MolEncoder

@dataclass
class CUEConfigs:
    """
    Args:
        fold_dir (str): Fold output directory
        device (torch.device): Torch device
        hidden_dim (int): MLP hidden dim
        num_epochs (int): MLP pretrain epochs
        learning_rate (float): MLP pretrain lr
        batch_size (int): MLP pretrain batch size
        save_every (int): Trajectory saving frequency
    """
    fold_dir: str
    device: torch.device
    hidden_dim: int = 32
    num_epochs: int = 2000
    learning_rate: float = 1e-4
    batch_size: int = 32
    save_every: int = 1


class CUE:
    """
    CUE = modality-specific MLP pretraining + uncertainty-aware fusion (in predict).
    """

    def __init__(self, cfg: CUEConfigs):
        """
        Args:
            cfg (CUEConfigs): CUE configuration
        """
        self.cfg = cfg
        self.device = cfg.device
        self.fold_dir = cfg.fold_dir

        os.makedirs(self.fold_dir, exist_ok=True)
        os.makedirs(os.path.join(self.fold_dir, "checkpoints"), exist_ok=True)

        # Encoder (features)
        self.encoder = MolEncoder(self.device)

        # Trainers (MLP per modality)
        self.mlp_fp = Trainer(MLP(self.encoder.feature_dim_fp, hidden_dim=cfg.hidden_dim), self.device)
        self.mlp_img = Trainer(MLP(self.encoder.feature_dim_img, hidden_dim=cfg.hidden_dim), self.device)

        # Scalers (per modality)
        self.scaler_fp = MinMaxScaler()
        self.scaler_img = MinMaxScaler()

    def fit(
        self,
        tr_id: np.ndarray,
        va_id: np.ndarray,
        X_tr: np.ndarray,
        X_va: np.ndarray,
        y_tr: np.ndarray,
        y_va: np.ndarray,
    ) -> Dict[str, float]:
        """
        Args:
            tr_id (np.ndarray): Train IDs
            va_id (np.ndarray): Val IDs
            X_tr (np.ndarray): Train SMILES
            X_va (np.ndarray): Val SMILES
            y_tr (np.ndarray): Train labels (UNNORMALIZED), shape (N_tr, 1)
            y_va (np.ndarray): Val labels (UNNORMALIZED), shape (N_va, 1)

        Returns:
            Dict[str, float]: best validation rmse per modality
        """
        # fp
        cfg_fp = MLPTrainConfigs(
            feature_name="fp",
            trainer=self.mlp_fp,
            scaler=self.scaler_fp,
            get_features=self.encoder.get_fpt_features,
            fold_dir=self.fold_dir,
            num_epochs=self.cfg.num_epochs,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            save_every=self.cfg.save_every,
        )

        loss_fp = train_mlp(tr_id, va_id, X_tr, X_va, y_tr, y_va, cfg_fp)

        # img
        cfg_img = MLPTrainConfigs(
            feature_name="img",
            trainer=self.mlp_img,
            scaler=self.scaler_img,
            get_features=self.encoder.get_img_features,
            fold_dir=self.fold_dir,
            num_epochs=self.cfg.num_epochs,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
            save_every=self.cfg.save_every,
        )

        loss_img = train_mlp(tr_id, va_id, X_tr, X_va, y_tr, y_va, cfg_img)

        return {"fp": loss_fp, "img": loss_img}

    def predict(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        uncertainties: Dict[str, np.ndarray] = None,
        temperature: float = 1.0,
        return_weights: bool = False,
        return_losses: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]] | np.ndarray:
        """
        Args:
            X (np.ndarray): SMILES array, shape (N,)
            y (Optional[np.ndarray]): Labels array, shape (N,) or (N,1)
            uncertainties (Dict[str, np.ndarray]):
                {'fp': (N,), 'img': (N,)}
            temperature (float)
            return_weights (bool)
            return_losses (bool)

        Returns:
            np.ndarray or tuple:
                - unified embedding (N, D)
                - weights (N, 2) if return_weights=True
                - losses (Dict[str, np.ndarray]) if return_losses=True (sample-wise)
        """
        # raw features
        raw_fp = np.array([self.encoder.get_fpt_features(s) for s in X], dtype=np.float32)
        raw_img = np.array([self.encoder.get_img_features(s) for s in X], dtype=np.float32)

        # scaling
        raw_fp = self.scaler_fp.transform(raw_fp)
        raw_img = self.scaler_img.transform(raw_img)

        # modality embeddings (hidden layer outputs)
        emb_fp = self.mlp_fp.get_embeddings(raw_fp)
        emb_img = self.mlp_img.get_embeddings(raw_img)

        loss_dict: Optional[Dict[str, np.ndarray]] = None

        if return_losses:
            n = int(len(X))

            if y is None:
                loss_fp = np.full(n, np.nan, dtype=float)
                loss_img = np.full(n, np.nan, dtype=float)

            else:
                y_flat = np.asarray(y, dtype=float).reshape(-1)

                # regression: absolute error per sample (uses prediction)
                p_fp = self.mlp_fp.inference(raw_fp, batch_size=1024).reshape(-1)
                p_img = self.mlp_img.inference(raw_img, batch_size=1024).reshape(-1)

                loss_fp = np.abs(y_flat - p_fp)
                loss_img = np.abs(y_flat - p_img)

            loss_dict = {
                "loss_fp": loss_fp.astype(float),
                "loss_img": loss_img.astype(float),
            }

        # uncertainty-aware fusion
        u = np.stack([uncertainties["fp"], uncertainties["img"]], axis=1)
        W = compute_fusion_weights(
            u=u,
            temperature=temperature,
        )
        uni = W[:, [0]] * emb_fp + W[:, [1]] * emb_img

        outputs = [uni]
        if return_weights:
            outputs.append(W)
        if return_losses:
            outputs.append(loss_dict)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def save(self) -> None:
        """
        Args:
            None

        Returns:
            None
        """
        ckpt_dir = os.path.join(self.fold_dir, "checkpoints")

        self.mlp_fp.save_model(os.path.join(ckpt_dir, "mlp_fp.pth"))
        joblib.dump(self.scaler_fp, os.path.join(ckpt_dir, "scaler_fp.pkl"))

        self.mlp_img.save_model(os.path.join(ckpt_dir, "mlp_img.pth"))
        joblib.dump(self.scaler_img, os.path.join(ckpt_dir, "scaler_img.pkl"))

    def load(self) -> None:
        """
        Args:
            None

        Returns:
            None
        """
        ckpt_dir = os.path.join(self.fold_dir, "checkpoints")

        self.mlp_fp.load_model(os.path.join(ckpt_dir, "mlp_fp.pth"))
        self.scaler_fp = joblib.load(os.path.join(ckpt_dir, "scaler_fp.pkl"))

        self.mlp_img.load_model(os.path.join(ckpt_dir, "mlp_img.pth"))
        self.scaler_img = joblib.load(os.path.join(ckpt_dir, "scaler_img.pkl"))

    def get_modality_descriptors(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extract modality-specific descriptors (hidden layer outputs).

        Args:
            X (np.ndarray): SMILES array, shape (N,)

        Returns:
            Dict[str, np.ndarray]:
                {
                    "fp":  (N, D),
                    "img": (N, D),
                }
        """

        # --------------------------------------------------
        # Raw feature extraction
        # --------------------------------------------------
        raw_fp = np.array(
            [self.encoder.get_fpt_features(s) for s in X],
            dtype=np.float32,
        )
        raw_img = np.array(
            [self.encoder.get_img_features(s) for s in X],
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Scaling (same as training)
        # --------------------------------------------------
        raw_fp = self.scaler_fp.transform(raw_fp)
        raw_img = self.scaler_img.transform(raw_img)

        # --------------------------------------------------
        # Hidden descriptors (modality embeddings)
        # --------------------------------------------------
        d_fp = self.mlp_fp.get_embeddings(raw_fp)
        d_img = self.mlp_img.get_embeddings(raw_img)

        return {
            "fp": d_fp,
            "img": d_img,
        }
