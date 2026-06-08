import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cue.utils.utils import MyDataset, NumpyDataset
from cue.utils.metric import get_metrics

from typing import Optional, Dict, Any, List
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

class Interface:
    """
    Torch model wrapper with save/load utilities.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Args:
            model (torch.nn.Module)
            device (torch.device)
        """
        self.model = model.to(device)
        self.device = device

    def save_model(self, filepath: str) -> None:
        """
        Args:
            filepath (str)
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """
        Args:
            filepath (str)
        """
        state = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X (np.ndarray): Feature array (N, D)

        Returns:
            np.ndarray: Embedding array (N, hidden_dim)
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, device=self.device, dtype=torch.float32)
            emb = self.model.get_embedding(x)
        return emb.cpu().numpy()


class Trainer(Interface):
    """
    Trainer for modality-specific MLP with LTAU trajectory support.
    """

    def fit(
        self,
        feature_name: str,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        validation_data: Optional[Dict[str, np.ndarray]] = None,
        filepath_ckpt: str = "model_state.pth",
        save_trajectory: bool = False,
        trajectory_dir: Optional[str] = None,
        save_every: int = 1,
        compound_ids_tr: Optional[np.ndarray] = None,
        compound_ids_va: Optional[np.ndarray] = None,
        smiles_tr: Optional[np.ndarray] = None,
        smiles_va: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Args:
            feature_name (str): Modality name, e.g., 'fp', 'img', 'smi'
            X (np.ndarray): Training features (N, D)
            y (np.ndarray): Training labels (N, 1) or (N,)
            batch_size (int): Batch size
            num_epochs (int): Epochs
            learning_rate (float): Learning rate
            validation_data (Optional[Dict[str, np.ndarray]]): {'X':..., 'y':...}
            filepath_ckpt (str): Best checkpoint path
            save_trajectory (bool): Whether to save trajectories
            trajectory_dir (Optional[str]): Directory for trajectories
            save_every (int): Save frequency for trajectories
            compound_ids_tr (Optional[np.ndarray]): Train IDs
            compound_ids_va (Optional[np.ndarray]): Val IDs
            smiles_tr (Optional[np.ndarray]): Train SMILES
            smiles_va (Optional[np.ndarray]): Val SMILES

        Returns:
            Dict[str, List[float]]: Training history
        """
        # Ensure y is (N, 1)
        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

        dataset = MyDataset(X, y_2d)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],       # regression: MSE(train)
            "val_rmse": [],
            "val_r2": [],
            "val_pcc": [],
        }

        # Best selection rule: minimize val_rmse
        best_score = float("inf")

        # trajectory containers (optional)
        if save_trajectory:
            if trajectory_dir is None:
                raise ValueError("trajectory_dir is required when save_trajectory=True")
            if compound_ids_tr is None or smiles_tr is None:
                raise ValueError("compound_ids_tr and smiles_tr are required when save_trajectory=True")
            os.makedirs(trajectory_dir, exist_ok=True)
            train_traj = {cid: [] for cid in compound_ids_tr}
            val_traj = {cid: [] for cid in (compound_ids_va if compound_ids_va is not None else [])}

        epoch_iter = tqdm(
            range(num_epochs),
            leave=True,
            dynamic_ncols=True,
            desc=f"Training {feature_name} MLP",
        )

        for epoch in epoch_iter:
            self.model.train()
            loss_sum = 0.0

            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                out = self.model(inputs)  # regression: pred
                loss = F.mse_loss(out, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())

            loss_avg = loss_sum / max(1, len(loader))
            history["epoch"].append(float(epoch))
            history["loss"].append(float(loss_avg))

            # -------------------------
            # Validation
            # -------------------------
            if validation_data is not None:
                met = self.evaluate(
                    X=validation_data["X"],
                    y=validation_data["y"],
                    batch_size=batch_size,
                )

                history["val_rmse"].append(float(met.get("rmse", 0.0)))
                history["val_r2"].append(float(met.get("r2", 0.0)))
                history["val_pcc"].append(float(met.get("pcc", 0.0)))

                score = float(met.get("rmse", float("inf")))

                if score < best_score:
                    best_score = score
                    self.save_model(filepath_ckpt)
            else:
                # No validation -> keep last checkpoint
                history["val_rmse"].append(0.0)
                history["val_r2"].append(0.0)
                history["val_pcc"].append(0.0)
                self.save_model(filepath_ckpt)

            # -------------------------
            # Trajectory accumulation
            # -------------------------
            if save_trajectory and (epoch % save_every == 0):
                self.model.eval()
                with torch.no_grad():
                    # Train outputs
                    tr_pred = self.model(
                        torch.tensor(X, device=self.device, dtype=torch.float32)
                    ).cpu().numpy().reshape(-1)

                    y_true_tr = y_2d.reshape(-1)
                    tr_err = np.abs(y_true_tr - tr_pred)

                    for cid, e in zip(compound_ids_tr, tr_err):
                        train_traj[cid].append(float(e))

                    # Val outputs
                    if validation_data is not None:
                        if compound_ids_va is None or smiles_va is None:
                            raise ValueError("compound_ids_va and smiles_va are required when validation_data is provided")

                        va_pred = self.model(
                            torch.tensor(validation_data["X"], device=self.device, dtype=torch.float32)
                        ).cpu().numpy().reshape(-1)

                        y_true_va = np.asarray(validation_data["y"]).reshape(-1)
                        va_err = np.abs(y_true_va - va_pred)

                        for cid, e in zip(compound_ids_va, va_err):
                            val_traj[cid].append(float(e))

        # -------------------------
        # Save trajectories
        # -------------------------
        if save_trajectory:
            def _save(path: str, ids: np.ndarray, smiles: np.ndarray, y_true_arr: np.ndarray, traj: Dict[Any, List[float]]) -> None:
                rows = []
                for cid, smi, yv in zip(ids, smiles, np.asarray(y_true_arr).reshape(-1)):
                    rows.append(
                        {
                            "compound_chembl_id": cid,
                            "SMILES": smi,
                            "y_true": float(yv),
                            "error_trajectory": traj[cid],
                        }
                    )
                pd.DataFrame(rows).to_csv(path, index=False)

            _save(
                os.path.join(trajectory_dir, "train_trajectories.csv"),
                compound_ids_tr,
                smiles_tr,
                y_2d,
                train_traj,
            )

            if validation_data is not None:
                _save(
                    os.path.join(trajectory_dir, "val_trajectories.csv"),
                    compound_ids_va,
                    smiles_va,
                    validation_data["y"],
                    val_traj,
                )

        # restore best
        self.load_model(filepath_ckpt)
        return history


    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
    ) -> dict:
        """
        Args:
            X (np.ndarray): features
            y (np.ndarray): labels
            batch_size (int): batch size

        Returns:
            dict: metrics dict from get_metrics
        """
        dataset = MyDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()

        y_true_list: List[np.ndarray] = []
        y_out_list: List[np.ndarray] = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                out = self.model(inputs)  # regression: pred
                y_true_list.append(targets.detach().cpu().numpy())
                y_out_list.append(out.detach().cpu().numpy())

        y_true = np.vstack(y_true_list).reshape(-1)
        y_out = np.vstack(y_out_list).reshape(-1)

        met = get_metrics(y_true=y_true, y_pred=y_out)
        return met


    def inference(
        self,
        X: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Args:
            X (np.ndarray): Input features, shape (N, D)
            batch_size (int): Batch size for inference

        Returns:
            np.ndarray: regression -> predictions, shape (N,)
        """
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        outs = []

        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                logits_or_pred = self.model(inputs)  # (B, 1)
                outs.append(logits_or_pred.cpu().numpy())

        return np.vstack(outs).reshape(-1)



class DownstreamTrainer(Interface):
    """
    Minimal trainer for an MLP downstream model (PyTorch).
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Args:
            model (torch.nn.Module): Downstream model, output shape (B, 1)
            device (torch.device): torch device
        """
        self.device = device
        self.model = model.to(device)

    def fit(
        self,
        feature_name: str,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        validation_data: Optional[Dict[str, np.ndarray]] = None,
        filepath_ckpt: str = "model_state.pth",
        save_every: int = 1,
        save_trajectory: bool = False,
        trajectory_dir: Optional[str] = None,
        compound_ids_tr: Optional[np.ndarray] = None,
        compound_ids_va: Optional[np.ndarray] = None,
        smiles_tr: Optional[np.ndarray] = None,
        smiles_va: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """
        Args:
            feature_name (str): display name, e.g., 'CUE'
            X (np.ndarray): Train features, shape (N, D)
            y (np.ndarray): Train labels, shape (N, 1) or (N,)
            batch_size (int): Batch size
            num_epochs (int): Epoch count
            learning_rate (float): Learning rate
            validation_data (Optional[Dict[str, np.ndarray]]): {'X':..., 'y':...}
            filepath_ckpt (str): checkpoint path
            save_every (int): trajectory save frequency
            save_trajectory (bool): whether to save per-epoch trajectories
            trajectory_dir (Optional[str]): trajectory directory
            compound_ids_tr (Optional[np.ndarray]): train compound ids
            compound_ids_va (Optional[np.ndarray]): val compound ids
            smiles_tr (Optional[np.ndarray]): train smiles
            smiles_va (Optional[np.ndarray]): val smiles

        Returns:
            Dict[str, List[float]]: training history
        """
        # Ensure y is (N, 1)
        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

        dataset = NumpyDataset(X, y_2d)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history = {
            "epoch": [],
            "loss": [],  # train MSE
            "val_rmse": [],
            "val_r2": [],
            "val_pcc": [],
        }

        # Best selection rule: minimize val_rmse
        best_score = float("inf")

        # trajectory containers (optional)
        train_traj: Dict[Any, List[float]] = defaultdict(list)
        val_traj: Dict[Any, List[float]] = defaultdict(list)

        epoch_iter = tqdm(
            range(num_epochs),
            leave=True,
            dynamic_ncols=True,
            desc=f"Training {feature_name} MLPRegressor",
        )

        for epoch in epoch_iter:
            self.model.train()
            loss_sum = 0.0

            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                out = self.model(inputs)  # regression: pred
                loss = F.mse_loss(out, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())

            loss_avg = loss_sum / max(1, len(loader))
            history["epoch"].append(float(epoch))
            history["loss"].append(float(loss_avg))

            # -------------------------
            # Validation
            # -------------------------
            if validation_data is not None:
                met = self.evaluate(
                    X=validation_data["X"],
                    y=validation_data["y"],
                    batch_size=batch_size,
                )

                history["val_rmse"].append(float(met.get("rmse", 0.0)))
                history["val_r2"].append(float(met.get("r2", 0.0)))
                history["val_pcc"].append(float(met.get("pcc", 0.0)))
                score = float(met.get("rmse", float("inf")))

                if score < best_score:
                    best_score = score
                    self.save_model(filepath_ckpt)
            else:
                # no validation: still keep schema consistent
                history["val_rmse"].append(0.0)
                history["val_r2"].append(0.0)
                history["val_pcc"].append(0.0)

                self.save_model(filepath_ckpt)


            # -------------------------
            # Trajectory accumulation
            # -------------------------
            if save_trajectory and (epoch % save_every == 0):
                if trajectory_dir is None:
                    raise ValueError("trajectory_dir is required when save_trajectory=True")
                if compound_ids_tr is None or smiles_tr is None:
                    raise ValueError("compound_ids_tr and smiles_tr are required when save_trajectory=True")

                self.model.eval()
                with torch.no_grad():
                    # Train outputs
                    tr_pred = self.inference(X, batch_size=256).reshape(-1)
                    y_true_tr = y_2d.reshape(-1)
                    tr_err = np.abs(y_true_tr - tr_pred)

                    for cid, e in zip(compound_ids_tr, tr_err):
                        train_traj[cid].append(float(e))

                    # Val outputs
                    if validation_data is not None:
                        if compound_ids_va is None or smiles_va is None:
                            raise ValueError("compound_ids_va and smiles_va are required for val trajectory")

                        va_pred = self.inference(validation_data["X"], batch_size=256).reshape(-1)
                        y_true_va = np.asarray(validation_data["y"]).reshape(-1)
                        va_err = np.abs(y_true_va - va_pred)

                        for cid, e in zip(compound_ids_va, va_err):
                            val_traj[cid].append(float(e))

        # -------------------------
        # Save trajectories
        # -------------------------
        if save_trajectory:
            os.makedirs(trajectory_dir or ".", exist_ok=True)

            def _save(
                path: str,
                ids: np.ndarray,
                smiles: np.ndarray,
                y_true: np.ndarray,
                traj: Dict[Any, List[float]],
            ) -> None:
                rows = []
                for cid, smi, yv in zip(ids, smiles, np.asarray(y_true).reshape(-1)):
                    rows.append(
                        {
                            "compound_chembl_id": cid,
                            "SMILES": smi,
                            "y_true": float(yv),
                            "error_trajectory": traj[cid],
                        }
                    )
                pd.DataFrame(rows).to_csv(path, index=False)

            _save(
                path=os.path.join(trajectory_dir, "train_trajectories.csv"),
                ids=compound_ids_tr,
                smiles=smiles_tr,
                y_true=y_2d,
                traj=train_traj,
            )

            if validation_data is not None:
                _save(
                    path=os.path.join(trajectory_dir, "val_trajectories.csv"),
                    ids=compound_ids_va,
                    smiles=smiles_va,
                    y_true=validation_data["y"],
                    traj=val_traj,
                )

        # restore best
        self.load_model(filepath_ckpt)
        return history


    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Args:
            X (np.ndarray): Input features, shape (N, D)
            y (np.ndarray): Labels, shape (N,) or (N, 1)
            batch_size (int): Batch size

        Returns:
            Dict[str, float]: Metrics dict from get_metrics
        """
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y.reshape(-1, 1) if y.ndim == 1 else y, dtype=torch.float32),
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.model.eval()

        y_true_list = []
        y_out_list = []

        with torch.no_grad():
            for inputs, targets in dl:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                out = self.model(inputs)  # regression: pred
                y_true_list.append(targets.detach().cpu().numpy())
                y_out_list.append(out.detach().cpu().numpy())

        y_true = np.vstack(y_true_list).reshape(-1)
        y_out = np.vstack(y_out_list).reshape(-1)

        met = get_metrics(y_true=y_true, y_pred=y_out)
        return met

    def inference(
        self,
        X: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Args:
            X (np.ndarray): Input features, shape (N, D)
            batch_size (int): Batch size

        Returns:
            np.ndarray: regression: predictions, shape (N,)
        """
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.model.eval()
        outs = []

        with torch.no_grad():
            for (inputs,) in dl:
                inputs = inputs.to(self.device)
                out = self.model(inputs)  # (B, 1)
                outs.append(out.detach().cpu().numpy())

        return np.vstack(outs).reshape(-1)
    
    def plot(
        self,
        history: Dict[str, List[float]],
        save_dir: str,
        best_key: str = "val_rmse",
    ) -> None:
        """
        Args:
            history (Dict[str, List[float]]): training history
            save_dir (str): output directory
            best_key (str): 'val_mse'

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)

        epochs = np.asarray(history.get("epoch", []), dtype=float)
        loss_tr = np.asarray(history.get("loss", []), dtype=float)

        if len(epochs) == 0:
            return

        if best_key not in history:
            best_key = "val_rmse"

        val_curve = np.asarray(history.get(best_key, []), dtype=float)
        if len(val_curve) == 0:
            best_epoch = 0
        else:
            best_epoch = int(np.argmin(val_curve))

        fig = plt.figure(figsize=(8, 4.5))

        sns.lineplot(x=epochs, y=loss_tr, label="train_loss")

        if len(val_curve) > 0:
            sns.lineplot(x=epochs, y=val_curve, label=best_key)

        plt.axvline(best_epoch, linestyle="--", color="red", label="best_epoch")

        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "loss_mlp.png"))
        plt.close(fig)
