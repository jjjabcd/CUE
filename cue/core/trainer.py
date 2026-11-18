import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cue.data.dataset import MyDataset, NumpyRegressionDataset
from cue.utils.metric import get_metrics
from cue.data.normalization import LabelNormalizer
from cue.core.evidential import evidential_loss, nig_uncertainty

from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def _enable_dropout_only(model: torch.nn.Module) -> None:
    """
    Switch only ``nn.Dropout`` submodules of a model into training mode,
    leaving every other submodule in evaluation mode. Required for MC Dropout
    uncertainty inference, where stochastic sampling must come exclusively
    from dropout and not from any other stateful layer.

    Args:
        model (torch.nn.Module): Model to toggle in place.

    Returns:
        None
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predict_ensemble(
    models: List[torch.nn.Module],
    inputs: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate predictions from independently initialized Deep Ensemble models.

    Each model performs a single deterministic forward pass (dropout inactive,
    ``model.eval()``) over the same input batch. The ensemble predictive mean
    and sample predictive variance are computed across the member axis.

    Args:
        models (List[torch.nn.Module]): Independently trained ensemble members,
            all sharing the same architecture and input feature space.
        inputs (np.ndarray): Input feature matrix, shape (N, D).
        device (torch.device): Device used for inference.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Ensemble predictive mean (N,) and
            sample predictive variance (N,), the latter clamped to be
            non-negative.
    """
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    member_preds: List[np.ndarray] = []

    with torch.no_grad():
        for model in models:
            model.eval()
            out = model(x).detach().cpu().numpy().reshape(-1)
            member_preds.append(out)

    stacked = np.stack(member_preds, axis=0)  # (M, N)
    mean = stacked.mean(axis=0)
    if stacked.shape[0] > 1:
        var = stacked.var(axis=0, ddof=1)
    else:
        var = np.zeros_like(mean)
    var = np.clip(var, a_min=0.0, a_max=None)
    return mean, var


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
        task_type: str = "regression",
        traj_error_mode: str = "abs",  # 'abs' | 'bce'
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
            task_type (str): 'regression' or 'classification'
            traj_error_mode (str): For classification trajectories:
                - 'abs': abs(y - prob)
                - 'bce': per-sample BCE

        Returns:
            Dict[str, List[float]]: Training history
        """
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")

        # Ensure y is (N, 1)
        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

        dataset = MyDataset(X, y_2d)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # History: keep both sets for compatibility,
        # but "loss" and best checkpoint selection are task-dependent.
        history: Dict[str, List[float]] = {
            "epoch": [],
            "loss": [],       # regression: MSE(train), classification: BCE(train)
            "val_mse": [],
            "val_rmse": [],
            "val_r2": [],
            "val_pcc": [],
            "val_ci": [],
            "val_bce": [],
            "val_auc": [],
            "val_acc": [],
        }

        # Best selection rule:
        # - regression: minimize val_mse
        # - classification: minimize val_bce
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

                out = self.model(inputs)  # regression: pred, classification: logits

                if task_type == "classification":
                    # BCEWithLogitsLoss (numerically stable)
                    loss = F.binary_cross_entropy_with_logits(out, targets)
                else:
                    # MSE for regression
                    loss = F.mse_loss(out, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())

            loss_avg = loss_sum / max(1, len(loader))
            history["epoch"].append(float(epoch))
            history["loss"].append(float(loss_avg))

            # -------------------------
            # Validation (task-specific)
            # -------------------------
            if validation_data is not None:
                met = self.evaluate(
                    X=validation_data["X"],
                    y=validation_data["y"],
                    batch_size=batch_size,
                    task_type=task_type,
                )

                # Regression metrics
                history["val_mse"].append(float(met.get("mse", 0.0)))
                history["val_rmse"].append(float(met.get("rmse", 0.0)))
                history["val_r2"].append(float(met.get("r2", 0.0)))
                history["val_pcc"].append(float(met.get("pcc", 0.0)))
                history["val_ci"].append(float(met.get("ci", 0.0)))

                # Classification metrics
                history["val_bce"].append(float(met.get("bce", 0.0)))
                history["val_auc"].append(float(met.get("auc", 0.0)))
                history["val_acc"].append(float(met.get("acc", 0.0)))

                if task_type == "classification":
                    score = float(met.get("bce", float("inf")))
                else:
                    score = float(met.get("mse", float("inf")))

                if score < best_score:
                    best_score = score
                    self.save_model(filepath_ckpt)
            else:
                # No validation -> keep last checkpoint
                history["val_mse"].append(0.0)
                history["val_rmse"].append(0.0)
                history["val_r2"].append(0.0)
                history["val_pcc"].append(0.0)
                history["val_ci"].append(0.0)
                history["val_bce"].append(0.0)
                history["val_auc"].append(0.0)
                history["val_acc"].append(0.0)
                self.save_model(filepath_ckpt)

            # -------------------------
            # Trajectory accumulation
            # -------------------------
            if save_trajectory and (epoch % save_every == 0):
                self.model.eval()
                with torch.no_grad():
                    # Train outputs
                    tr_logits_or_pred = self.model(
                        torch.tensor(X, device=self.device, dtype=torch.float32)
                    ).cpu().numpy().reshape(-1)

                    y_true_tr = y_2d.reshape(-1)

                    if task_type == "classification":
                        tr_prob = 1.0 / (1.0 + np.exp(-tr_logits_or_pred))
                        if traj_error_mode == "bce":
                            eps = 1e-8
                            tr_err = -(y_true_tr * np.log(tr_prob + eps) + (1.0 - y_true_tr) * np.log(1.0 - tr_prob + eps))
                        else:
                            tr_err = np.abs(y_true_tr - tr_prob)
                    else:
                        tr_pred = tr_logits_or_pred
                        tr_err = np.abs(y_true_tr - tr_pred)

                    for cid, e in zip(compound_ids_tr, tr_err):
                        train_traj[cid].append(float(e))

                    # Val outputs
                    if validation_data is not None:
                        if compound_ids_va is None or smiles_va is None:
                            raise ValueError("compound_ids_va and smiles_va are required when validation_data is provided")

                        va_logits_or_pred = self.model(
                            torch.tensor(validation_data["X"], device=self.device, dtype=torch.float32)
                        ).cpu().numpy().reshape(-1)

                        y_true_va = np.asarray(validation_data["y"]).reshape(-1)

                        if task_type == "classification":
                            va_prob = 1.0 / (1.0 + np.exp(-va_logits_or_pred))
                            if traj_error_mode == "bce":
                                eps = 1e-8
                                va_err = -(y_true_va * np.log(va_prob + eps) + (1.0 - y_true_va) * np.log(1.0 - va_prob + eps))
                            else:
                                va_err = np.abs(y_true_va - va_prob)
                        else:
                            va_pred = va_logits_or_pred
                            va_err = np.abs(y_true_va - va_pred)

                        for cid, e in zip(compound_ids_va, va_err):
                            val_traj[cid].append(float(e))

        # -------------------------
        # Save trajectories
        # -------------------------
        if save_trajectory:
            def _save(path: str, ids: np.ndarray, smiles: np.ndarray, y_true_arr: np.ndarray, traj: Dict[Any, List[float]]) -> None:
                """
                Args:
                    path (str): Output csv path
                    ids (np.ndarray): Compound IDs
                    smiles (np.ndarray): SMILES
                    y_true_arr (np.ndarray): Labels
                    traj (Dict[Any, List[float]]): Trajectory dict

                Returns:
                    None
                """
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
        task_type: str = "regression",
    ) -> dict:
        """
        Args:
            X (np.ndarray): features
            y (np.ndarray): labels
            batch_size (int): batch size
            task_type (str): 'regression' | 'classification'

        Returns:
            dict: metrics dict from get_metrics (+ bce for classification)
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

                out = self.model(inputs)  # regression: pred, classification: logits
                y_true_list.append(targets.detach().cpu().numpy())
                y_out_list.append(out.detach().cpu().numpy())

        y_true = np.vstack(y_true_list).reshape(-1)
        y_out = np.vstack(y_out_list).reshape(-1)

        met = get_metrics(y_true=y_true, y_pred=y_out, task_type=task_type)

        if task_type == "classification":
            # NOTE: y_out is logits here
            bce = float(
                F.binary_cross_entropy_with_logits(
                    torch.tensor(y_out, dtype=torch.float32),
                    torch.tensor(y_true, dtype=torch.float32),
                ).item()
            )
            met["bce"] = bce

        return met


    def inference(
        self,
        X: np.ndarray,
        batch_size: int = 256,
        task_type: str = "regression",
    ) -> np.ndarray:
        """
        Args:
            X (np.ndarray): Input features, shape (N, D)
            batch_size (int): Batch size for inference
            task_type (str): 'regression' | 'classification'

        Returns:
            np.ndarray: regression -> predictions, classification -> probabilities
        """
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        outs = []

        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                logits_or_pred = self.model(inputs)  # (B, 1)

                if task_type == "classification":
                    prob = torch.sigmoid(logits_or_pred)
                    outs.append(prob.cpu().numpy())
                else:
                    outs.append(logits_or_pred.cpu().numpy())

        return np.vstack(outs).reshape(-1)

    def predict_mc_dropout(
        self,
        inputs: np.ndarray,
        num_samples: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stochastic MC Dropout prediction.

        Runs ``num_samples`` stochastic forward passes of the modality MLP's
        raw prediction head (the same output used for ordinary inference,
        i.e. the normalized-target-space prediction, not the hidden
        embedding) with only dropout layers active. The sample predictive
        mean and variance across these passes are used as the MC Dropout
        predictive mean and modality uncertainty, respectively.

        Args:
            inputs (np.ndarray): Input feature matrix, shape (N, D).
            num_samples (int): Number of stochastic forward passes (T).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictive mean (N,) and sample
                predictive variance (N,), the latter clamped to be
                non-negative.
        """
        if num_samples < 2:
            raise ValueError("num_samples must be >= 2 to compute a sample variance")

        x = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        _enable_dropout_only(self.model)
        samples: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self.model(x).detach().cpu().numpy().reshape(-1)
                samples.append(out)
        self.model.eval()

        stacked = np.stack(samples, axis=0)  # (T, N)
        mean = stacked.mean(axis=0)
        var = stacked.var(axis=0, ddof=1)
        var = np.clip(var, a_min=0.0, a_max=None)
        return mean, var



class EvidentialTrainer(Trainer):
    """
    Trainer for an ``EvidentialMLP`` modality model.

    Only the training objective differs from ``Trainer``: the NIG negative
    log-likelihood plus an evidence regularizer replaces MSE. Everything
    downstream of training is inherited unchanged, because
    ``EvidentialMLP.forward`` returns the predictive mean gamma and so
    satisfies the ``(N, 1)`` contract that ``evaluate``, ``inference`` and
    ``get_embeddings`` are written against.

    Regression only -- the NIG likelihood is defined over a continuous
    target, so there is no classification path here.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, lambda_reg: float = 0.01):
        """
        Args:
            model (torch.nn.Module): An ``EvidentialMLP`` instance.
            device (torch.device): Torch device.
            lambda_reg (float): Weight on the evidence regularizer.
        """
        super().__init__(model, device)
        self.lambda_reg = float(lambda_reg)

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
        task_type: str = "regression",
        traj_error_mode: str = "abs",
    ) -> Dict[str, List[float]]:
        """
        Train the NIG head by minimizing ``evidential_loss``.

        The best checkpoint is the one with the lowest validation NIG total
        loss, not the lowest validation MSE. The NIG objective scores the
        whole predicted distribution, so selecting on it keeps the
        uncertainty calibrated -- which matters here because that
        uncertainty sets the fusion weights. Validation MSE is still
        recorded for comparability with the other UQ methods.

        Args:
            feature_name (str): Modality name, e.g., 'fp', 'img'.
            X (np.ndarray): Training features (N, D).
            y (np.ndarray): Training labels (N, 1) or (N,).
            batch_size (int): Batch size.
            num_epochs (int): Epochs.
            learning_rate (float): Learning rate.
            validation_data (Optional[Dict[str, np.ndarray]]): {'X':..., 'y':...}.
            filepath_ckpt (str): Best checkpoint path.
            save_trajectory (bool): Must be False. Accepted only because
                ``train_mlp`` passes the LTAU trajectory arguments to every
                trainer; Evidential Regression has no loss trajectory.
            trajectory_dir, save_every, compound_ids_tr, compound_ids_va,
            smiles_tr, smiles_va, traj_error_mode: LTAU-only, ignored here.
            task_type (str): Must be 'regression'.

        Returns:
            Dict[str, List[float]]: Training history, using the same keys as
                ``Trainer.fit`` plus ``val_total``/``val_nll``/``val_reg``.
                The ``loss`` column holds the training NIG total loss.

        Raises:
            ValueError: If ``task_type`` is not 'regression'.
            FloatingPointError: If the NIG loss becomes non-finite before
                any checkpoint could be saved.
        """
        if save_trajectory:
            raise ValueError(
                "save_trajectory is not supported for Evidential Regression: it produces "
                "NIG parameters, not the per-epoch error trajectory LTAU consumes."
            )
        if task_type != "regression":
            raise ValueError(
                f"EvidentialTrainer supports task_type='regression' only, got {task_type!r}"
            )

        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y
        loader = DataLoader(MyDataset(X, y_2d), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history: Dict[str, List[float]] = {
            "epoch": [], "loss": [],
            "val_mse": [], "val_rmse": [], "val_r2": [], "val_pcc": [], "val_ci": [],
            "val_bce": [], "val_auc": [], "val_acc": [],
            "val_total": [], "val_nll": [], "val_reg": [],
        }

        has_val = validation_data is not None
        if has_val:
            x_val_t = torch.tensor(validation_data["X"], dtype=torch.float32, device=self.device)
            y_val_raw = validation_data["y"]
            y_val_t = torch.tensor(
                y_val_raw.reshape(-1, 1) if y_val_raw.ndim == 1 else y_val_raw,
                dtype=torch.float32, device=self.device,
            )

        best_score = float("inf")
        diverged_at_epoch = None

        epoch_iter = tqdm(
            range(num_epochs), leave=True, dynamic_ncols=True,
            desc=f"Training {feature_name} evidential MLP",
        )

        for epoch in epoch_iter:
            self.model.train()
            loss_sum = 0.0
            train_diverged = False

            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                params = self.model.forward_nig(inputs)
                total, _, _ = evidential_loss(targets, params, self.lambda_reg)

                if not torch.isfinite(total):
                    train_diverged = True
                    break

                optimizer.zero_grad()
                total.backward()
                # The NIG log terms can produce large gradients early on;
                # clipping keeps a bad batch from destroying the weights.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                loss_sum += float(total.item())

            if train_diverged:
                diverged_at_epoch = epoch
                break

            history["epoch"].append(float(epoch))
            history["loss"].append(float(loss_sum / max(1, len(loader))))

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_params = self.model.forward_nig(x_val_t)
                    val_total, val_nll, val_reg = evidential_loss(
                        y_val_t, val_params, self.lambda_reg
                    )
                vals = [float(val_total.item()), float(val_nll.item()), float(val_reg.item())]

                if not np.isfinite(vals).all():
                    diverged_at_epoch = epoch
                    history["epoch"].pop()
                    history["loss"].pop()
                    break

                met = self.evaluate(
                    X=validation_data["X"], y=validation_data["y"],
                    batch_size=batch_size, task_type="regression",
                )
                history["val_mse"].append(float(met.get("mse", 0.0)))
                history["val_rmse"].append(float(met.get("rmse", 0.0)))
                history["val_r2"].append(float(met.get("r2", 0.0)))
                history["val_pcc"].append(float(met.get("pcc", 0.0)))
                history["val_ci"].append(float(met.get("ci", 0.0)))
                history["val_total"].append(vals[0])
                history["val_nll"].append(vals[1])
                history["val_reg"].append(vals[2])

                if vals[0] < best_score:
                    best_score = vals[0]
                    self.save_model(filepath_ckpt)
            else:
                for key in ("val_mse", "val_rmse", "val_r2", "val_pcc", "val_ci",
                            "val_total", "val_nll", "val_reg"):
                    history[key].append(0.0)
                self.save_model(filepath_ckpt)

            for key in ("val_bce", "val_auc", "val_acc"):
                history[key].append(0.0)

        if diverged_at_epoch is not None:
            if not os.path.exists(filepath_ckpt):
                raise FloatingPointError(
                    f"[{feature_name}] NIG loss became non-finite at epoch "
                    f"{diverged_at_epoch} with no earlier checkpoint to fall back on"
                )
            print(
                f"[WARN] {feature_name}: NIG loss became non-finite at epoch "
                f"{diverged_at_epoch}; falling back to the best checkpoint so far "
                f"(val_total={best_score:.6g})",
                flush=True,
            )

        self.load_model(filepath_ckpt)
        return history

    def predict_evidential(
        self,
        inputs: np.ndarray,
        term: str = "epistemic",
        batch_size: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the mean and the NIG uncertainty in a single forward pass.

        The return signature matches ``predict_mc_dropout`` and
        ``predict_ensemble`` so ``CUE.compute_uncertainty`` can treat all
        three uncertainty methods the same way.

        Args:
            inputs (np.ndarray): Input feature matrix, shape (N, D).
            term (str): Which NIG uncertainty term to return; see
                ``cue.core.evidential.nig_uncertainty``.
            batch_size (int): Batch size for inference.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictive mean gamma (N,) and
                non-negative uncertainty (N,), both in the same (possibly
                normalized) target space the model was trained on.
        """
        dataset = torch.utils.data.TensorDataset(torch.tensor(inputs, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                params = self.model.forward_nig(batch.to(self.device))
                chunks.append(
                    np.stack([p.detach().cpu().numpy().reshape(-1) for p in params], axis=1)
                )

        stacked = np.vstack(chunks)
        gamma, nu, alpha, beta = (stacked[:, i] for i in range(4))
        return gamma, nig_uncertainty(nu, alpha, beta, term=term)


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
        task_type: str = "regression",
        traj_error_mode: str = "abs",  # 'abs' | 'bce'
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
            task_type (str): 'regression' or 'classification'
            traj_error_mode (str): For classification trajectories:
                - 'abs': abs(y - prob)
                - 'bce': per-sample BCE

        Returns:
            Dict[str, List[float]]: training history
        """
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")

        # Ensure y is (N, 1)
        y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

        dataset = NumpyRegressionDataset(X, y_2d)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # History init (task-specific)
        if task_type == "classification":
            history: Dict[str, List[float]] = {
                "epoch": [],
                "loss": [],  # train BCE
                "val_bce": [],
                "val_f1": [],
                "val_precision": [],
                "val_recall": [],
                "val_accuracy": [],
                "val_auc_roc": [],
                "val_mcc": [],
            }
        else:
            history = {
                "epoch": [],
                "loss": [],  # train MSE
                "val_mse": [],
                "val_rmse": [],
                "val_r2": [],
                "val_pcc": [],
                "val_ci": [],
            }


        # Best selection rule:
        # - regression: minimize val_mse
        # - classification: minimize val_bce
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

                out = self.model(inputs)  # regression: pred, classification: logits

                if task_type == "classification":
                    loss = F.binary_cross_entropy_with_logits(out, targets)
                else:
                    loss = F.mse_loss(out, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.item())

            loss_avg = loss_sum / max(1, len(loader))
            history["epoch"].append(float(epoch))
            history["loss"].append(float(loss_avg))

            # -------------------------
            # Validation (task-specific)
            # -------------------------
            if validation_data is not None:
                met = self.evaluate(
                    X=validation_data["X"],
                    y=validation_data["y"],
                    batch_size=batch_size,
                    task_type=task_type,
                )

                if task_type == "classification":
                    history["val_bce"].append(float(met.get("bce", 0.0)))
                    history["val_f1"].append(float(met.get("f1", 0.0)))
                    history["val_precision"].append(float(met.get("precision", 0.0)))
                    history["val_recall"].append(float(met.get("recall", 0.0)))
                    history["val_accuracy"].append(float(met.get("accuracy", 0.0)))
                    history["val_auc_roc"].append(float(met.get("auc_roc", 0.0)))
                    history["val_mcc"].append(float(met.get("mcc", 0.0)))
                    score = float(met.get("bce", float("inf")))
                else:
                    history["val_mse"].append(float(met.get("mse", 0.0)))
                    history["val_rmse"].append(float(met.get("rmse", 0.0)))
                    history["val_r2"].append(float(met.get("r2", 0.0)))
                    history["val_pcc"].append(float(met.get("pcc", 0.0)))
                    history["val_ci"].append(float(met.get("ci", 0.0)))
                    score = float(met.get("mse", float("inf")))

                if score < best_score:
                    best_score = score
                    self.save_model(filepath_ckpt)
            else:
                # no validation: still keep schema consistent
                if task_type == "classification":
                    history["val_bce"].append(0.0)
                    history["val_f1"].append(0.0)
                    history["val_precision"].append(0.0)
                    history["val_recall"].append(0.0)
                    history["val_accuracy"].append(0.0)
                    history["val_auc_roc"].append(0.0)
                    history["val_mcc"].append(0.0)
                else:
                    history["val_mse"].append(0.0)
                    history["val_rmse"].append(0.0)
                    history["val_r2"].append(0.0)
                    history["val_pcc"].append(0.0)
                    history["val_ci"].append(0.0)

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
                    tr_logits_or_pred = self.inference(X, batch_size=256, task_type=task_type).reshape(-1)
                    y_true_tr = y_2d.reshape(-1)

                    if task_type == "classification":
                        tr_prob = 1.0 / (1.0 + np.exp(-tr_logits_or_pred))
                        if traj_error_mode == "bce":
                            eps = 1e-8
                            tr_err = -(y_true_tr * np.log(tr_prob + eps) + (1.0 - y_true_tr) * np.log(1.0 - tr_prob + eps))
                        else:
                            tr_err = np.abs(y_true_tr - tr_prob)
                    else:
                        tr_pred = tr_logits_or_pred
                        tr_err = np.abs(y_true_tr - tr_pred)

                    for cid, e in zip(compound_ids_tr, tr_err):
                        train_traj[cid].append(float(e))

                    # Val outputs
                    if validation_data is not None:
                        if compound_ids_va is None or smiles_va is None:
                            raise ValueError("compound_ids_va and smiles_va are required for val trajectory")

                        va_logits_or_pred = self.inference(validation_data["X"], batch_size=256, task_type=task_type).reshape(-1)
                        y_true_va = np.asarray(validation_data["y"]).reshape(-1)

                        if task_type == "classification":
                            va_prob = 1.0 / (1.0 + np.exp(-va_logits_or_pred))
                            if traj_error_mode == "bce":
                                eps = 1e-8
                                va_err = -(y_true_va * np.log(va_prob + eps) + (1.0 - y_true_va) * np.log(1.0 - va_prob + eps))
                            else:
                                va_err = np.abs(y_true_va - va_prob)
                        else:
                            va_pred = va_logits_or_pred
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
                """
                Args:
                    path (str): csv output path
                    ids (np.ndarray): compound ids
                    smiles (np.ndarray): SMILES
                    y_true (np.ndarray): labels
                    traj (Dict[Any, List[float]]): error trajectory dict

                Returns:
                    None
                """
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
        task_type: str = "regression",
    ) -> Dict[str, float]:
        """
        Args:
            X (np.ndarray): Input features, shape (N, D)
            y (np.ndarray): Labels, shape (N,) or (N, 1)
            batch_size (int): Batch size
            task_type (str): 'regression' | 'classification'

        Returns:
            Dict[str, float]: Metrics dict from get_metrics (+ bce for classification)
        """
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")

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

                out = self.model(inputs)  # regression: pred, classification: logits
                y_true_list.append(targets.detach().cpu().numpy())
                y_out_list.append(out.detach().cpu().numpy())

        y_true = np.vstack(y_true_list).reshape(-1)
        y_out = np.vstack(y_out_list).reshape(-1)

        # NOTE:
        # - regression: y_out is prediction
        # - classification: y_out is logits (we keep logits for metrics & bce)
        met = get_metrics(y_true=y_true, y_pred=y_out, task_type=task_type)

        if task_type == "classification":
            bce = float(
                F.binary_cross_entropy_with_logits(
                    torch.tensor(y_out, dtype=torch.float32),
                    torch.tensor(y_true, dtype=torch.float32),
                ).item()
            )
            met["bce"] = bce

        return met

    def inference(
        self,
        X: np.ndarray,
        batch_size: int = 256,
        task_type: str = "regression",
        return_logits: bool = False,
    ) -> np.ndarray:
        """
        Args:
            X (np.ndarray): Input features, shape (N, D)
            batch_size (int): Batch size
            task_type (str): 'regression' | 'classification'
            return_logits (bool): If True (classification only), return logits instead of probabilities

        Returns:
            np.ndarray:
                - regression: predictions, shape (N,)
                - classification: probabilities by default, shape (N,)
                  (or logits if return_logits=True)
        """
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")

        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

        self.model.eval()
        outs = []

        with torch.no_grad():
            for (inputs,) in dl:
                inputs = inputs.to(self.device)
                out = self.model(inputs)  # (B, 1) pred or logits

                if task_type == "classification":
                    if return_logits:
                        outs.append(out.detach().cpu().numpy())
                    else:
                        prob = torch.sigmoid(out)
                        outs.append(prob.detach().cpu().numpy())
                else:
                    outs.append(out.detach().cpu().numpy())

        return np.vstack(outs).reshape(-1)
    
    def plot(
        self,
        history: Dict[str, List[float]],
        save_dir: str,
        task_type: str = "regression",
        best_key: str = "val_mse",
    ) -> None:
        """
        Args:
            history (Dict[str, List[float]]): training history
            save_dir (str): output directory
            task_type (str): 'regression' | 'classification'
            best_key (str): 'val_mse' or 'val_bce'

        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)

        epochs = np.asarray(history.get("epoch", []), dtype=float)
        loss_tr = np.asarray(history.get("loss", []), dtype=float)

        if len(epochs) == 0:
            return

        if best_key not in history:
            # fallback
            best_key = "val_bce" if task_type == "classification" else "val_mse"

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
