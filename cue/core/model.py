from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from .fusion import UncertaintyNormalizer, compute_fusion_weights
from .train_mlp import MLPTrainConfigs, train_mlp
from .mlp import MLP
from .evidential import EVIDENTIAL_UNCERTAINTY_TERMS, EvidentialMLP
from .trainer import EvidentialTrainer, Trainer, predict_ensemble
from .downstream_runner import load_embedding_lookup
from cue.utils.FeatureExtraction import MolEncoder
from cue.utils.seed import seed

# smi/gnn/conf are cache-only (no live path - see MODALITY_LIVE_GETTER). fp/img prefer the
# cue.utils.prepare_features.py .npy cache when cfg.data_fold_dir is set and the SMILES is
# present in it (training: cue_runner.py always passes the fold's own data_fold_dir, whose
# cache covers every training SMILES) but fall back to live MolEncoder computation otherwise
# (predict.py scores arbitrary new compounds with no cache / not present in any fold's cache).
MODALITY_DIM_ATTR = {
    "fp": "feature_dim_fp",
    "img": "feature_dim_img",
    "smi": "feature_dim_smi",
    "conf": "feature_dim_conf",
    # "gnn" is resolved via GNN_BACKEND_DIM_ATTR below (backend-dependent dim).
}
MODALITY_LIVE_GETTER = {
    "fp": "get_fpt_features",
    "img": "get_img_features",
}
MODALITY_CACHE_KEY = {
    "fp": "fp",
    "img": "img",
    "smi": "smi_mean",
    "conf": "conf",
    # "gnn" is resolved via GNN_BACKEND_CACHE_KEY below (backend-dependent cache key).
}

# "gnn" is a single modality name with two selectable backends (both pre-cached by
# cue.utils.prepare_features so switching backend never requires recomputation):
#   - 'mole'  : MolE (cue/mole/), 768-d, cache key 'gnn_features'
#   - 'mgssl' : MGSSL (cue/mgssl/), 300-d, cache key 'gnn_features_mgssl'
SUPPORTED_GNN_BACKENDS = ("mole", "mgssl")
GNN_BACKEND_DIM_ATTR = {"mole": "feature_dim_gnn", "mgssl": "feature_dim_gnn_mgssl"}
GNN_BACKEND_CACHE_KEY = {"mole": "gnn", "mgssl": "gnn_mgssl"}

# Per-modality, per-sample uncertainty u_m(x) can be computed three ways (freely combinable
# with any modality subset above):
#   - 'ltau'         : error-trajectory histograms -> KNN entropy/expected-error estimator
#                      (cue/ltau/, orchestrated in cue/core/cue_runner.py). Default; unchanged
#                      from the original implementation.
#   - 'mc_dropout'    : T stochastic dropout forward passes of a single trained MLP member,
#                      predictive variance (ported from cue4unc).
#   - 'deep_ensemble' : `ensemble_size` independently seeded MLPs per modality, predictive
#                      variance across members (ported from cue4unc).
#   - 'evidential'    : one MLP per modality predicting the four NIG parameters; uncertainty
#                      comes from a single forward pass (ported from cue4unc). Regression only.
SUPPORTED_UNCERTAINTY_METHODS = ("ltau", "mc_dropout", "deep_ensemble", "evidential")


@dataclass
class CUEConfigs:
    """
    Args:
        fold_dir (str): Fold output directory
        device (torch.device): Torch device
        modalities (List[str]): Active modalities, subset of {fp, img, smi, gnn, conf}
        data_fold_dir (Optional[str]): Fold *data* directory (e.g.
            chembl/data/selectivity_df/Kd/fold1 — the on-disk assay name), used to locate the cached .npy embeddings
            produced by cue.utils.prepare_features. Required if smi/gnn/conf are active
            (cache-only, no live path); for fp/img it is optional but preferred when set -
            those two fall back to live MolEncoder computation for any SMILES missing from
            the cache (or when data_fold_dir is None, e.g. predict.py scoring arbitrary new
            compounds). Distinct from fold_dir (the output directory).
        gnn_backend (str): One of SUPPORTED_GNN_BACKENDS ('mole' | 'mgssl'). Selects which
            pretrained GNN backs the 'gnn' modality when it's active. Both backends are
            pre-cached separately (see GNN_BACKEND_CACHE_KEY), so switching this never
            requires recomputing embeddings.
        hidden_dim (int): MLP hidden dim
        num_epochs (int): MLP pretrain epochs
        learning_rate (float): MLP pretrain lr
        batch_size (int): MLP pretrain batch size
        save_every (int): Trajectory saving frequency
        uncertainty_method (str): One of SUPPORTED_UNCERTAINTY_METHODS. Selects how per-modality,
            per-sample uncertainty u_m(x) is computed for adaptive fusion.
        mc_samples (int): Stochastic forward passes (T) for MC Dropout. Only used when
            uncertainty_method == 'mc_dropout'.
        dropout_rate (float): Dropout probability inside each modality MLP. Only used when
            uncertainty_method == 'mc_dropout'; forced to 0.0 for 'deep_ensemble' so ensemble
            uncertainty reflects only random-initialization variance.
        ensemble_size (int): Independently initialized modality MLPs (M) per modality. Only
            used when uncertainty_method == 'deep_ensemble'.
        ensemble_seeds (List[int]): Random seeds, one per ensemble member. Must have length
            >= ensemble_size when uncertainty_method == 'deep_ensemble'.
        evidential_lambda (float): Weight on the NIG evidence regularizer. Only used when
            uncertainty_method == 'evidential'.
        evidential_uncertainty (str): Which NIG term drives fusion, one of
            EVIDENTIAL_UNCERTAINTY_TERMS. 'epistemic' matches what MC Dropout and Deep
            Ensemble supply, so the methods stay comparable.
    """
    fold_dir: str
    device: torch.device
    modalities: List[str] = field(default_factory=lambda: ["fp", "img"])
    data_fold_dir: Optional[str] = None
    gnn_backend: str = "mgssl"
    hidden_dim: int = 32
    num_epochs: int = 2000
    learning_rate: float = 1e-4
    batch_size: int = 32
    save_every: int = 1
    task_type: str = "regression"
    uncertainty_method: str = "ltau"
    mc_samples: int = 10
    dropout_rate: float = 0.1
    ensemble_size: int = 10
    ensemble_seeds: List[int] = field(default_factory=lambda: list(range(1, 11)))
    evidential_lambda: float = 0.01
    evidential_uncertainty: str = "epistemic"

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If uncertainty_method or gnn_backend is not supported, or if
                ensemble_seeds is shorter than ensemble_size while using the
                'deep_ensemble' method.
        """
        if self.uncertainty_method not in SUPPORTED_UNCERTAINTY_METHODS:
            raise ValueError(
                f"Unsupported uncertainty_method={self.uncertainty_method!r}; "
                f"expected one of {SUPPORTED_UNCERTAINTY_METHODS}"
            )
        if self.gnn_backend not in SUPPORTED_GNN_BACKENDS:
            raise ValueError(
                f"Unsupported gnn_backend={self.gnn_backend!r}; expected one of {SUPPORTED_GNN_BACKENDS}"
            )
        if self.uncertainty_method == "deep_ensemble" and len(self.ensemble_seeds) < self.ensemble_size:
            raise ValueError(
                f"ensemble_seeds (len={len(self.ensemble_seeds)}) must have at least "
                f"ensemble_size={self.ensemble_size} entries"
            )
        if self.evidential_uncertainty not in EVIDENTIAL_UNCERTAINTY_TERMS:
            raise ValueError(
                f"Unsupported evidential_uncertainty={self.evidential_uncertainty!r}; "
                f"expected one of {EVIDENTIAL_UNCERTAINTY_TERMS}"
            )
        if self.uncertainty_method == "evidential" and self.task_type != "regression":
            raise ValueError(
                "uncertainty_method='evidential' supports task_type='regression' only; "
                f"got task_type={self.task_type!r} (the NIG likelihood is defined over a "
                "continuous target)"
            )


class CUE:
    """
    CUE = modality-specific MLP pretraining + uncertainty-aware fusion (in predict),
    generalized to any subset of {fp, img, smi, gnn, conf} x any of
    {ltau, mc_dropout, deep_ensemble} uncertainty methods.

    Per modality, one or more independently trained ``Trainer`` instances are kept in a list:
      - 'ltau' / 'mc_dropout': a single member (index 0). For 'mc_dropout', dropout layers are
        enabled inside the MLP and uncertainty comes from T stochastic forward passes of this
        member (see ``compute_uncertainty``); for 'ltau', uncertainty is computed upstream in
        cue/core/cue_runner.py via the LTAU descriptor/KNN pipeline instead.
      - 'deep_ensemble': ``cfg.ensemble_size`` members, each trained from a distinct random
        seed/initialization on the same data. Uncertainty comes from the predictive variance
        across members.
    """

    def __init__(self, cfg: CUEConfigs):
        """
        Args:
            cfg (CUEConfigs): CUE configuration
        """
        self.cfg = cfg
        self.device = cfg.device
        self.fold_dir = cfg.fold_dir
        self.modalities: List[str] = list(cfg.modalities)

        os.makedirs(self.fold_dir, exist_ok=True)
        os.makedirs(os.path.join(self.fold_dir, "checkpoints"), exist_ok=True)

        # Encoder (features)
        self.encoder = MolEncoder(self.device)

        # Number of independently trained members per modality.
        self.num_members = 1 if cfg.uncertainty_method != "deep_ensemble" else cfg.ensemble_size

        # Trainers (list per modality, length == self.num_members) + scalers (per modality,
        # shared across members)
        self.mlps: Dict[str, List[Trainer]] = {}
        self.scalers: Dict[str, MinMaxScaler] = {}
        for m in self.modalities:
            dim = getattr(self.encoder, self._dim_attr(m))
            self.mlps[m] = [self._build_member(dim) for _ in range(self.num_members)]
            self.scalers[m] = MinMaxScaler()

        # Lazily-built SMILES -> embedding lookup, one per cacheable modality (smi/gnn/conf
        # always populated; fp/img may be None if no cache file was found, meaning "always
        # fall back to live computation for this modality").
        self._lookup_cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}

    def _build_member(self, input_dim: int) -> Trainer:
        """Build one modality-MLP member for the configured uncertainty method.

        Evidential members predict the four NIG parameters and are wrapped in
        EvidentialTrainer; every other method uses a plain MLP. Dropout is only
        enabled for MC Dropout -- Deep Ensemble members isolate random-initialization
        variance and therefore use dropout_rate=0.0, as does LTAU.

        Args:
            input_dim (int): Feature dimension of the modality.

        Returns:
            Trainer: The member's trainer (EvidentialTrainer for 'evidential').
        """
        cfg = self.cfg
        if cfg.uncertainty_method == "evidential":
            return EvidentialTrainer(
                EvidentialMLP(input_dim, hidden_dim=cfg.hidden_dim, dropout_rate=0.0),
                self.device,
                lambda_reg=cfg.evidential_lambda,
            )

        member_dropout_rate = cfg.dropout_rate if cfg.uncertainty_method == "mc_dropout" else 0.0
        return Trainer(
            MLP(input_dim, hidden_dim=cfg.hidden_dim, dropout_rate=member_dropout_rate),
            self.device,
        )

    def _dim_attr(self, modality: str) -> str:
        """
        Args:
            modality (str): modality name

        Returns:
            str: MolEncoder attribute name holding this modality's feature dim ('gnn'
                resolves via cfg.gnn_backend since MolE/MGSSL have different dims).
        """
        if modality == "gnn":
            return GNN_BACKEND_DIM_ATTR[self.cfg.gnn_backend]
        return MODALITY_DIM_ATTR[modality]

    def _cache_key(self, modality: str) -> str:
        """
        Args:
            modality (str): modality name

        Returns:
            str: cue.utils.prepare_features cache key for this modality ('gnn' resolves
                via cfg.gnn_backend to either the MolE or MGSSL cache).
        """
        if modality == "gnn":
            return GNN_BACKEND_CACHE_KEY[self.cfg.gnn_backend]
        return MODALITY_CACHE_KEY[modality]

    def _get_feature_fn(self, modality: str):
        """
        Args:
            modality (str): modality name

        Returns:
            Callable[[str], np.ndarray]: SMILES -> feature vector. For smi/gnn/conf, always
                reads from the prepare_features cache (raises if unavailable). For fp/img,
                reads from cache when possible and falls back to live MolEncoder computation
                per-SMILES for entries missing from it (or when no cache is available at all).
        """
        live_attr = MODALITY_LIVE_GETTER.get(modality)

        if live_attr is None:
            # smi/gnn/conf: no live path, cache is mandatory (unchanged behavior).
            if modality not in self._lookup_cache:
                if self.cfg.data_fold_dir is None:
                    raise ValueError(
                        f"CUEConfigs.data_fold_dir is required to use modality '{modality}' "
                        "(it is read from the cached .npy produced by cue.utils.prepare_features)."
                    )
                self._lookup_cache[modality] = load_embedding_lookup(
                    self.cfg.data_fold_dir, self._cache_key(modality)
                )
            lookup = self._lookup_cache[modality]
            return lambda s: lookup[s]

        # fp/img: cache-first (when data_fold_dir is set), live fallback otherwise/for
        # SMILES missing from the cache.
        live_fn = getattr(self.encoder, live_attr)
        if self.cfg.data_fold_dir is None:
            return live_fn

        if modality not in self._lookup_cache:
            try:
                self._lookup_cache[modality] = load_embedding_lookup(
                    self.cfg.data_fold_dir, self._cache_key(modality)
                )
            except FileNotFoundError:
                self._lookup_cache[modality] = None

        lookup = self._lookup_cache[modality]
        if lookup is None:
            return live_fn
        return lambda s: lookup[s] if s in lookup else live_fn(s)

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
        Train modality-specific MLPs. For 'deep_ensemble', trains cfg.ensemble_size independent
        MLPs per modality, each reseeded with a distinct entry from cfg.ensemble_seeds
        immediately before construction and training so that both weight initialization and
        data-loader shuffling differ across members.

        Args:
            tr_id (np.ndarray): Train IDs
            va_id (np.ndarray): Val IDs
            X_tr (np.ndarray): Train SMILES
            X_va (np.ndarray): Val SMILES
            y_tr (np.ndarray): Train labels (UNNORMALIZED), shape (N_tr, 1)
            y_va (np.ndarray): Val labels (UNNORMALIZED), shape (N_va, 1)

        Returns:
            Dict[str, float]: best validation loss per modality (last-trained member's score;
                for 'deep_ensemble' all members' scores are comparable since they share
                data/architecture).
        """
        losses: Dict[str, float] = {}
        for m in self.modalities:
            dim = getattr(self.encoder, self._dim_attr(m))

            loss_m = 0.0
            for member_idx in range(self.num_members):
                if self.cfg.uncertainty_method == "deep_ensemble":
                    member_seed = self.cfg.ensemble_seeds[member_idx]
                    print(f"[Deep Ensemble] {m}: training member {member_idx + 1}/{self.num_members} (seed={member_seed})")
                    # Reseed before (re)building each member so both the random weight
                    # initialization and the DataLoader shuffling order differ across members.
                    seed(member_seed)
                    self.mlps[m][member_idx] = self._build_member(dim)

                cfg_m = MLPTrainConfigs(
                    feature_name=m,
                    trainer=self.mlps[m][member_idx],
                    scaler=self.scalers[m],
                    get_features=self._get_feature_fn(m),
                    fold_dir=self.fold_dir,
                    num_epochs=self.cfg.num_epochs,
                    learning_rate=self.cfg.learning_rate,
                    batch_size=self.cfg.batch_size,
                    save_every=self.cfg.save_every,
                    task_type=self.cfg.task_type,
                    save_trajectory=(self.cfg.uncertainty_method == "ltau"),
                    member_idx=member_idx,
                    num_members=self.num_members,
                )
                loss_m = train_mlp(tr_id, va_id, X_tr, X_va, y_tr, y_va, cfg_m)
            losses[m] = loss_m
        return losses

    def compute_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute per-sample, per-modality uncertainty u_m(x) via MC Dropout or Deep Ensemble.
        Not used when cfg.uncertainty_method == 'ltau' (that path is handled upstream in
        cue/core/cue_runner.py via the LTAU descriptor/KNN pipeline instead).

        Args:
            X (np.ndarray): SMILES array, shape (N,).

        Returns:
            Dict[str, np.ndarray]: {modality: (N,)} predictive variance per modality, in the
                same (possibly normalized) target space the modality MLPs were trained on.
        """
        result: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            f_m = np.array([self._get_feature_fn(m)(s) for s in X], dtype=np.float32)
            f_m = self.scalers[m].transform(f_m)

            if self.cfg.uncertainty_method == "mc_dropout":
                _, var_m = self.mlps[m][0].predict_mc_dropout(f_m, num_samples=self.cfg.mc_samples)
            elif self.cfg.uncertainty_method == "deep_ensemble":
                _, var_m = predict_ensemble([t.model for t in self.mlps[m]], f_m, self.device)
            elif self.cfg.uncertainty_method == "evidential":
                _, var_m = self.mlps[m][0].predict_evidential(
                    f_m, term=self.cfg.evidential_uncertainty
                )
            else:
                raise ValueError(
                    f"compute_uncertainty() is not applicable for uncertainty_method="
                    f"{self.cfg.uncertainty_method!r} (use the LTAU pipeline instead)"
                )

            result[m] = np.clip(var_m, a_min=0.0, a_max=None)
        return result

    def predict(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        uncertainties: Optional[Dict[str, np.ndarray]] = None,
        fusion_policy: str = "softmax",
        temperature: float = 1.0,
        uncertainty_norm: str = "none",
        random_state: Optional[int] = None,
        return_weights: bool = False,
        return_losses: bool = False,
        task_type: str = "regression",
        custom_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]] | np.ndarray:
        """
        Args:
            X (np.ndarray): SMILES array, shape (N,)
            y (Optional[np.ndarray]): Labels array, shape (N,) or (N,1)
            uncertainties (Optional[Dict[str, np.ndarray]]): {modality: (N,)}
            fusion_policy (str): 'softmax' | 'random' | 'uniform' | 'sum' | 'concat' | 'custom'
            temperature (float)
            uncertainty_norm (str): 'none' | 'minmax' | 'zscore'
            random_state (Optional[int])
            return_weights (bool)
            return_losses (bool)
            task_type (str): 'regression' | 'classification'
            custom_weight (Optional[np.ndarray]): Fixed per-modality weight, length == len(modalities),
                required when fusion_policy='custom'.

        Returns:
            np.ndarray or tuple:
                - fused embeddings (N, D) (or (N, M*D) for fusion_policy='concat')
                - weights (N, M) if return_weights=True
                - losses (Dict[str, np.ndarray]) if return_losses=True (sample-wise, keyed loss_{modality})

        Note:
            The fused embedding and downstream-prediction pathway always use ensemble member 0
            (deterministic, dropout inactive). Other members (Deep Ensemble) or stochastic
            samples (MC Dropout) are used only to compute ``uncertainties`` upstream via
            ``compute_uncertainty`` - this pathway is identical across all uncertainty methods.
        """
        modalities = self.modalities
        M = len(modalities)
        n = int(len(X))

        mlp0 = {m: self.mlps[m][0] for m in modalities}

        feats = {m: np.array([self._get_feature_fn(m)(s) for s in X], dtype=np.float32) for m in modalities}
        feats_scaled = {m: self.scalers[m].transform(feats[m]) for m in modalities}
        embs = {m: mlp0[m].get_embeddings(feats_scaled[m]) for m in modalities}

        loss_dict: Optional[Dict[str, np.ndarray]] = None

        if return_losses:
            if y is None:
                loss_dict = {f"loss_{m}": np.full(n, np.nan, dtype=float) for m in modalities}
            else:
                y_flat = np.asarray(y, dtype=float).reshape(-1)
                loss_dict = {}

                if task_type == "classification":
                    # NOTE: compute BCEWithLogits per-sample (needs logits, not probabilities)
                    def _logits_forward(trainer: Trainer, feats_m: np.ndarray, bs: int = 1024) -> np.ndarray:
                        ds = torch.utils.data.TensorDataset(torch.tensor(feats_m, dtype=torch.float32))
                        dl = DataLoader(ds, batch_size=bs, shuffle=False)

                        trainer.model.eval()
                        outs: List[np.ndarray] = []
                        with torch.no_grad():
                            for (inp,) in dl:
                                inp = inp.to(trainer.device)
                                logits = trainer.model(inp)  # (B,1)
                                outs.append(logits.detach().cpu().numpy())
                        return np.vstack(outs).reshape(-1)

                    yt = torch.tensor(y_flat, dtype=torch.float32)
                    for m in modalities:
                        logit_m = _logits_forward(mlp0[m], feats_scaled[m])
                        loss_m = torch.nn.functional.binary_cross_entropy_with_logits(
                            torch.tensor(logit_m, dtype=torch.float32),
                            yt,
                            reduction="none",
                        ).detach().cpu().numpy()
                        loss_dict[f"loss_{m}"] = loss_m.astype(float)

                else:
                    # regression: absolute error per sample (uses prediction)
                    for m in modalities:
                        p_m = mlp0[m].inference(feats_scaled[m], batch_size=1024, task_type="regression").reshape(-1)
                        loss_dict[f"loss_{m}"] = np.abs(y_flat - p_m)

        if fusion_policy == "concat":
            # Concat embeddings (N, M*D). If uncertainties are provided, respect ablation
            # masking (uncertainty >= 1e8).
            if uncertainties is not None:
                masks = {m: (uncertainties[m] < 1e8).astype(np.float32).reshape(-1, 1) for m in modalities}
                for m in modalities:
                    embs[m] = embs[m] * masks[m]
                W = np.concatenate([masks[m] for m in modalities], axis=1)
            else:
                W = np.ones((n, M), dtype=np.float32)

            emb = np.concatenate([embs[m] for m in modalities], axis=1)
        elif uncertainties is None:
            # no uncertainties -> uniform average
            emb = sum(embs[m] for m in modalities) / float(M)
            W = np.ones((n, M), dtype=np.float32) / float(M)
        else:
            # uncertainty-aware fusion
            u = np.stack([uncertainties[m] for m in modalities], axis=1)
            u = UncertaintyNormalizer(uncertainty_norm)(u)

            W = compute_fusion_weights(
                u=u,
                policy=fusion_policy,
                temperature=temperature,
                random_state=random_state,
                custom_weight=custom_weight,
            )
            emb = sum(W[:, [i]] * embs[m] for i, m in enumerate(modalities))

        outputs = [emb]
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
        suffix = lambda idx: f"_m{idx}" if self.num_members > 1 else ""
        for m in self.modalities:
            for idx, trainer in enumerate(self.mlps[m]):
                trainer.save_model(os.path.join(ckpt_dir, f"mlp_{m}{suffix(idx)}.pth"))
            joblib.dump(self.scalers[m], os.path.join(ckpt_dir, f"scaler_{m}.pkl"))

    def load(self) -> None:
        """
        Args:
            None

        Returns:
            None
        """
        ckpt_dir = os.path.join(self.fold_dir, "checkpoints")
        suffix = lambda idx: f"_m{idx}" if self.num_members > 1 else ""
        for m in self.modalities:
            for idx, trainer in enumerate(self.mlps[m]):
                trainer.load_model(os.path.join(ckpt_dir, f"mlp_{m}{suffix(idx)}.pth"))
            self.scalers[m] = joblib.load(os.path.join(ckpt_dir, f"scaler_{m}.pkl"))

    def get_modality_descriptors(
        self,
        X: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extract modality-specific descriptors (hidden layer outputs) from ensemble member 0.

        Args:
            X (np.ndarray): SMILES array, shape (N,)

        Returns:
            Dict[str, np.ndarray]: {modality: (N, D)}
        """
        result: Dict[str, np.ndarray] = {}
        for m in self.modalities:
            f_m = np.array([self._get_feature_fn(m)(s) for s in X], dtype=np.float32)
            f_m = self.scalers[m].transform(f_m)
            result[m] = self.mlps[m][0].get_embeddings(f_m)
        return result
