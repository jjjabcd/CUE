"""
ConforFormer conformation-embedding encoder.

Wraps the vendored ConforFormer/unimol `contrast` architecture (a fine-tuned
Uni-Mol model) to turn a single SMILES string into a 512-d 3D-conformation CLS
embedding. Only ONE forward pass on a single RDKit-generated conformer is used
here (not the dual conformer/contrastive-pair setup ConforFormer was trained
with) since we only need one embedding per molecule.

Requires `unicore` (see environment.yml) - raises ImportError with a clear
message if unavailable, so the smi/gnn/fp/img modalities keep working
independently of this one.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

_UNIMOL_SRC_DIR = Path(__file__).resolve().parent / "unimol_src"
_CKPT_DIR = Path(__file__).resolve().parent / "ckpts"

HF_REPO_ID = "ConforFormer/ConforFormer"
CKPT_FILENAME = "ConforFormer.pt"
DICT_FILENAME = "OMOL_full_dict.txt"


def _ensure_unimol_importable() -> None:
    """Put the vendored (unmodified) ConforFormer `unimol` package on sys.path."""
    src = str(_UNIMOL_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def _download_weights() -> Tuple[str, str]:
    from huggingface_hub import hf_hub_download

    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CKPT_FILENAME, local_dir=str(_CKPT_DIR))
    dict_path = hf_hub_download(repo_id=HF_REPO_ID, filename=DICT_FILENAME, local_dir=str(_CKPT_DIR))
    return ckpt_path, dict_path


class ConforFormerEncoder:
    """
    Args:
        device: torch device to run inference on.
        ckpt_path/dict_path: local overrides; if omitted, both are downloaded
            (once, then cached) from the ConforFormer/ConforFormer HF repo.
    """

    embed_dim = 512

    def __init__(self, device=None, ckpt_path: Optional[str] = None, dict_path: Optional[str] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _ensure_unimol_importable()

        try:
            from unicore.data import Dictionary
            from unicore import checkpoint_utils
        except ImportError as e:
            raise ImportError(
                "Conformation embeddings require `unicore`. A plain `pip install` always "
                "tries to build its optional fused CUDA kernels and fails without nvcc; "
                "install the Python-only build instead (see environment.yml):\n"
                "  git clone --branch torch1.x https://github.com/dptech-corp/Uni-Core.git\n"
                "  cd Uni-Core && python setup.py install --disable-cuda-ext\n"
                "The smi/gnn/fp/img modalities do not need this."
            ) from e

        from unimol.models.unimol_contrast import UniMolModel_contrast

        if ckpt_path is None or dict_path is None:
            dl_ckpt, dl_dict = _download_weights()
            ckpt_path = ckpt_path or dl_ckpt
            dict_path = dict_path or dl_dict

        # Dictionary + [MASK] symbol, mirroring UniMolContrast.__init__
        self.dictionary = Dictionary.load(dict_path)
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self.num_types = len(self.dictionary)

        # `mode="infer"` makes forward() return (encoder_rep, encoder_pair_rep, logits)
        # unconditionally; every other arch hyperparameter is filled in by
        # `base_architecture()` (called inside the model's own __init__) with the
        # same defaults ConforFormer's own inference scripts rely on (they don't
        # pass --encoder-layers etc. either): layers=15, embed_dim=512, ffn=2048,
        # heads=64, dropout=0.1, activation_fn=gelu, max_seq_len=512,
        # masked_*_loss=-1.0 (so lm_head/pair2coord_proj/dist_head aren't built,
        # and their checkpoint weights are simply dropped by strict=False below).
        args = argparse.Namespace(mode="infer")
        model = UniMolModel_contrast(args, self.dictionary)

        state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path)
        model.load_state_dict(state["model"], strict=False)

        self.model = model.to(self.device)
        self.model.eval()

    def _generate_conformer(self, smi: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """RDKit ETKDGv3 + MMFF/UFF optimization -> (atom_symbols, coords[N,3])."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
            if cid < 0:
                return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, confId=cid)
            except Exception:
                pass

        conf = mol.GetConformer(cid)
        coords = conf.GetPositions().astype(np.float32)
        atoms = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        return atoms, coords

    def _build_model_inputs(self, atoms: np.ndarray, coords: np.ndarray):
        """
        Reproduces, for a single molecule, the exact preprocessing chain
        `unimol/tasks/unimol_contrast.py::UniMolContrast.load_dataset` applies
        with `only_polar=0` (RemoveHydrogenDataset -> NormalizeDataset ->
        Tokenize -> Edge/Distance), without needing the LMDB/task/collation
        machinery.
        """
        # only_polar=0 -> remove_hydrogen=True: drop ALL hydrogens
        mask = atoms != "H"
        atoms = atoms[mask]
        coords = coords[mask]

        # NormalizeDataset: center coordinates
        coords = coords - coords.mean(axis=0, keepdims=True)

        # TokenizeDataset: direct atom-symbol -> dictionary index lookup
        token_ids = [self.dictionary.index(a) for a in atoms]
        src_tokens = [self.dictionary.bos()] + token_ids + [self.dictionary.eos()]
        src_tokens = torch.tensor(src_tokens, dtype=torch.long)

        # Prepend/append 0,0,0 coordinate for BOS/EOS
        src_coord = np.vstack(
            [np.zeros((1, 3), dtype=np.float32), coords.astype(np.float32), np.zeros((1, 3), dtype=np.float32)]
        )
        src_coord = torch.from_numpy(src_coord)

        # Pairwise euclidean distance matrix (DistanceDataset equivalent)
        src_distance = torch.cdist(src_coord, src_coord)

        # EdgeTypeDataset: token_i * num_types + token_j
        src_edge_type = src_tokens.view(-1, 1) * self.num_types + src_tokens.view(1, -1)

        return (
            src_tokens.unsqueeze(0).to(self.device),
            src_distance.unsqueeze(0).to(self.device),
            src_coord.unsqueeze(0).to(self.device),
            src_edge_type.unsqueeze(0).to(self.device),
        )

    def get_conf_features(self, smi: str) -> np.ndarray:
        """
        Args:
            smi: SMILES string.

        Returns:
            np.ndarray of shape (512,). Zeros on any failure (invalid SMILES,
            failed conformer embedding, etc.), matching the other MolEncoder
            feature getters' fallback behavior.
        """
        try:
            generated = self._generate_conformer(smi)
            if generated is None:
                return np.zeros(self.embed_dim, dtype=np.float32)
            atoms, coords = generated
            src_tokens, src_distance, src_coord, src_edge_type = self._build_model_inputs(atoms, coords)

            with torch.no_grad():
                encoder_rep, _, _ = self.model(
                    src_tokens=src_tokens,
                    src_distance=src_distance,
                    src_coord=src_coord,
                    src_edge_type=src_edge_type,
                    features_only=True,
                )
            cls_emb = encoder_rep[:, 0, :].squeeze(0).detach().cpu().numpy()
            return cls_emb.astype(np.float32)
        except Exception as e:
            warnings.warn(f"[ConforFormerEncoder] failed for SMILES '{smi}': {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)

    def get_conf_features_all(self, smis: List[str]) -> np.ndarray:
        """
        Args:
            smis: List of SMILES strings.

        Returns:
            np.ndarray of shape (N, 512).
        """
        if not smis:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        return np.stack([self.get_conf_features(s) for s in smis]).astype(np.float32)
