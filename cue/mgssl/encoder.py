"""
MGSSL (Motif-based Graph Self-Supervised Learning, Zhang et al. NeurIPS'21) graph embedding
encoder. Wraps the vendored `GNN` (motif-pretrained GIN encoder, cue/mgssl/gnn_model.py) plus
the exact SMILES-to-graph featurization used to train/pretrain it
(mol_to_graph_data_obj_simple, vendored below from MGSSL/finetune/loader.py) to turn a SMILES
string into a 300-d graph embedding (mean-pooled node representations).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from .gnn_model import GNN

_CKPT_PATH = Path(__file__).resolve().parent / "ckpts" / "pretrained.pth"

# Matches MGSSL/finetune/loader.py::allowable_features (atom/bond feature vocab the
# pretrained checkpoint's embeddings were trained against) and
# MGSSL/finetune/finetune.py's default hyperparameters (num_layer=5, emb_dim=300, JK="last",
# gnn_type="gin") - both must match exactly for the checkpoint's weights to be meaningful.
_ALLOWABLE_ATOMIC_NUM = list(range(1, 119))
_ALLOWABLE_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
_ALLOWABLE_BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_ALLOWABLE_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

NUM_LAYER = 5
EMB_DIM = 300
JK = "last"
GNN_TYPE = "gin"


def _mol_to_graph_data_obj(mol) -> Data:
    """Vendored from MGSSL/finetune/loader.py::mol_to_graph_data_obj_simple (unchanged)."""
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [
            _ALLOWABLE_ATOMIC_NUM.index(atom.GetAtomicNum()),
            _ALLOWABLE_CHIRALITY.index(atom.GetChiralTag()),
        ]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                _ALLOWABLE_BONDS.index(bond.GetBondType()),
                _ALLOWABLE_BOND_DIRS.index(bond.GetBondDir()),
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MGSSLEncoder:
    """
    Args:
        device: torch device to run inference on.
        ckpt_path: local override for the pretrained checkpoint path.
    """

    embed_dim = EMB_DIM

    def __init__(self, device=None, ckpt_path: Optional[str] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gnn = GNN(num_layer=NUM_LAYER, emb_dim=EMB_DIM, JK=JK, drop_ratio=0, gnn_type=GNN_TYPE)
        state = torch.load(str(ckpt_path or _CKPT_PATH), map_location="cpu")
        self.gnn.load_state_dict(state)

        self.gnn = self.gnn.to(self.device)
        self.gnn.eval()

    def _mol_to_data(self, smi: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return _mol_to_graph_data_obj(mol)

    def get_gnn_features(self, smi: str) -> np.ndarray:
        """
        Args:
            smi: SMILES string.

        Returns:
            np.ndarray of shape (300,). Zeros on failure (invalid SMILES, etc.).
        """
        data = self._mol_to_data(smi)
        if data is None:
            return np.zeros(self.embed_dim, dtype=np.float32)

        try:
            batch = Batch.from_data_list([data]).to(self.device)
            with torch.no_grad():
                node_rep = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
                graph_rep = global_mean_pool(node_rep, batch.batch)
            return graph_rep.squeeze(0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            return np.zeros(self.embed_dim, dtype=np.float32)

    def get_gnn_features_all(self, smis: List[str], batch_size: int = 128) -> np.ndarray:
        """
        Args:
            smis: List of SMILES strings.
            batch_size: molecules per forward pass.

        Returns:
            np.ndarray of shape (N, 300).
        """
        if not smis:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        out = np.zeros((len(smis), self.embed_dim), dtype=np.float32)
        datas = [self._mol_to_data(s) for s in smis]

        valid_idx = [i for i, d in enumerate(datas) if d is not None]
        for start in range(0, len(valid_idx), batch_size):
            chunk_idx = valid_idx[start:start + batch_size]
            try:
                batch = Batch.from_data_list([datas[i] for i in chunk_idx]).to(self.device)
                with torch.no_grad():
                    node_rep = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
                    graph_rep = global_mean_pool(node_rep, batch.batch)
                out[chunk_idx] = graph_rep.detach().cpu().numpy().astype(np.float32)
            except Exception:
                pass  # leave zeros for this chunk

        return out
