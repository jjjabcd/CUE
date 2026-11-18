import os
import sys
import importlib
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
from PIL import Image
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles
from transformers import AutoModelForMaskedLM, AutoTokenizer
from cue.molscribe.model import Encoder

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Asking to truncate to max_length.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _ensure_mole_aliases():
    """
    Hydra/omegaconf targets in the vendored MolE config reference top-level
    "mole.*" (see cue/mole/cli/mole_predict.py's `encode()` overrides), so we
    alias that name to our vendored cue.mole package.
    """
    if "mole" not in sys.modules:
        sys.modules["mole"] = importlib.import_module("cue.mole")

    alias_map = {
        "mole.training": "cue.mole.training",
        "mole.training.data": "cue.mole.training.data",
        "mole.training.data.data_modules": "cue.mole.training.data.data_modules",
        "mole.training.data.datasets": "cue.mole.training.data.datasets",
        "mole.training.data.utils": "cue.mole.training.data.utils",
        "mole.training.models": "cue.mole.training.models",
        "mole.training.models.base": "cue.mole.training.models.base",
        "mole.training.models.encoder": "cue.mole.training.models.encoder",
    }

    for k, v in alias_map.items():
        if k not in sys.modules:
            sys.modules[k] = importlib.import_module(v)


class Args:
    def __init__(self):
        self.encoder = 'swin_base_patch4_window12_384'
        self.use_checkpoint = False

class MolEncoder:
    def __init__(self, device=None):
        ## torch device
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ## fp - Morgan
        self.feature_dim_fp = 2048
        self.morgan = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        
        
        ## img - MolScribe
        self.feature_dim_img = 1024
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ocsr = self._load_pretrained_ocsr()
        for param in self.ocsr.parameters():
            param.requires_grad = False

        ## txt - SMILES (ChemBERTa)
        self.feature_dim_smi = 768
        self.chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-100M-MLM").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM")

        ## graph - GNN (MolE)
        self.feature_dim_gnn = 768
        _ensure_mole_aliases()

        from cue.mole.cli.mole_predict import encode
        self._mole_encode = encode
        self.mole_ckpt_path, self._mole_accelerator = self._load_pretrained_gnn()

        ## graph - GNN (MGSSL, alternate backend alongside MolE; lazy: only constructed
        ## on first use)
        self.feature_dim_gnn_mgssl = 300
        self._mgssl_encoder = None

        ## conformation - ConforFormer (lazy: only constructed on first use so a
        ## missing `unicore` install doesn't break the other modalities)
        self.feature_dim_conf = 512
        self._conformer_encoder = None


    def _load_pretrained_ocsr(self):
        encoder = Encoder(Args(), pretrained=False)
        
        ## load states
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'molscribe', 'ckpts', 'swin_base_char_aux_1m680k.pth')
        states = torch.load(filepath, map_location=self.device)['encoder'] #, weights_only=True)['encoder']
        
        ## safe load
        def _remove_prefix(state_dict):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        missing_keys, unexpected_keys = encoder.load_state_dict(_remove_prefix(states), strict=False)
        #if missing_keys:
        #    print('Missing keys: ' + str(missing_keys))
        #if unexpected_keys:
        #    print('Unexpected keys: ' + str(unexpected_keys))
        
        ## mode:eval
        encoder.eval()
        encoder.trainable = False
        return encoder.to(self.device)
        
        
    def get_fpt_features(self, smi):
        mol = MolFromSmiles(smi)
        if mol:
            fp = self.morgan.GetFingerprint(mol)
            return np.array(fp.ToList())
        else:
            return np.zeros(2048, dtype=int)

    def _load_pretrained_gnn(self):
        """
        Args:
            None

        Returns:
            ckpt_path: Absolute path to the vendored MolE checkpoint file.
            accelerator: "gpu" or "cpu" depending on self.device.
        """
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "mole",
            "ckpts",
            "MolE_GuacaMol_27113.ckpt",
        )
        ckpt_path = os.path.abspath(ckpt_path)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[MolE] checkpoint not found: {ckpt_path}")

        accelerator = "gpu" if str(self.device).startswith("cuda") else "cpu"

        # PyTorch >=2.6 safety: allowlist numpy reconstruct used in some ckpt pickles
        try:
            import numpy as _np
            if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct])
        except Exception:
            pass

        return ckpt_path, accelerator

    def _get_devices(self):
        if self.device.type == "cuda":
            return [self.device.index] if self.device.index is not None else [0]
        return 1

    def get_gnn_features(self, smi):
        """
        Args:
            smi: Input SMILES string.

        Returns:
            MolE graph embedding as a numpy array with shape (768,).
        """
        if smi is None or len(smi) == 0:
            return np.zeros(self.feature_dim_gnn, dtype=np.float32)

        try:
            emb = self._mole_encode(
                smiles=[smi],
                pretrained_model=self.mole_ckpt_path,
                batch_size=1,
                num_workers=0,
                accelerator=self._mole_accelerator,
                devices=self._get_devices(),
            )
            emb = np.asarray(emb, dtype=np.float32)

            if emb.ndim == 2 and emb.shape[0] == 1 and emb.shape[1] == self.feature_dim_gnn:
                return emb[0]

            flat = emb.reshape(-1)
            if flat.shape[0] == self.feature_dim_gnn:
                return flat

            return np.zeros(self.feature_dim_gnn, dtype=np.float32)

        except Exception:
            return np.zeros(self.feature_dim_gnn, dtype=np.float32)

    def get_gnn_features_all(self, smis: List[str]) -> np.ndarray:
        """
        Args:
            smis: List of SMILES strings.

        Returns:
            MolE graph embeddings as a numpy array with shape (N, 768).
        """
        if not smis:
            return np.empty((0, self.feature_dim_gnn), dtype=np.float32)

        try:
            emb = self._mole_encode(
                smiles=smis,
                pretrained_model=self.mole_ckpt_path,
                batch_size=32,
                num_workers=4,
                accelerator=self._mole_accelerator,
                devices=self._get_devices(),
            )
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 2 and emb.shape[1] == self.feature_dim_gnn:
                return emb

            if emb.reshape(-1).shape[0] == len(smis) * self.feature_dim_gnn:
                return emb.reshape(len(smis), self.feature_dim_gnn)

            print(f"[Warning] MolE returned unexpected shape {emb.shape} for {len(smis)} smiles. Returning zeros.")
            return np.zeros((len(smis), self.feature_dim_gnn), dtype=np.float32)

        except Exception as e:
            print(f"[Error] MolE encoding failed: {e}")
            return np.zeros((len(smis), self.feature_dim_gnn), dtype=np.float32)

    def get_smi_features_both(self, smi, padding=True):
        """
        Runs ChemBERTa once and returns both CLS and mean-pooled embeddings.

        Args:
            smi: Input SMILES string.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (cls_emb, mean_emb), each shape (768,)
        """
        max_length = self.chemberta.config.max_position_embeddings

        encoded = self.tokenizer(
            smi,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length" if padding else False,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.chemberta(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, 768)
            cls_emb = last_hidden[:, 0, :]
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            mean_emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return cls_emb.squeeze(0).cpu().numpy(), mean_emb.squeeze(0).cpu().numpy()

    def get_smi_features(self, smi, pooling="mean", padding=True):
        """
        Args:
            smi: Input SMILES string.
            pooling: "mean" (attention-mask-weighted mean) or "cls" ([CLS] token).

        Returns:
            np.ndarray of shape (768,)
        """
        cls_emb, mean_emb = self.get_smi_features_both(smi, padding=padding)
        return cls_emb if pooling == "cls" else mean_emb

    @property
    def mgssl_encoder(self):
        """Lazily constructs the MGSSLEncoder (alternate GNN backend alongside MolE)."""
        if self._mgssl_encoder is None:
            from cue.mgssl.encoder import MGSSLEncoder
            self._mgssl_encoder = MGSSLEncoder(device=self.device)
        return self._mgssl_encoder

    def get_gnn_features_mgssl(self, smi: str) -> np.ndarray:
        """
        Args:
            smi: Input SMILES string.

        Returns:
            MGSSL (motif-pretrained GIN) graph embedding, shape (300,).
        """
        return self.mgssl_encoder.get_gnn_features(smi)

    def get_gnn_features_mgssl_all(self, smis: List[str]) -> np.ndarray:
        """
        Args:
            smis: List of SMILES strings.

        Returns:
            MGSSL graph embeddings as a numpy array with shape (N, 300).
        """
        return self.mgssl_encoder.get_gnn_features_all(smis)

    @property
    def conformer_encoder(self):
        """Lazily constructs the ConforFormerEncoder (requires `unicore`)."""
        if self._conformer_encoder is None:
            from cue.conformer.encoder import ConforFormerEncoder
            self._conformer_encoder = ConforFormerEncoder(device=self.device)
        return self._conformer_encoder

    def get_conf_features(self, smi: str) -> np.ndarray:
        """
        Args:
            smi: Input SMILES string.

        Returns:
            ConforFormer 3D-conformation CLS embedding, shape (512,).
        """
        return self.conformer_encoder.get_conf_features(smi)

    def get_conf_features_all(self, smis: List[str]) -> np.ndarray:
        """
        Args:
            smis: List of SMILES strings.

        Returns:
            ConforFormer embeddings as a numpy array with shape (N, 512).
        """
        return self.conformer_encoder.get_conf_features_all(smis)



            

    def get_img_features(self, smi, rotation_angle=0):
        mol = MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol)
            if rotation_angle % 360 > 0:
                img = self._rotate_image(img, rotation_angle)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature, _ = self.ocsr(img_tensor)
                feature_avg_pooled = torch.mean(feature, dim=1).squeeze()
            return feature_avg_pooled.cpu().numpy()
        else:
            return np.zeros(1024)
    

    def _rotate_image(self, img, angle):
        cv2_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        (h, w) = cv2_image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(cv2_image, M, (w, h))
        return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))