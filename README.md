# CUE: A Chemical Uncertainty-aware Embedding Framework for Multi-modal Drug Selectivity Prediction

CUE (**Chemical Uncertainty-aware Embedding**) is a multi-modal molecular representation framework for drug selectivity prediction. CUE integrates molecular fingerprints and 2D molecular structure image embeddings, and combines them through **Loss Trajectory Analysis for Uncertainty (LTAU)**-based weighted fusion. The fusion weights are determined from modality-specific uncertainty, allowing the model to adaptively emphasize the more reliable modality for each molecule.

## Getting Started

### 1. Environment Setup

Create the Conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate cue
```

### 2. Download for molecular image feature extractor (Important)

**Required**: CUE uses the pretrained MolScribe encoder to extract molecular image features from 2D molecular structure images. Download the `swin_base_char_aux_1m680k.pth` checkpoint from the [MolScribe repository](https://github.com/thomas0809/MolScribe), and place it at:

```text
./cue/molscribe/ckpts/swin_base_char_aux_1m680k.pth
```

### 3. Training and Evaluation

Run the 5-fold cross-validation pipeline with:

```bash
bash scripts/run.sh
```

During training, a `logs` directory is automatically created. Log files are organized by dataset name, target column, and experiment version.

```text
logs/{DATA_NAME}/{TARGET_COL}/{VERSION}/{TARGET_COL}.log
```


## Output Directory Structure

After training, all output artifacts are saved under:

```text
results/{DATA_NAME}/{TARGET_COL}/{VERSION}/CUE/
```

<details>
<summary><strong>Details of output directory</strong></summary>

```text
results/DIS/PI/v1/CUE/
├── config.json                  # Training and evaluation configuration
├── predictions_all.csv          # Test predictions concatenated across all five folds
├── metrics.csv                  # Cross-validation metrics summarized across folds
├── fold1/                       # Fold-specific output directory
│   ├── predictions.csv          # Test predictions for the current fold
│   ├── checkpoints/             # Modality-specific MLP checkpoints and feature scalers
│   │   ├── mlp_fp.pth           # FP modality-specific MLP checkpoint
│   │   ├── mlp_img.pth          # IMG modality-specific MLP checkpoint
│   │   ├── scaler_fp.pkl        # Feature scaler fitted on FP features
│   │   └── scaler_img.pkl       # Feature scaler fitted on IMG features
│   ├── ltau/                    # LTAU estimator artifacts
│   │   ├── fp/                  # FP-specific LTAU estimator directory
│   │   │   └── estimator.pkl    # LTAU estimator for the FP modality
│   │   └── img/                 # IMG-specific LTAU estimator directory
│   │       └── estimator.pkl    # LTAU estimator for the IMG modality
│   ├── fusion/                  # Sample-wise uncertainty and fusion outputs
│   │   ├── train_fusion.csv     # Training-set uncertainty values, fusion weights, and losses
│   │   ├── va_fusion.csv        # Validation-set uncertainty values, fusion weights, and losses
│   │   └── test_fusion.csv      # Test-set uncertainty values, fusion weights, and losses
│   ├── emb/                     # Unified embedding arrays
│   │   ├── train_emb.npy        # Unified embeddings for the training set
│   │   ├── va_emb.npy           # Unified embeddings for the validation set
│   │   └── test_emb.npy         # Unified embeddings for the test set
│   └── regressors/              # Downstream selectivity regressor artifacts
│       ├── MLP.pth              # Downstream MLP regressor checkpoint
│       ├── label_normalizer.pkl # Target label normalizer
│       └── loss_mlp.png         # Training and validation loss curve of the downstream MLP
├── fold2/                       # Same structure as fold1
├── fold3/                       # Same structure as fold1
├── fold4/                       # Same structure as fold1
└── fold5/                       # Same structure as fold1
```
</details>

---

## Prediction on New SMILES

To predict selectivity scores for new molecules represented by SMILES strings, run:

```bash
bash scripts/predict.sh
```

The prediction script loads the trained modality-specific MLP encoders, LTAU estimators, feature scalers, and downstream MLP regressor from the saved checkpoint directory.

### Input CSV Format

The input file should be provided as a CSV file containing a `SMILES` column. An additional molecule identifier column, such as `Compound`, `ID`, or `compound_chembl_id`, may be included for tracking individual compounds. If no identifier column is provided, molecule identifiers are automatically assigned using the `CID_{index}` format.

```csv
compound_chembl_id,SMILES
CHEMBL10,C[S+]([O-])c1ccc(-c2nc(-c3ccc(F)cc3)c(-c3ccncc3)[nH]2)cc1
CHEMBL101253,Clc1ccc(Nc2nnc(Cc3ccncc3)c3ccccc23)cc1
CHEMBL103667,Cc1ccc(-n2nc(C(C)(C)C)cc2NC(=O)Nc2ccc(OCCN3CCOCC3)c3ccccc23)cc1
```

> The training dataset format includes additional selectivity target columns (e.g., `PI`, `Ssel`, `WS2`, `RS2`), but these are not required for prediction — only the `SMILES` column is used.

### Configuring `scripts/predict.sh`

Before running prediction, configure the following variables in `scripts/predict.sh`:

| Variable          | Description                                                                         |
| :---------------- | :---------------------------------------------------------------------------------- |
| `INPUT_FILE`      | Path to the input CSV file containing molecular SMILES strings                      |
| `SMILES_COL`      | Name of the SMILES column in the input CSV file                                     |
| `TARGET_COLS`     | Selectivity metrics to predict; these should match the targets used during training |
| `DATA_NAMES`      | Dataset name associated with the trained model, such as `DIS` or `INH`              |
| `VERSION`         | Experiment version corresponding to the trained checkpoint                          |
| `MODEL_NAME`      | Model directory name; the default value is `CUE`                                    |
| `OUTPUT_BASE`     | Output directory for saving prediction results                                      |
| `DEVICE`          | Computing device used for inference, such as `cuda:0` or `cpu`                      |
| `SPECIFIC_FOLD`   | Optional argument for performing prediction with a single fold checkpoint           |


### Output

For each `(DATA_NAME, TARGET_COL)` pair, the prediction script generates one CSV file:

```text
prediction_{DATA_NAME}_{TARGET_COL}_{VERSION}_{MODEL_NAME}.csv
```

The output file contains the predicted selectivity scores for the input molecules, together with fold-wise predictions, modality-specific uncertainty values (`u_fp`, `u_img`), and modality fusion weights (`w_fp`, `w_img`).

---

## Configuration Parameters

<details>
<summary><strong>Description of training configuration</strong></summary>

Training and evaluation settings are defined in `scripts/run.sh`.

### General Settings

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `VERSION` | `"v1"` | Experiment version identifier |
| `TARGET_COLS` | `("PI" "Ssel" "WS2" "RS2")` | Selectivity metrics used as prediction targets |
| `DATA_NAMES` | `("DIS" "INH")` | Dataset names used for training and evaluation |
| `LABEL_MODE` | `"none"` | Label normalization method: `zscore`, `robust`, `minmax`, or `none` |
| `DEVICE` | `"cuda:0"` | Computing device |
| `SEED` | `2025` | Random seed for reproducibility |

### CUE and LTAU Settings

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `LTAU_K` | `5` | Number of nearest neighbors used for LTAU-based uncertainty estimation |
| `LTAU_BINS` | `5000` | Number of histogram bins used to construct empirical error distributions |
| `TEMPERATURE` | `1.0` | Optional temperature parameter for uncertainty-to-weight scaling |

### Modality-specific MLP Pretraining Settings

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `HIDDEN_DIM` | `32` | Latent embedding dimension of each modality-specific MLP |
| `NUM_EPOCHS` | `1000` | Number of pretraining epochs |
| `LEARNING_RATE` | `1e-4` | Learning rate for modality-specific MLP pretraining |
| `BATCH_SIZE` | `256` | Batch size for modality-specific MLP pretraining |
| `SAVE_EVERY` | `1` | Checkpoint and trajectory save interval |

### Downstream Regressor Settings

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `REG_HIDDEN_DIM_1` | `16` | First hidden layer dimension of the downstream MLP |
| `REG_HIDDEN_DIM_2` | `16` | Second hidden layer dimension of the downstream MLP |
| `REG_HIDDEN_DIM_3` | `8` | Third hidden layer dimension of the downstream MLP |
| `REG_NUM_EPOCHS` | `500` | Number of downstream regressor training epochs |
| `REG_LEARNING_RATE` | `1e-3` | Learning rate for the downstream MLP regressor |
| `REG_BATCH_SIZE` | `256` | Batch size for downstream regressor training |
| `REG_SAVE_EVERY` | `1` | Checkpoint save interval for the downstream MLP |

</details>

---

## Details of Code Variable

<details>
<summary><strong>Variables</strong></summary>

| Variable / Path | Paper Notation | Description |
| :--- | :--- | :--- |
| `mlp_fp` | $f_{FP}$ | Modality-specific MLP for the FP modality |
| `mlp_img` | $f_{IMG}$ | Modality-specific MLP for the IMG modality |
| `X`, `X_tr`, `X_va`, `X_te` | $\{x_i\}$, $\mathcal{S}_{tr}$, $\mathcal{S}_{va}$, $\mathcal{S}_{te}$ | Molecular input sets represented by SMILES strings |
| `y`, `y_tr`, `y_va`, `y_te` | $\{y_i\}$ | Selectivity score vectors |
| `raw_fp` | — | Raw molecular fingerprint feature vector |
| `raw_img` | — | Raw molecular image feature vector |
| `emb_fp` | $h_{FP}$ | FP modality-specific embedding |
| `emb_img` | $h_{IMG}$ | IMG modality-specific embedding |
| `uni`, `uni_tr`, `uni_va`, `uni_te` | $h_{uni}$ | Unified embedding obtained by uncertainty-aware fusion |
| `W`, `W_tr`, `W_va`, `W_te` | $[w_{FP}, w_{IMG}]$ | Modality fusion weights |
| `u`, `u_tr`, `u_va`, `u_te` | $[u_{FP}, u_{IMG}]$ | Modality-specific uncertainty values |
| `loss_trajectories` | $\mathcal{T}=\{T_1,\ldots,T_N\}$ | Sample-wise loss trajectories collected during modality-specific MLP training |
| `loss_trajectory_i` | $T_i$ | Loss trajectory of sample $i$ |
| `sample_error` | $\epsilon_{i,t}$ | Prediction error of sample $i$ at epoch $t$ |
| `shared_bin_edges` | $edges=\{a_0,a_1,\ldots,a_B\}$ | Shared histogram bin edges used for empirical error distribution construction |
| `reference_pdfs` | $\mathcal{P}=\{p_1,\ldots,p_N\}$ | Reference empirical error distributions |
| `reference_embeddings` | $\mathcal{H}=\{h_1,\ldots,h_N\}$ | Reference embeddings used for nearest-neighbor search |
| `neighbor_indices` | $\mathcal{N}(x)$ | Indices of the k-nearest reference embeddings for query molecule $x$ |
| `neighbor_ref_pdf` | $\{p_i\}_{i\in\mathcal{N}(x)}$ | Reference empirical error distributions associated with the nearest neighbors |
| `query_pdf` | $\hat{p}$ | Aggregated empirical error distribution for a query molecule |

</details>

---

## Authors

- [Jin Hyuk Kim](https://scholar.google.com/citations?user=8ly72dcAAAAJ&hl=ko)
- Gyeong Hwan Kim
- Hyeon Jun Park
- [Jonghwan Choi](https://scholar.google.com/citations?user=4xSw9C8AAAAJ&hl=ko)

---

## Contact

For questions or issues, please contact:

- **Email**: rlawlsgurjh@gmail.com
