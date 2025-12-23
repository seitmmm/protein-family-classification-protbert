# Protein Family Classification using ProtBERT Embeddings + Ensembles

A reproducible pipeline for **multi-class protein family classification** using **ProtBERT (ProtTrans) embeddings** and multiple downstream classifiers (**Random Forest, MLP, KAN-inspired, soft voting, stacking meta-ensemble**). The repository also includes **representation analysis** (t-SNE/UMAP, silhouette, family-distance heatmaps) and **mutation robustness** evaluation to test prediction stability under sequence perturbations. A lightweight **web app** (`web/app.py`) provides an interactive interface for inference and robustness testing.

---

## Key Features

- **Data**: UniProt FASTA sequences for **8 families** (balanced sampling)
- **Embeddings**: ProtBERT → **1024D** vectors (`embeddings/X.npy`, `embeddings/Y.npy`)
- **Models**:
  - Random Forest (`rf.pkl`)
  - MLP variants (`mlp.pt`, `mlp_v2.pt`)
  - KAN-inspired models (`kan.pt`, `kan_smooth.pt`, `hybrid_kan.pt`)
  - **Soft ensemble** + **stacked meta-ensemble** (`meta_ensemble_lr.pkl`)
  - Optional experiments: **XGBoost**, **CNN1D** (`cnn1d.pt`)
  - Optional: **Two-stage** verifier pipeline (`two_stage_ensemble.pkl`)
- **Evaluation & analysis**:
  - Confusion matrices and metrics tables
  - **t-SNE** + family-highlight plots
  - **UMAP** + family distance heatmap
  - **Silhouette scores**
- **Robustness**:
  - Random mutation stress testing with curves & saved `.npz` results

---

## Protein Families (8 classes)

- `gpcr`
- `ion_channel`
- `abc_transporter`
- `kinase`
- `cytochrome_p450`
- `immunoglobulin`
- `ribosomal_protein`
- `zinc_finger`

---

## Repository Layout

High-level overview of important folders:

- `data/`
  - `fasta/` — FASTA files per family
  - `processed/` — processed labels/sequences (`label_map.json`, `labels.npy`, `sequences.npy`)
  - `prepare_dataset.py` — build dataset from FASTA
  - `generate_embeddings.py` — generate ProtBERT embeddings
- `embeddings/`
  - `X.npy` — embeddings (N × 1024)
  - `Y.npy` — labels (N,)
- `models/`
  - training scripts (`train_*`)
  - saved checkpoints (`*.pt`, `*.pkl`)
  - `meta/` and `binary/` artifacts for ensemble/two-stage logic
- `results/`
  - confusion matrices and final score tables
- `logs_tsne/`, `analysis_umap/`, `analysis_silhouette/`
  - analysis plots
- `logs_mutations/`
  - mutation robustness plots + cached results
- `web/`
  - `app.py` — web interface entrypoint
- `utils/`
  - `embedder.py` — embedding helper
  - `mutations.py` — mutation utilities
  - `esmfold_wrapper.py` — optional structure wrapper (if enabled)
- top-level scripts:
  - `tsne_protbert.py`, `umap_and_heatmap.py`, `silhouette_scores.py`
  - `mutation.py`, `mutation_stress_test_meta.py`, `mutations_web.py`

---

## Large Files (Google Drive)

Due to GitHub limits, large artifacts are provided via Google Drive:

**Embeddings, models, and FASTA files:**  
https://drive.google.com/drive/folders/1t0hucDAnO8aRfAY3T_agfsaWedPjR_Wz?usp=sharing

### Expected placement after downloading

Put the downloaded content into the same folder structure as in the repo:

- FASTA → `data/fasta/*.fasta`
- Embeddings → `embeddings/X.npy`, `embeddings/Y.npy`
- Models → `models/*.pt`, `models/*.pkl`, and `models/binary/`, `models/meta/` if provided

If you download a zip, extract it so these paths match exactly.

---

## Installation

### 1) Create environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

### 2) Install requirements

pip install -r requirements.txt
If you plan to regenerate embeddings with GPU, install the CUDA-compatible PyTorch build from the official PyTorch site.

Quick Start (Run Web App)

The interactive app lives in web/app.py.

### 1) Ensure required artifacts exist

Before running, confirm you have:

embeddings/X.npy and embeddings/Y.npy (optional for pure inference, but used in some utilities)

trained models, at minimum:

models/rf.pkl

models/mlp_v2.pt (or models/mlp.pt)

models/kan.pt (or models/kan_smooth.pt)

models/meta_ensemble_lr.pkl (for stacking)

data/processed/label_map.json (label decoding)


### 2) Run the app

From the repository root:

python web/app.py


Then open the local URL printed in the terminal (typically http://127.0.0.1:...).

What the app does (typical workflow)

Depending on the UI options you enable:

1. Input: paste a protein sequence (amino acids) or select a sample

2. Embedding: ProtBERT embedding extraction (1024D) using utils/embedder.py

3. Prediction:

* base models (RF, MLP, KAN)
* optional ensembles (soft voting, meta-ensemble)

4. Output:

* predicted family + confidence
* top-k probabilities
* optional unknown threshold

5. Mutation Robustness Test:

* generate mutants (absolute k or % of length)
* measure accuracy-on-mutants and confidence drop
* optionally save plots/results

Reproducing the Full Pipeline
1) Build dataset from FASTA
python data/prepare_dataset.py

Outputs expected in data/processed/:

* sequences.npy
* labels.npy
* label_map.json

2) Generate embeddings (ProtBERT → 1024D)
python data/generate_embeddings.py

Outputs expected in embeddings/:

* X.npy
* Y.npy

3) Train models

Run training scripts under models/ (examples; use the ones you need):

python models/train_rf
python models/train_mlp_v2
python models/train_kan
python models/train_kan_smooth
python models/generate_meta_features
python models/ensemble_meta


Saved artifacts appear in models/ and models/meta/.

4) Evaluate and produce confusion matrices / scores
python models/evaluate_models
python models/eval_on_test


Outputs:

* results/model_scores.csv
* confusion matrices in results/ and/or logged/logs/

Analyses & Figures
t-SNE
python tsne_protbert.py


Outputs in logs_tsne/:

* tsne_all_families.png
* per-family highlight plots
* tsne_kinase_highlight.png

UMAP + Heatmap
python umap_and_heatmap.py


Outputs in analysis_umap/:

* UMAP plots
* family-distance heatmap(s)

Silhouette scores
python silhouette_scores.py


Outputs in analysis_silhouette/:

* silhouette_scores.png

Mutation Robustness
Single-sequence robustness experiments
python mutation.py

Meta-ensemble robustness stress test
python mutation_stress_test_meta.py


Outputs in logs_mutations/:

* mutation_compare_accuracy.png
* mutation_compare_confidence.png
* per-family curves and cached .npz results

Output Artifacts (Where to Look)

* Final metrics: results/model_scores.csv
* Confusion matrices: results/*.png and logged/logs/*confusion*.png
* Training curves: logged/figs/ and logged/figures/
* t-SNE plots: logs_tsne/
* UMAP & heatmaps: analysis_umap/
* Mutation robustness: logs_mutations/

Notes / Common Issues

* Slow embedding extraction: ProtBERT embedding generation is heavy. Use CUDA if possible.
* Missing files: If the web app fails due to missing artifacts, download from Drive and verify the expected paths.
* FASTA format: sequences should contain valid amino acid letters; remove whitespace and headers when pasting into the app.
* Unknown threshold: the UI threshold is useful if you want to return unknown when confidence is low.

Credits

* UniProt for sequence retrieval
* ProtTrans / ProtBERT for embeddings
* Standard ML references: Random Forest, stacking, t-SNE, etc.

License

All rights reserved.

Contact

Repository owner: https://github.com/seitmmm


