# HaMeR ViT-B Ablation Study

Exam Project — Computer Vision, Master's Degree in Computer Science, UniBO
## Task

Ablation study on the [HaMeR (CVPR 2024)](https://github.com/geopavlakos/hamer) architecture: replacing the ViT-H backbone (632M parameters) with ViT-B (123M parameters) to analyze the trade-off between model size and 3D hand reconstruction quality.

## Project Structure
hamer-vitb-ablation/
├── notebooks/
│   └── inference_demo.ipynb  # Inference demo on Google Colab
├── results/
│   └── confronto_finale.png  # Qualitative comparison: ViT-H vs. ViT-B
├── scripts/
│   ├── __init__.py           # Updated backbone factory (overwrites hamer/models/backbones/__init__.py)
│   ├── aggiorna_npz.py       # Preprocessing: adds 2D keypoints to FreiHAND dataset
│   ├── hamer.py              # Renderer made optional (overwrites hamer/models/hamer.py)
│   ├── job.sbatch            # SLURM job v1 (baseline)
│   ├── job_v2.sbatch         # SLURM job v2 (optimized for l40 partition on DISI cluster)
│   ├── train_vitb.py         # Training script v1 (baseline)
│   ├── train_vitb_v2.py      # Training script v2 (with 2D reprojection loss)
│   └── vit.py                # ViT-B backbone implementation (overwrites hamer/models/backbones/vit.py)
├── README.md
└── requirements.txt

## Modifiche al codice originale HaMeR

### 1. Nuovo backbone ViT-B (`vit.py`)
Aggiunta funzione `vit_base()` con parametri ridotti rispetto a ViT-H:
- `embed_dim`: 1280 → 768
- `depth`: 32 → 12
- `num_heads`: 16 → 12

### 2. Factory backbone (`__init__.py`)
Registrazione di `vit_base` in `create_backbone()`.

### 3. Renderer opzionale (`hamer.py`)
Reso opzionale l'import di `pyrender` per compatibilità con ambienti headless (cluster HPC).

### 4. Config ViT-B (`model_config_vitb.yaml`)
- `BACKBONE.TYPE`: `vit` → `vit_base`
- `context_dim`: 1280 → 768

## Setup e riproduzione

### Requisiti
- Python 3.11
- PyTorch 2.5.1 + CUDA 11.8
- Dataset: [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
- Pesi MAE ViT-B: [OpenMMLab](https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base.pth)

### Installazione
```bash
git clone https://github.com/geopavlakos/hamer.git
cd hamer
pip install -e .[all]
pip install -e third-party/ViTPose

# Copia i file modificati
cp scripts/vit.py hamer/models/backbones/vit.py
cp scripts/__init__.py hamer/models/backbones/__init__.py
cp scripts/hamer.py hamer/models/hamer.py
```

### Preprocessing
```bash
python3 scripts/aggiorna_npz.py
```

### Training (cluster DISI)
```bash
sbatch scripts/job_v2.sbatch
```

## Risultati

| Modello | Backbone | Params | Train Loss | Val Loss |
|---------|----------|--------|------------|----------|
| HaMeR (originale) | ViT-H | 632M | — | — |
| HaMeR-B v1 (nostro) | ViT-B | 123M | 0.4126 | 0.3396 |
| HaMeR-B v2 (nostro) | ViT-B | 123M | TBD | TBD |

Il checkpoint del modello addestrato è disponibile su Google Drive: [link TBD]

## Riferimento

Pavlakos et al., *Reconstructing Hands in 3D with Transformers*, CVPR 2024.
