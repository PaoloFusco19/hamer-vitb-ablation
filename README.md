# HaMeR ViT-B Ablation Study

Progetto d'esame — Computer Vision, Magistrale Informatica, UniBO

## Obiettivo

Ablation study sull'architettura di [HaMeR (CVPR 2024)](https://github.com/geopavlakos/hamer): sostituzione del backbone ViT-H (632M parametri) con ViT-B (123M parametri) per analizzare il trade-off tra dimensione del modello e qualità della ricostruzione 3D della mano.

## Struttura
scripts/
├── vit.py              # backbone ViT-B aggiunto (modifica di hamer/models/backbones/vit.py)
├── init.py         # factory aggiornato (modifica di hamer/models/backbones/init.py)
├── hamer.py            # renderer reso opzionale (modifica di hamer/models/hamer.py)
├── aggiorna_npz.py     # preprocessing: aggiunge keypoint 2D al dataset FreiHAND
├── train_vitb_v2.py    # training script v2 (con reprojection loss 2D)
├── train_vitb.py       # training script v1
├── job_v2.sbatch       # SLURM job per cluster DISI (partizione l40)
└── job.sbatch          # SLURM job v1
notebooks/
└── inference_demo.ipynb  # demo inferenza su Colab
results/
└── confronto_finale.png  # confronto qualitativo ViT-H vs ViT-B

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
