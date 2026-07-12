# HaMeR ViT-B Ablation Study

<<<<<<< HEAD
Computer Vision Exam Project — Master's Degree in Computer Science, UniBO

## Objective

Ablation study on the [HaMeR (CVPR 2024)](https://github.com/geopavlakos/hamer) architecture: replacing the ViT-H backbone (632M parameters) with ViT-B (123M parameters) to analyze the trade-off between model size and 3D hand reconstruction quality.

## Project Structure
=======
Exam Project — Computer Vision, Master's Degree in Computer Science, UniBO
## Task

Ablation study on the [HaMeR (CVPR 2024)](https://github.com/geopavlakos/hamer) architecture: replacing the ViT-H backbone (632M parameters) with ViT-B (123M parameters) to analyze the trade-off between model size and 3D hand reconstruction quality.

## Project Structure

    hamer-vitb-ablation/
    ├── notebooks/
    │   └── inference_demo.ipynb   # Inference demo on Google Colab
    ├── results/
    │   └── confronto_finale.png   # Qualitative comparison: ViT-H vs. ViT-B
    ├── scripts/
    │   ├── vit.py                 # ViT-B backbone (overwrites hamer/models/backbones/vit.py)
    │   ├── __init__.py            # Updated backbone factory (overwrites hamer/models/backbones/__init__.py)
    │   ├── hamer.py               # Optional renderer (overwrites hamer/models/hamer.py)
    │   ├── aggiorna_npz.py        # Preprocessing: adds 2D keypoints to FreiHAND dataset
    │   ├── train_vitb.py          # Training script v1 (baseline)
    │   ├── train_vitb_v2.py       # Training script v2 (with 2D reprojection loss)
    │   ├── job.sbatch             # SLURM job v1
    │   └── job_v2.sbatch          # SLURM job v2 (l40 partition, DISI cluster)
    └── README.md

## Changes to the Original HaMeR Codebase

**vit.py** — Added vit_base() function with reduced parameters:
- embed_dim: 1280 → 768
- depth: 32 → 12
- num_heads: 16 → 12

**__init__.py** — Registered vit_base in create_backbone().

**hamer.py** — Made pyrender import optional for headless environments (HPC cluster).

**model_config_vitb.yaml** — BACKBONE.TYPE: vit → vit_base, context_dim: 1280 → 768

## Setup and Reproduction

1. Clone the original HaMeR repo and install dependencies:

        git clone https://github.com/geopavlakos/hamer.git
        cd hamer
        pip install -e .[all]
        pip install -e third-party/ViTPose

2. Replace the modified files:

        cp scripts/vit.py hamer/models/backbones/vit.py
        cp scripts/__init__.py hamer/models/backbones/__init__.py
        cp scripts/hamer.py hamer/models/hamer.py

3. Download the MAE ViT-B pretrained weights from OpenMMLab and the FreiHAND dataset.

4. Preprocessing (adds 2D keypoints to the NPZ):

        python3 scripts/aggiorna_npz.py

5. Launch training on SLURM cluster:

        sbatch scripts/job_v2.sbatch

## Results

| Model | Backbone | Params | Val Loss |
|-------|----------|--------|----------|
| HaMeR (original) | ViT-H | 632M | — |
| HaMeR-B v1 (ours) | ViT-B | 123M | 0.3396 |
| HaMeR-B v2 (ours) | ViT-B | 123M | TBD |

Trained checkpoint available on Google Drive: [link TBD]

## Reference

Pavlakos et al., *Reconstructing Hands in 3D with Transformers*, CVPR 2024.
