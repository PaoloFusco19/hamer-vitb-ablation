#!/bin/bash
#SBATCH --job-name=hamer_vitb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paolo.fusco2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/paolo.fusco2/hamer_training.log
#SBATCH --chdir=/scratch.hpc/paolo.fusco2
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Attiva il virtual environment
source /scratch.hpc/paolo.fusco2/venv/bin/activate

# Verifica CUDA
python3 -c "import torch; print('CUDA disponibile:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nessuna')"

# Lancia il training
python3 /scratch.hpc/paolo.fusco2/hamer/train_vitb.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
