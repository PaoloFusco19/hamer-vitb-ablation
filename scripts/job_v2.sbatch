#!/bin/bash
#SBATCH --job-name=hamer_vitb_v2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=paolo.fusco2@studio.unibo.it
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=l40
#SBATCH --output=/scratch.hpc/paolo.fusco2/hamer_training_v2.log
#SBATCH --chdir=/scratch.hpc/paolo.fusco2
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

source /scratch.hpc/paolo.fusco2/venv/bin/activate

python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Step 1: aggiorna NPZ con keypoint 2D (se non esiste già)
if [ ! -f /scratch.hpc/paolo.fusco2/hamer/hamer_training_data/freihand_train_v2.npz ]; then
    echo "Creazione freihand_train_v2.npz..."
    python3 /scratch.hpc/paolo.fusco2/hamer/aggiorna_npz.py
else
    echo "freihand_train_v2.npz già presente, skip."
fi

# Step 2: training v2
python3 /scratch.hpc/paolo.fusco2/hamer/train_vitb_v2.py

echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
