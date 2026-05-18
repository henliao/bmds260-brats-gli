#!/bin/bash
#SBATCH --job-name=mednext_v2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/home/users/hankliao/bmds260/logs/mednext_v2_%j.out
#SBATCH --error=/home/users/hankliao/bmds260/logs/mednext_v2_%j.err

# MedNeXt-B (kernel 5x5x5) training on BraTS-GLI 2024
# Task 501 = v2 dataset (1,459 cases from Anthony's train_val_set)
#
# Usage:
#   sbatch train_mednext_v2.slurm        # trains fold 0
#   sbatch train_mednext_v2.slurm 2      # trains fold 2
#
# Submit all 5 folds:
#   for f in 0 1 2 3 4; do sbatch train_mednext_v2.slurm $f; done
#
# Prerequisites:
#   - MedNeXt installed: pip install --user nnunet-mednext
#   - Data converted to nnU-Net format (see setup_brats_nnunet.sh)

# Environment
export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results
export PATH=~/.local/bin:$PATH

mkdir -p ~/bmds260/logs

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Start: $(date)"
echo "Task: 501 (v2, 1459 cases from Anthony's train_val_set)"
echo ""

FOLD=${1:-0}

# Step 1: Plan and preprocess (skip if already done)
if [ ! -d "$nnUNet_preprocessed/Task501_BraTSGLI_v2" ]; then
    echo "=== Step 1: Planning and Preprocessing ==="
    mednextv1_plan_and_preprocess -t 501 --verify_dataset_integrity
    echo "Preprocessing done: $(date)"
else
    echo "=== Step 1: Skipping (already preprocessed) ==="
fi

# Step 2: Train MedNeXt-B with 5x5x5 kernels
echo ""
echo "=== Step 2: Training MedNeXt-B k5, Fold $FOLD ==="
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_B_kernel5 Task501_BraTSGLI_v2 $FOLD

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
