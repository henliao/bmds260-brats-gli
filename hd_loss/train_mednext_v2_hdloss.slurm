#!/bin/bash
#SBATCH --job-name=hdloss_f%a
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/users/hankliao/bmds260/logs/hdloss_%j.out

# MedNeXt-B kernel5 with staged HD boundary loss (Experiment B)
#
# Schedule: epochs 0-799 = Dice+CE (identical to baseline), epochs 800-999 = Dice+CE+Boundary
# Separate results dir: ~/bmds260/nnunet/results_hdloss/
#
# Usage:
#   sbatch --array=0-4 train_mednext_v2_hdloss.slurm
#   sbatch train_mednext_v2_hdloss.slurm <fold>

FOLD=${SLURM_ARRAY_TASK_ID:-$1}
if [ -z "$FOLD" ]; then
    echo "Usage: sbatch --array=0-4 train_mednext_v2_hdloss.slurm"
    echo "   or: sbatch train_mednext_v2_hdloss.slurm <fold>"
    exit 1
fi

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "Start: $(date)"
echo "Task: 501 (v2, HD Loss Experiment B, fold $FOLD)"

module load python/3.13.11
module load cuda/12.9.0
module load cudnn/9.8.0.87-12

export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results_hdloss
export PATH=~/.local/bin:$PATH

mkdir -p ~/bmds260/logs
mkdir -p "$RESULTS_FOLDER"

echo ""
echo "=== Step 1: Skipping (already preprocessed) ==="

echo ""
echo "=== Step 2: Training MedNeXt-B k5 HD Loss, Fold $FOLD ==="

# Check for existing checkpoint
RESULT_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTSGLI_v2/nnUNetTrainerV2_MedNeXt_B_kernel5_HDLoss__nnUNetPlansv2.1/fold_$FOLD"
if [ -f "$RESULT_DIR/model_latest.model" ] && [ -f "$RESULT_DIR/model_latest.model.pkl" ]; then
    PKL_SIZE=$(stat -c%s "$RESULT_DIR/model_latest.model.pkl" 2>/dev/null || echo 0)
    if [ "$PKL_SIZE" -gt 1000 ]; then
        echo "Resuming from checkpoint (fold $FOLD, pkl size: ${PKL_SIZE} bytes)"
        RESUME_FLAG="-c"
    else
        echo "ERROR: Checkpoint appears corrupted (pkl size: ${PKL_SIZE}). Refusing to start."
        exit 1
    fi
elif [ -d "$RESULT_DIR" ] && [ "$(ls -A "$RESULT_DIR" 2>/dev/null)" ]; then
    echo "WARNING: Result dir exists but no valid checkpoint found!"
    echo "Contents: $(ls "$RESULT_DIR")"
    echo "SAFETY: Refusing to overwrite. To start fresh, remove: $RESULT_DIR"
    exit 1
else
    echo "Starting fresh training (fold $FOLD)"
    RESUME_FLAG=""
fi

# Stream training log to SLURM stdout
TRAIN_LOG_DIR="$RESULT_DIR"
mkdir -p "$TRAIN_LOG_DIR"
(
    for i in $(seq 1 300); do
        LOG_FILE=$(ls -t "$TRAIN_LOG_DIR"/training_log_*.txt 2>/dev/null | head -1)
        if [ -n "$LOG_FILE" ]; then
            echo "[monitor] Streaming: $LOG_FILE"
            tail -f "$LOG_FILE" &
            TAIL_PID=$!
            echo $TAIL_PID > /tmp/hdloss_tail_${SLURM_JOB_ID}.pid
            wait $TAIL_PID
            break
        fi
        sleep 1
    done
) &
MONITOR_PID=$!

mednextv1_train 3d_fullres \
    nnUNetTrainerV2_MedNeXt_B_kernel5_HDLoss \
    Task501_BraTSGLI_v2 \
    $FOLD \
    $RESUME_FLAG
TRAIN_EXIT=$?

kill $MONITOR_PID 2>/dev/null
TAIL_PID=$(cat /tmp/hdloss_tail_${SLURM_JOB_ID}.pid 2>/dev/null)
kill $TAIL_PID 2>/dev/null
rm -f /tmp/hdloss_tail_${SLURM_JOB_ID}.pid

echo ""
echo "=== Training Complete (exit code: $TRAIN_EXIT) ==="
echo "End: $(date)"
