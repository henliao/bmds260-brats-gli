#!/bin/bash
#SBATCH --job-name=curgan_f%a
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/home/users/hankliao/bmds260/logs/curgan_%j.out

FOLD=${SLURM_ARRAY_TASK_ID:-$1}
if [ -z "$FOLD" ]; then
    echo "Usage: sbatch --array=0-4 train_mednext_v2_curriculum_gan.slurm"
    echo "   or: sbatch train_mednext_v2_curriculum_gan.slurm <fold>"
    exit 1
fi

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "Start: $(date)"
echo "Task: 501 (v2, CurriculumGAN, fold $FOLD)"

module load python/3.13.11
module load cuda/12.9.0
module load cudnn/9.8.0.87-12

export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results
export GLIGAN_WEIGHTS_DIR=~/bmds260/gligan_weights/brats2024
export PATH=~/.local/bin:$PATH

mkdir -p ~/bmds260/logs

echo ""
echo "=== Step 1: Skipping (already preprocessed) ==="

echo ""
echo "=== Step 2: Training MedNeXt-B k5 CurriculumGAN, Fold $FOLD ==="

# Check if there's a checkpoint to resume from
# SAFETY: Never overwrite existing checkpoints. Always use -c if checkpoint exists.
RESULT_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTSGLI_v2/nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN__nnUNetPlansv2.1/fold_$FOLD"
if [ -f "$RESULT_DIR/model_latest.model" ] && [ -f "$RESULT_DIR/model_latest.model.pkl" ]; then
    # Verify checkpoint pkl is non-trivial (> 1KB)
    PKL_SIZE=$(stat -c%s "$RESULT_DIR/model_latest.model.pkl" 2>/dev/null || echo 0)
    if [ "$PKL_SIZE" -gt 1000 ]; then
        echo "Resuming from checkpoint (fold $FOLD, pkl size: ${PKL_SIZE} bytes)"
        RESUME_FLAG="-c"
    else
        echo "ERROR: Checkpoint appears corrupted (pkl size: ${PKL_SIZE}). Refusing to start."
        echo "Manually inspect: $RESULT_DIR/"
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

# Stream nnU-Net's training log to SLURM stdout for real-time monitoring.
# The log file is created after training starts, so we wait for it.
TRAIN_LOG_DIR="$RESULT_DIR"
(
    # Wait up to 5 min for the training log to appear
    for i in $(seq 1 300); do
        LOG_FILE=$(ls -t "$TRAIN_LOG_DIR"/training_log_*.txt 2>/dev/null | head -1)
        if [ -n "$LOG_FILE" ]; then
            echo "[monitor] Streaming: $LOG_FILE"
            tail -f "$LOG_FILE" &
            TAIL_PID=$!
            echo $TAIL_PID > /tmp/curgan_tail_${SLURM_JOB_ID}.pid
            wait $TAIL_PID
            break
        fi
        sleep 1
    done
) &
MONITOR_PID=$!

mednextv1_train 3d_fullres \
    nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN \
    Task501_BraTSGLI_v2 \
    $FOLD \
    $RESUME_FLAG
TRAIN_EXIT=$?

# Stop the log streamer
kill $MONITOR_PID 2>/dev/null
TAIL_PID=$(cat /tmp/curgan_tail_${SLURM_JOB_ID}.pid 2>/dev/null)
kill $TAIL_PID 2>/dev/null
rm -f /tmp/curgan_tail_${SLURM_JOB_ID}.pid

echo ""
echo "=== Training Complete (exit code: $TRAIN_EXIT) ==="
echo "End: $(date)"

# Run validation predictions
echo ""
echo "=== Step 3: Validation Predictions ==="
mednextv1_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTSGLI_v2/imagesTs \
    -o $RESULT_DIR/validation_raw \
    -tr nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN \
    -t 501 -m 3d_fullres -f $FOLD 2>/dev/null || echo "Validation predict skipped (may need all folds)"
