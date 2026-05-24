#!/bin/bash
#SBATCH --job-name=curgan_rr
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/home/users/hankliao/bmds260/logs/curgan_rr_%j.out

# Round-robin CurriculumGAN trainer.
# Trains all 5 folds in rotation, pausing at curriculum phase boundaries
# (epoch 300, 700, 1000) to give each fold fair GPU time.
#
# Each fold is independently resumable: if more GPU nodes come online,
# submit individual folds with the original train_mednext_v2_curriculum_gan.slurm
# and they pick up from wherever this script left them.
#
# Usage:
#   sbatch train_curgan_roundrobin.slurm           # all folds, all phases
#   sbatch train_curgan_roundrobin.slurm 0,2,4     # specific folds only

FOLDS_ARG=${1:-"0,1,2,3,4"}
IFS=',' read -ra FOLDS <<< "$FOLDS_ARG"

PHASE_EPOCHS=(300 700 1000)

echo "=== CurriculumGAN Round-Robin Trainer ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "Start: $(date)"
echo "Folds: ${FOLDS[*]}"
echo "Phase milestones: ${PHASE_EPOCHS[*]}"
echo ""

module load python/3.13.11
module load cuda/12.9.0
module load cudnn/9.8.0.87-12

export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results
export GLIGAN_WEIGHTS_DIR=~/bmds260/gligan_weights/brats2024

TRAINER=nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN
TASK=Task501_BraTSGLI_v2
BASE_DIR="$RESULTS_FOLDER/nnUNet/3d_fullres/$TASK/${TRAINER}__nnUNetPlansv2.1"

mkdir -p ~/bmds260/logs

get_current_epoch() {
    local fold=$1
    local log=$(ls -t "$BASE_DIR/fold_$fold"/training_log_*.txt 2>/dev/null | head -1)
    if [ -n "$log" ]; then
        local ep=$(grep "^epoch:" "$log" | tail -1 | awk '{print $2}')
        echo "${ep:-0}"
    else
        echo "0"
    fi
}

has_valid_checkpoint() {
    local fold=$1
    local dir="$BASE_DIR/fold_$fold"
    if [ -f "$dir/model_latest.model" ] && [ -f "$dir/model_latest.model.pkl" ]; then
        local size=$(stat -c%s "$dir/model_latest.model.pkl" 2>/dev/null || echo 0)
        [ "$size" -gt 1000 ]
    else
        return 1
    fi
}

train_fold_to_epoch() {
    local fold=$1
    local target=$2

    local current=$(get_current_epoch "$fold")
    echo "[$(date +%H:%M:%S)] Fold $fold: currently at epoch $current, target $target"

    if [ "$current" -ge "$target" ]; then
        echo "[$(date +%H:%M:%S)] Fold $fold: already past epoch $target, skipping"
        return 0
    fi

    # Determine resume flag
    local resume=""
    if has_valid_checkpoint "$fold"; then
        echo "[$(date +%H:%M:%S)] Fold $fold: resuming from checkpoint"
        resume="-c"
    elif [ -d "$BASE_DIR/fold_$fold" ] && [ "$(ls -A "$BASE_DIR/fold_$fold" 2>/dev/null)" ]; then
        echo "[$(date +%H:%M:%S)] Fold $fold: ERROR - result dir exists but no valid checkpoint"
        return 1
    else
        echo "[$(date +%H:%M:%S)] Fold $fold: starting fresh"
    fi

    # nnU-Net trains to max_num_epochs (1000). We can't tell it to stop at 300.
    # Instead, we set max_num_epochs via env var and let the trainer respect it.
    # But nnU-Net v1 doesn't support that. So we use a different approach:
    # Run training in background, poll the log, and kill when target is reached.
    mednextv1_train 3d_fullres "$TRAINER" "$TASK" "$fold" $resume &
    local pid=$!

    echo "[$(date +%H:%M:%S)] Fold $fold: training PID $pid, polling until epoch $target"

    # Poll every 60s. Kill when target reached.
    while kill -0 $pid 2>/dev/null; do
        sleep 60
        local ep=$(get_current_epoch "$fold")
        if [ "$ep" -ge "$target" ] && [ "$target" -lt 1000 ]; then
            echo "[$(date +%H:%M:%S)] Fold $fold: reached epoch $ep (target $target), pausing"
            kill -TERM $pid 2>/dev/null
            wait $pid 2>/dev/null
            # Give nnU-Net a moment to flush checkpoint
            sleep 10
            return 0
        fi
    done

    # Process exited on its own (finished 1000 or error)
    wait $pid
    local exit_code=$?
    local final_ep=$(get_current_epoch "$fold")
    echo "[$(date +%H:%M:%S)] Fold $fold: training exited (code $exit_code) at epoch $final_ep"
    return $exit_code
}

echo "=== Phase rotation begins ==="
echo ""

for target in "${PHASE_EPOCHS[@]}"; do
    echo "=========================================="
    echo "  PHASE TARGET: epoch $target"
    echo "=========================================="
    for fold in "${FOLDS[@]}"; do
        current=$(get_current_epoch "$fold")
        if [ "$current" -ge 1000 ]; then
            echo "[$(date +%H:%M:%S)] Fold $fold: already complete (epoch $current), skipping"
            continue
        fi
        if [ "$current" -ge "$target" ]; then
            echo "[$(date +%H:%M:%S)] Fold $fold: already at $current >= $target, skipping"
            continue
        fi

        echo ""
        echo "--- Training fold $fold to epoch $target ---"
        train_fold_to_epoch "$fold" "$target"
        status=$?

        if [ $status -ne 0 ]; then
            echo "[$(date +%H:%M:%S)] WARNING: Fold $fold failed (exit $status), continuing with next fold"
        fi

        echo ""
    done
done

echo ""
echo "=== Round-robin complete ==="
echo "End: $(date)"
echo ""
echo "=== Final status ==="
for fold in "${FOLDS[@]}"; do
    ep=$(get_current_epoch "$fold")
    echo "Fold $fold: epoch $ep"
done
