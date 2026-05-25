#!/bin/bash
#SBATCH --job-name=holdout_inf
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/hankliao/bmds260/logs/holdout_inf_%j.out

# Holdout inference: 5-fold ensemble prediction on 162 holdout cases.
# Usage: sbatch run_holdout_inference.slurm [trainer]
# Default trainer: nnUNetTrainerV2_MedNeXt_B_kernel5

TRAINER=${1:-nnUNetTrainerV2_MedNeXt_B_kernel5}

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "Start: $(date)"
echo "Trainer: $TRAINER"

module load python/3.13.11
module load cuda/12.9.0
module load cudnn/9.8.0.87-12

export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results
export PATH=~/.local/bin:$PATH

mkdir -p ~/bmds260/logs

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTSGLI_v2/imagesTs

# Set output dir based on trainer
if [ "$TRAINER" = "nnUNetTrainerV2_MedNeXt_B_kernel5" ]; then
    OUTPUT_DIR=~/bmds260/holdout_predictions_baseline
elif [ "$TRAINER" = "nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN" ]; then
    OUTPUT_DIR=~/bmds260/holdout_predictions_curgan
else
    OUTPUT_DIR=~/bmds260/holdout_predictions_${TRAINER}
fi

echo ""
echo "=== Running 5-fold ensemble inference ==="
echo "Input: $INPUT_DIR ($(ls "$INPUT_DIR"/*.nii.gz 2>/dev/null | wc -l) files)"
echo "Output: $OUTPUT_DIR"

mednextv1_predict \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -t 501 \
    -m 3d_fullres \
    -tr "$TRAINER" \
    -f 0 1 2 3 4

PRED_EXIT=$?

echo ""
echo "=== Inference Complete (exit code: $PRED_EXIT) ==="
echo "Predictions: $(ls "$OUTPUT_DIR"/*.nii.gz 2>/dev/null | wc -l) files"
echo "End: $(date)"

# Run evaluation if predictions succeeded
if [ $PRED_EXIT -eq 0 ]; then
    echo ""
    echo "=== Running lesion-wise Dice evaluation ==="
    python3 ~/bmds260/eval_lesion_dice.py \
        --pred-dir "$OUTPUT_DIR" \
        --gt-dir ~/bmds260/holdout_gt \
        -o ~/bmds260/holdout_${TRAINER}_lesion_dice.json

    EVAL_EXIT=$?
    echo "Evaluation exit code: $EVAL_EXIT"
    if [ $EVAL_EXIT -eq 0 ]; then
        echo ""
        echo "=== Results ==="
        python3 -c "
import json
with open('/home/users/hankliao/bmds260/holdout_${TRAINER}_lesion_dice.json') as f:
    data = json.load(f)
for label, stats in data['summary'].items():
    print(f'{label}: {stats[\"mean\"]:.4f} +/- {stats[\"std\"]:.4f} (median {stats[\"median\"]:.4f}, n={stats[\"n_cases\"]})')
"
    fi
fi
