#!/bin/bash
# Prepare BraTS-GLI 2024 holdout test set for nnU-Net inference and evaluation.
#
# Creates:
#   1. imagesTs/ symlinks in the nnU-Net raw data directory (for nnUNetv2_predict)
#   2. Flat GT directory with renamed segmentations (for eval_lesion_dice.py)
#   3. ID mapping JSON (HOLDOUT_XXXX -> BraTS-GLI-XXXXX-YYY)
#
# Usage:
#   bash prep_holdout.sh                                    # defaults
#   bash prep_holdout.sh 501 /path/to/holdout_test_set      # custom task + data
#
# After training completes, run inference:
#   export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
#   export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
#   export RESULTS_FOLDER=~/bmds260/nnunet/results
#   nnUNetv2_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTSGLI_v2/imagesTs \
#     -o ~/bmds260/holdout_predictions -tr nnUNetTrainerV2_MedNeXt_B_kernel5 \
#     -t 501 -m 3d_fullres -f 0 1 2 3 4
#
# Then evaluate:
#   python3 eval_lesion_dice.py --pred-dir ~/bmds260/holdout_predictions \
#     --gt-dir ~/bmds260/holdout_gt -o holdout_lesion_dice.json

set -e

TASK_ID=${1:-501}
HOLDOUT_DIR=${2:-~/bmds260/holdout_test_set}
HOLDOUT_DIR=$(eval echo "$HOLDOUT_DIR")

export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base

TASK_NAME="Task${TASK_ID}_BraTSGLI"
[ "$TASK_ID" = "501" ] && TASK_NAME="Task501_BraTSGLI_v2"

RAW_DIR="$nnUNet_raw_data_base/nnUNet_raw_data/$TASK_NAME"
GT_DIR=~/bmds260/holdout_gt

mkdir -p "$RAW_DIR/imagesTs"
mkdir -p "$GT_DIR"

echo "Preparing holdout test set..."
echo "Source: $HOLDOUT_DIR"
echo "imagesTs: $RAW_DIR/imagesTs"
echo "GT dir: $GT_DIR"

N=$(ls -d "$HOLDOUT_DIR"/BraTS-GLI-* 2>/dev/null | wc -l)
echo "Found $N holdout cases"

if [ "$N" -eq 0 ]; then
    echo "No holdout data found. Check HOLDOUT_DIR path."
    exit 1
fi

# Create imagesTs symlinks and flat GT directory
i=0
declare -A id_map
for patient_dir in "$HOLDOUT_DIR"/BraTS-GLI-*; do
    patient=$(basename "$patient_dir")
    case_id=$(printf "HOLDOUT_%04d" $i)

    # Symlink test images (4 modalities)
    ln -sf "$patient_dir/${patient}-t1n.nii.gz" "$RAW_DIR/imagesTs/${case_id}_0000.nii.gz"
    ln -sf "$patient_dir/${patient}-t1c.nii.gz" "$RAW_DIR/imagesTs/${case_id}_0001.nii.gz"
    ln -sf "$patient_dir/${patient}-t2w.nii.gz" "$RAW_DIR/imagesTs/${case_id}_0002.nii.gz"
    ln -sf "$patient_dir/${patient}-t2f.nii.gz" "$RAW_DIR/imagesTs/${case_id}_0003.nii.gz"

    # Symlink GT segmentation (flat, for eval_lesion_dice.py)
    ln -sf "$patient_dir/${patient}-seg.nii.gz" "$GT_DIR/${case_id}.nii.gz"

    i=$((i + 1))
done

echo "Prepared $i holdout cases"

# Save ID mapping
python3 << 'PYEOF'
import json, os, glob

holdout_dir = os.path.expanduser(os.environ.get("HOLDOUT_DIR", "~/bmds260/holdout_test_set"))
patients = sorted([os.path.basename(d) for d in glob.glob(os.path.join(holdout_dir, "BraTS-GLI-*"))])
mapping = {f"HOLDOUT_{i:04d}": p for i, p in enumerate(patients)}
out = os.path.expanduser("~/bmds260/holdout_id_mapping.json")
with open(out, "w") as f:
    json.dump(mapping, f, indent=2)
print(f"Saved holdout ID mapping ({len(mapping)} cases) to {out}")
PYEOF

echo ""
echo "Done. Next steps:"
echo "  1. Run inference: nnUNetv2_predict -i $RAW_DIR/imagesTs -o ~/bmds260/holdout_predictions ..."
echo "  2. Evaluate: python3 eval_lesion_dice.py --pred-dir ~/bmds260/holdout_predictions --gt-dir $GT_DIR"
