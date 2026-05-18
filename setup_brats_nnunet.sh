#!/bin/bash
# Convert BraTS-GLI 2024 data to nnU-Net v1 format using symlinks.
#
# BraTS format:
#   BraTS-GLI-XXXXX-YYY/{*-t1n,*-t1c,*-t2w,*-t2f,*-seg}.nii.gz
#
# nnU-Net format:
#   TaskXXX_BraTSGLI/imagesTr/BRATS_XXXX_{0000,0001,0002,0003}.nii.gz
#                    labelsTr/BRATS_XXXX.nii.gz
#
# Usage:
#   bash setup_brats_nnunet.sh                          # default: Task500, training_data1_v2
#   bash setup_brats_nnunet.sh 501 /path/to/brats_data  # custom task ID and data dir
#
# This creates symlinks instead of copies to save ~35 GB of disk.

set -e

TASK_ID=${1:-500}
BRATS_DIR=${2:-~/bmds260/data/BraTS-GLI/training_data1_v2}

export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results

TASK_NAME="Task${TASK_ID}_BraTSGLI"
[ "$TASK_ID" = "501" ] && TASK_NAME="Task501_BraTSGLI_v2"

RAW_DIR="$nnUNet_raw_data_base/nnUNet_raw_data/$TASK_NAME"

mkdir -p "$RAW_DIR/imagesTr"
mkdir -p "$RAW_DIR/labelsTr"
mkdir -p "$RAW_DIR/imagesTs"

echo "Converting BraTS-GLI to nnU-Net format..."
echo "Source: $BRATS_DIR"
echo "Target: $RAW_DIR"

# Count available patients
N=$(ls -d "$BRATS_DIR"/BraTS-GLI-* 2>/dev/null | wc -l)
echo "Found $N patient folders"

if [ "$N" -eq 0 ]; then
    echo "No data found. Check BRATS_DIR path."
    exit 1
fi

# Convert: create symlinks (saves disk space)
# Modality mapping: 0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f
i=0
for patient_dir in "$BRATS_DIR"/BraTS-GLI-*; do
    patient=$(basename "$patient_dir")
    case_id=$(printf "BRATS_%04d" $i)

    # Symlink images (4 modalities)
    ln -sf "$patient_dir/${patient}-t1n.nii.gz" "$RAW_DIR/imagesTr/${case_id}_0000.nii.gz"
    ln -sf "$patient_dir/${patient}-t1c.nii.gz" "$RAW_DIR/imagesTr/${case_id}_0001.nii.gz"
    ln -sf "$patient_dir/${patient}-t2w.nii.gz" "$RAW_DIR/imagesTr/${case_id}_0002.nii.gz"
    ln -sf "$patient_dir/${patient}-t2f.nii.gz" "$RAW_DIR/imagesTr/${case_id}_0003.nii.gz"

    # Symlink label
    ln -sf "$patient_dir/${patient}-seg.nii.gz" "$RAW_DIR/labelsTr/${case_id}.nii.gz"

    i=$((i + 1))
    if [ $((i % 100)) -eq 0 ]; then
        echo "  Processed $i / $N"
    fi
done

echo "Converted $i patients"
echo ""

# Save ID mapping (BRATS_XXXX -> BraTS-GLI-XXXXX-YYY)
python3 << 'PYEOF'
import json, os, glob

brats_dir = os.path.expanduser(os.environ.get("BRATS_DIR_PY", "~/bmds260/data/BraTS-GLI/training_data1_v2"))
patients = sorted([os.path.basename(d) for d in glob.glob(os.path.join(brats_dir, "BraTS-GLI-*"))])
mapping = {f"BRATS_{i:04d}": p for i, p in enumerate(patients)}
out = os.path.expanduser(f"~/bmds260/id_mapping_task{os.environ.get('TASK_ID_PY', '500')}.json")
with open(out, "w") as f:
    json.dump(mapping, f, indent=2)
print(f"Saved ID mapping ({len(mapping)} cases) to {out}")
PYEOF

# Create dataset.json
BRATS_DIR_PY="$BRATS_DIR" TASK_ID_PY="$TASK_ID" python3 << 'PYEOF'
import json, os, glob

raw_dir = os.environ.get("RAW_DIR_PY", "")
if not raw_dir:
    task_id = os.environ.get("TASK_ID_PY", "500")
    task_name = f"Task{task_id}_BraTSGLI"
    if task_id == "501":
        task_name = "Task501_BraTSGLI_v2"
    raw_dir = os.path.expanduser(f"~/bmds260/nnunet/raw_data_base/nnUNet_raw_data/{task_name}")

cases = sorted(glob.glob(os.path.join(raw_dir, "imagesTr", "*_0000.nii.gz")))
case_ids = [os.path.basename(c).replace("_0000.nii.gz", "") for c in cases]

dataset = {
    "name": "BraTS-GLI 2024",
    "description": "Post-treatment glioma segmentation",
    "reference": "BraTS 2024 Challenge",
    "licence": "CC BY-SA 4.0",
    "release": "1.0",
    "tensorImageSize": "4D",
    "modality": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
    "labels": {"0": "Background", "1": "NETC", "2": "SNFH", "3": "ET", "4": "RC"},
    "numTraining": len(case_ids),
    "numTest": 0,
    "training": [
        {"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"}
        for cid in case_ids
    ],
    "test": []
}

out_path = os.path.join(raw_dir, "dataset.json")
with open(out_path, "w") as f:
    json.dump(dataset, f, indent=2)
print(f"Wrote dataset.json with {len(case_ids)} training cases")
PYEOF
