# BraTS-GLI 2024 Evaluation Tools

Lesion-wise Dice evaluation for the [BraTS 2024 Post-Treatment Glioma Segmentation Challenge](https://www.synapse.org/Synapse:syn53708249), reimplemented from the [official metrics](https://github.com/rachitsaluja/BraTS-2024-Metrics).

## What this computes

The BraTS 2024 challenge uses **lesion-wise Dice**, not voxel-wise Dice. Key differences:

1. **Dilation before connected component analysis** -- nearby blobs within a few voxels are merged into a single lesion (5 iterations for NETC/SNFH/RC, 3 for ET)
2. **Volume thresholding** -- GT lesions <= 20 voxels are excluded from false negative counting; predicted lesions <= 20 voxels are removed entirely (10 for ET)
3. **False positive penalty** -- unmatched predicted lesions appear in the denominator: `sum(dice_per_gt_lesion) / (n_gt_lesions + n_fp_lesions)`
4. **Per-label evaluation** -- scores are computed separately for NETC (label 1), SNFH (label 2), ET (label 3), RC (label 4)

## Setup

```bash
pip install connected-components-3d nibabel scipy numpy
```

On FarmShare:
```bash
python3 -m pip install --user --break-system-packages connected-components-3d
```

## Usage

### Generic (any framework)

```bash
python3 eval_lesion_dice.py \
  --pred-dir /path/to/predictions \
  --gt-dir /path/to/ground_truth \
  -o results.json
```

**pred-dir**: directory containing predicted segmentation NIfTI files (`.nii.gz`). Each file is a 3D volume with integer labels 0-4.

**gt-dir**: directory containing ground truth segmentation NIfTI files (`.nii.gz`). Same format. Filenames must match prediction filenames exactly.

### nnU-Net shortcut

If you trained with nnU-Net, the script can find predictions and ground truth automatically:

```bash
# MedNeXt on Task501, fold 0
python3 eval_lesion_dice.py --task 501 --fold 0

# Different trainer
python3 eval_lesion_dice.py --task 501 --fold 0 --trainer nnUNetTrainerV2__nnUNetPlansv2.1

# Specify output location
python3 eval_lesion_dice.py --task 501 --fold 0 -o my_results.json
```

This looks for predictions at:
```
~/bmds260/nnunet/results/nnUNet/3d_fullres/Task{ID}*/
  {trainer}__{plans}/fold_{N}/validation_raw/
```

And ground truth at:
```
~/bmds260/nnunet/preprocessed/Task{ID}*/gt_segmentations/
```

### Swin UNETR (MONAI)

If you trained Swin UNETR with MONAI, point to your output directories:

```bash
python3 eval_lesion_dice.py \
  --pred-dir ~/bmds260/swin_unetr/predictions/fold_0 \
  --gt-dir ~/bmds260/data/BraTS-GLI/training_data1_v2_segs \
  -o swin_fold0_lesion_dice.json
```

Your prediction directory should contain one `.nii.gz` per case. The ground truth directory should contain the corresponding segmentation files with matching filenames.

**Preparing GT for Swin UNETR**: If your GT files are stored per-case in BraTS format (`BraTS-GLI-XXXXX-YYY/BraTS-GLI-XXXXX-YYY-seg.nii.gz`), create a flat directory with renamed files:

```bash
mkdir -p gt_flat
for d in /path/to/brats_data/BraTS-GLI-*; do
  name=$(basename "$d")
  cp "$d/${name}-seg.nii.gz" gt_flat/${name}.nii.gz
done
```

Then use `--gt-dir gt_flat`. The prediction filenames must match (e.g., `BraTS-GLI-00005-100.nii.gz`).

### Running on FarmShare (SLURM)

For large evaluation sets, submit as a SLURM job:

```bash
cat > eval_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=eval-dice
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=eval_dice_%j.out

python3 eval_lesion_dice.py \
  --pred-dir /path/to/predictions \
  --gt-dir /path/to/ground_truth \
  -o results.json
EOF

sbatch eval_job.sh
```

Processing time: ~1-2 minutes per case (270 cases takes ~45 min on a compute node).

## Output format

```json
{
  "pred_dir": "/path/to/predictions",
  "gt_dir": "/path/to/ground_truth",
  "summary": {
    "NETC": {"mean": 0.4680, "std": 0.3587, "median": 0.5774, "n_cases": 127},
    "SNFH": {"mean": 0.8736, "std": 0.1224, "median": 0.9006, "n_cases": 270},
    "ET":   {"mean": 0.8029, "std": 0.1855, "median": 0.8966, "n_cases": 200},
    "RC":   {"mean": 0.7233, "std": 0.2969, "median": 0.8523, "n_cases": 248}
  },
  "per_case": [
    {
      "case": "BRATS_0010",
      "NETC": {"dice": 0.5774, "n_gt": 1, "n_pred": 1, "tp": 1, "fp": 0, "fn": 0},
      "SNFH": {"dice": 0.9092, "n_gt": 1, "n_pred": 1, "tp": 1, "fp": 0, "fn": 0},
      "ET": {"dice": 0.9133, "n_gt": 1, "n_pred": 1, "tp": 1, "fp": 0, "fn": 0},
      "RC": {"dice": 0.8523, "n_gt": 1, "n_pred": 1, "tp": 1, "fp": 0, "fn": 0}
    }
  ]
}
```

Labels where both GT and prediction are empty are marked `"absent"` and excluded from averages.

## Label mapping

| Label ID | Name | Description |
|----------|------|-------------|
| 0 | Background | Non-tumor |
| 1 | NETC | Non-enhancing tumor core |
| 2 | SNFH | Surrounding non-enhancing FLAIR hyperintensity |
| 3 | ET | Enhancing tumor |
| 4 | RC | Resection cavity |

## Comparison with challenge results

The BraTS 2024 challenge leaderboard reports lesion-wise Dice. Published benchmarks:

| Team | NETC | SNFH | ET | RC | Method |
|------|------|------|----|----|--------|
| 2024 Winner | 0.808 | 0.893 | 0.790 | 0.776 | 6-model ensemble + GAN augmentation |
| 2025 Winner | 0.749 | 0.825 | 0.790 | 0.872 | On-the-fly GAN augmentation |

Note: these are test set scores. Validation fold scores are typically similar but not directly comparable.
