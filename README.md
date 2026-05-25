# BraTS-GLI 2024 Tools

Shared tooling for the [BraTS 2024 Post-Treatment Glioma Segmentation Challenge](https://www.synapse.org/Synapse:syn53708249):

- **eval_lesion_dice.py** -- Official lesion-wise Dice evaluation, reimplemented from the [challenge metrics](https://github.com/rachitsaluja/BraTS-2024-Metrics)
- **setup_brats_nnunet.sh** -- Convert BraTS-GLI data to nnU-Net v1 format (symlinks)
- **train_mednext_v2.slurm** -- SLURM script for MedNeXt-B (kernel 5x5x5) training on FarmShare
- **prep_holdout.sh** -- Prepare holdout test set for inference and evaluation

## What this computes

The BraTS 2024 challenge uses **lesion-wise Dice**, not voxel-wise Dice. Key differences:

1. **Dilation before connected component analysis** -- nearby blobs within a few voxels are merged into a single lesion (5 iterations for NETC/SNFH/RC, 3 for ET)
2. **Volume thresholding** -- GT lesions <= 20 voxels are excluded from false negative counting; predicted lesions <= 20 voxels are removed entirely (10 for ET)
3. **False positive penalty** -- unmatched predicted lesions appear in the denominator: `sum(dice_per_gt_lesion) / (n_gt_lesions + n_fp_lesions)`
4. **Per-label evaluation** -- scores are computed separately for NETC (label 1), SNFH (label 2), ET (label 3), RC (label 4)

## Setup

### FarmShare

`nibabel`, `scipy`, and `numpy` are already installed system-wide. You only need one extra package:

```bash
python3 -m pip install --user --break-system-packages connected-components-3d
```

That's it. No venv needed.

### Other environments

```bash
pip install connected-components-3d nibabel scipy numpy
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

## Results

### Lesion-wise Dice (cross-validation, 5-fold, 292 cases per fold)

**Baseline: MedNeXt-B kernel 5x5x5** (1000 epochs, Task501, 1459 training cases)

| Fold | NETC | SNFH | ET | RC |
|------|------|------|----|----|
| 0 | 0.657 | 0.854 | 0.752 | 0.753 |
| 1 | 0.616 | 0.879 | 0.752 | 0.777 |
| 2 | 0.724 | 0.859 | 0.751 | 0.787 |
| 3 | 0.653 | 0.873 | 0.754 | 0.777 |
| 4 | 0.630 | 0.869 | 0.765 | 0.761 |
| **Mean** | **0.656** | **0.867** | **0.755** | **0.771** |
| Std | 0.040 | 0.010 | 0.006 | 0.013 |

**CurriculumGAN: MedNeXt-B kernel 5x5x5** (curriculum-scheduled GliGAN augmentation)

| Fold | NETC | SNFH | ET | RC |
|------|------|------|----|----|
| 0 | TBD | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD |
| **Mean** | **TBD** | **TBD** | **TBD** | **TBD** |

### nnU-Net voxel-wise Dice (cross-validation, 1459 cases)

From `mednextv1_determine_postprocessing` (consolidates all 5 folds):

| Metric | NETC | SNFH | ET | RC |
|--------|------|------|----|----|
| Raw | 0.596 | 0.898 | 0.761 | 0.751 |
| Postprocessed | 0.584 | 0.880 | 0.745 | 0.745 |

Postprocessing (connected component removal) did not help: `for_which_classes: []`. Raw predictions are used as final.

### Holdout evaluation (162 cases, disjoint from training)

| Method | NETC | SNFH | ET | RC |
|--------|------|------|----|----|
| Baseline (5-fold ensemble) | TBD | TBD | TBD | TBD |
| CurriculumGAN (5-fold ensemble) | TBD | TBD | TBD | TBD |

### Comparison with challenge results

All scores are **lesion-wise Dice**. Different evaluation sets are noted.

| Method | NETC | SNFH | ET | RC | Eval Set | Source |
|--------|------|------|----|----|----------|--------|
| **Ours: Baseline MedNeXt-B k5** | 0.656 | 0.867 | 0.755 | 0.771 | Internal CV (5-fold) | This repo |
| **Ours: CurriculumGAN MedNeXt-B k5** | TBD | TBD | TBD | TBD | Internal CV | This repo |
| **Ours: Baseline (holdout)** | TBD | TBD | TBD | TBD | Holdout (162 cases) | This repo |
| **Ours: CurriculumGAN (holdout)** | TBD | TBD | TBD | TBD | Holdout (162 cases) | This repo |
| 2025 Winner: nnU-Net baseline | 0.821 | 0.818 | 0.812 | 0.894 | Internal test | [Jia et al. 2025](https://arxiv.org/abs/2509.24973) |
| 2025 Winner: Regular on-the-fly GAN | 0.824 | 0.815 | 0.813 | 0.883 | Internal test | [Jia et al. 2025](https://arxiv.org/abs/2509.24973) |
| 2025 Winner: Custom on-the-fly GAN | 0.830 | 0.802 | 0.813 | 0.888 | Internal test | [Jia et al. 2025](https://arxiv.org/abs/2509.24973) |
| 2025 Winner: 3-model ensemble | 0.833 | 0.813 | 0.812 | 0.893 | Internal test | [Jia et al. 2025](https://arxiv.org/abs/2509.24973) |
| 2025 Winner: 3-model ensemble | 0.749 | 0.825 | 0.790 | 0.872 | Online validation | [Jia et al. 2025](https://arxiv.org/abs/2509.24973) |
| 2024 Winner (Faking_it): best model | 0.787 | 0.870 | 0.756 | 0.705 | Online validation | [Ferreira et al. 2024](https://github.com/andre-fs-ferreira/BraTS_2023_2024_solutions) |
| 2024 Winner (Faking_it): 30-model ens. | -- | -- | -- | -- | Test (overall 0.873) | [Ferreira et al. 2024](https://doi.org/10.5281/zenodo.14001262) |

**Notes:**
- Our internal CV uses the v2 dataset (1,459 cases from Anthony's train_val_set). The 2025 winner used the full BraTS 2024 GLI training set.
- The 2025 winner's "internal test" is a held-out split from training data, comparable to our CV validation folds.
- The 2025 winner's "online validation" scores are from the official BraTS platform (different cases, typically harder).
- Our holdout set (162 cases) is disjoint from the 1,459 training cases, drawn from the full 1,621 BraTS-GLI 2024 corpus.
- The 2024 winner used a 30-checkpoint ensemble (3 architectures x 5 folds x 2 data configs). We use a single architecture.

## CurriculumGAN augmentation

Curriculum-scheduled on-the-fly GliGAN tumor injection for MedNeXt-B training. Pretrained GliGAN generators synthesize realistic tumors and inject them into healthy brain regions during training, with injection probability increasing over epochs.

**Curriculum schedule:**
- Phase 1 (epochs 0-299): No injection. Learn clean anatomical representations from real data only.
- Phase 2 (epochs 300-699): 30% injection rate. Model sees both real and synthetic tumors.
- Phase 3 (epochs 700-999): 50% injection rate. Aggressive augmentation as learning rate decays.

**Files:**
- `gligan_augment.py` -- GliGAN augmentation module (vectorized, lazy imports)
- `nnUNetTrainerV2_MedNeXt_CurriculumGAN.py` -- MedNeXt-B kernel5 trainer subclass
- `train_mednext_v2_curriculum_gan.slurm` -- SLURM script with checkpoint safety guards

**Dependencies:** MONAI (for loading pretrained Swin UNETR generator weights), scipy

**GliGAN weights:** Download from [Zenodo](https://doi.org/10.5281/zenodo.14001262) (39.8 GB archive). Extract the `brats2024/` checkpoint directory containing 4 modality generators (Swin UNETR, ~720 MB each) and 1 label generator (ConvTranspose3d, 91 MB). Set `GLIGAN_WEIGHTS_DIR` to the extracted `brats2024/` path.

### Training CurriculumGAN

```bash
# Install the trainer files into nnunet_mednext
MEDNEXT_DIR=$(python3 -c "import nnunet_mednext; import os; print(os.path.dirname(nnunet_mednext.__file__))")
cp gligan_augment.py "$MEDNEXT_DIR/training/network_training/MedNeXt/"
cp nnUNetTrainerV2_MedNeXt_CurriculumGAN.py "$MEDNEXT_DIR/training/network_training/MedNeXt/"

# Single fold
sbatch train_mednext_v2_curriculum_gan.slurm 0

# All 5 folds
sbatch --array=0-4 train_mednext_v2_curriculum_gan.slurm
```

## Training scripts

### Data setup

Convert BraTS-GLI format to nnU-Net v1 format using symlinks (saves ~35 GB):

```bash
# Default: Task500, uses ~/bmds260/data/BraTS-GLI/training_data1_v2
bash setup_brats_nnunet.sh

# Custom task ID and data directory
bash setup_brats_nnunet.sh 501 /scratch/users/abieleck/brats_2024/train_val_set
```

This creates `imagesTr/` and `labelsTr/` directories with symlinks, generates `dataset.json`, and saves an ID mapping JSON.

### Training (MedNeXt-B, FarmShare)

```bash
# Single fold
sbatch train_mednext_v2.slurm 0

# All 5 folds
for f in 0 1 2 3 4; do sbatch train_mednext_v2.slurm $f; done
```

Requires: `pip install --user --break-system-packages nnunet-mednext`

### Monitoring training

The CurriculumGAN SLURM script streams the nnU-Net training log to stdout in real time. Monitor a running job with:

```bash
# tail the SLURM output (includes epoch, loss, curriculum phase transitions)
tail -f ~/bmds260/logs/curgan_<JOBID>.out

# check all folds at once
for f in ~/bmds260/logs/curgan_*.out; do
  echo "--- $f ---"
  tail -3 "$f"
  echo
done
```

For baseline (non-GAN) training, the training log is in the nnU-Net results directory:

```bash
tail -f ~/bmds260/nnunet/results/nnUNet/3d_fullres/Task501_BraTSGLI_v2/\
  nnUNetTrainerV2_MedNeXt_B_kernel5__nnUNetPlansv2.1/fold_<N>/training_log_*.txt
```

### Holdout test set

Prepare the holdout set for inference after training completes:

```bash
bash prep_holdout.sh 501 ~/bmds260/holdout_test_set
```

Then run inference and evaluate:

```bash
# Inference (ensemble all 5 folds)
export nnUNet_raw_data_base=~/bmds260/nnunet/raw_data_base
export nnUNet_preprocessed=~/bmds260/nnunet/preprocessed
export RESULTS_FOLDER=~/bmds260/nnunet/results

nnUNetv2_predict \
  -i $nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTSGLI_v2/imagesTs \
  -o ~/bmds260/holdout_predictions \
  -tr nnUNetTrainerV2_MedNeXt_B_kernel5 \
  -t 501 -m 3d_fullres -f 0 1 2 3 4

# Evaluate
python3 eval_lesion_dice.py \
  --pred-dir ~/bmds260/holdout_predictions \
  --gt-dir ~/bmds260/holdout_gt \
  -o holdout_lesion_dice.json
```
