# Evaluation Quickstart

## Install

```bash
# FarmShare (scipy/nibabel/numpy already available)
python3 -m pip install --user --break-system-packages connected-components-3d

# Other environments
pip install connected-components-3d nibabel scipy numpy
```

## Run

```bash
# All metrics, all regions (default)
python3 eval_lesion_dice.py --pred-dir <predictions> --gt-dir <ground_truth> -o results.json

# nnU-Net shortcut (auto-finds paths)
python3 eval_lesion_dice.py --task 501 --fold 0
```

## Metrics

| Flag | Metric | Speed |
|------|--------|-------|
| `lw-dice` | Lesion-wise Dice (BraTS 2024 primary) | ~1 min/case |
| `lw-hd95` | Lesion-wise HD95 (337mm FP penalty) | ~2 min/case |
| `legacy-dice` | Standard voxel-level Dice | fast |
| `legacy-hd95` | Standard voxel-level HD95 | ~2 min/case |

Select with `--metrics`:
```bash
# Fast: lesion-wise Dice only
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --metrics lw-dice

# Paper: lesion-wise Dice + HD95
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --metrics lw-dice lw-hd95

# Everything
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --metrics lw-dice lw-hd95 legacy-dice legacy-hd95
```

## Regions

| Flag | Labels | Description |
|------|--------|-------------|
| `NETC` | 1 | Non-enhancing tumor core |
| `SNFH` | 2 | Surrounding non-enhancing FLAIR hyperintensity |
| `ET` | 3 | Enhancing tumor |
| `RC` | 4 | Resection cavity |
| `TC` | 1+3 | Tumor Core (NETC + ET) |
| `WT` | 1+2+3 | Whole Tumor (NETC + SNFH + ET, **no RC**) |

Select with `--regions`:
```bash
# Individual labels only (v1 behavior)
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --regions NETC SNFH ET RC

# Composite only
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --regions TC WT

# Specific combo
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --metrics lw-dice lw-hd95 --regions NETC ET TC WT
```

## Common recipes

### Quick check (fastest)
```bash
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt --metrics lw-dice --regions NETC SNFH ET RC -o quick.json
```

### Full paper table
```bash
python3 eval_lesion_dice.py --pred-dir preds --gt-dir gt -o full.json
```

### Holdout eval (after inference)
```bash
# Run inference first
export PATH=$HOME/.local/bin:$PATH
mednextv1_predict \
  -i ~/bmds260/nnunet/raw_data_base/nnUNet_raw_data/Task501_BraTSGLI_v2/imagesTs \
  -o ~/bmds260/holdout_predictions \
  -t 501 -m 3d_fullres \
  -tr nnUNetTrainerV2_MedNeXt_B_kernel5 \
  -f 0 1 2 3 4

# Then eval
python3 eval_lesion_dice.py \
  --pred-dir ~/bmds260/holdout_predictions \
  --gt-dir ~/bmds260/holdout_gt \
  -o holdout_results.json
```

### CurriculumGAN holdout eval
```bash
mednextv1_predict \
  -i ~/bmds260/nnunet/raw_data_base/nnUNet_raw_data/Task501_BraTSGLI_v2/imagesTs \
  -o ~/bmds260/holdout_predictions_curgan \
  -t 501 -m 3d_fullres \
  -tr nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN \
  -f 0 1 2 3 4

python3 eval_lesion_dice.py \
  --pred-dir ~/bmds260/holdout_predictions_curgan \
  --gt-dir ~/bmds260/holdout_gt \
  -o holdout_curgan_results.json
```

### Cross-validation fold eval
```bash
python3 eval_lesion_dice.py --task 501 --fold 0
python3 eval_lesion_dice.py --task 501 --fold 1
python3 eval_lesion_dice.py --task 501 --fold 2
python3 eval_lesion_dice.py --task 501 --fold 3
python3 eval_lesion_dice.py --task 501 --fold 4
```

### Different trainer
```bash
python3 eval_lesion_dice.py --task 501 --fold 0 --trainer nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN
```

## Output

JSON with `summary` (per-region aggregates) and `per_case` (per-patient detail):

```
NETC : lw_dice=0.7347  lw_hd95=36.33  legacy_dice=0.6975  legacy_hd95=41.14  (n=81)
SNFH : lw_dice=0.8895  lw_hd95=9.01   legacy_dice=0.9110  legacy_hd95=4.51   (n=162)
ET   : lw_dice=0.7816  lw_hd95=25.30  legacy_dice=0.7962  legacy_hd95=18.67  (n=132)
RC   : lw_dice=0.7783  lw_hd95=32.89  legacy_dice=0.7580  legacy_hd95=22.99  (n=143)
TC   : lw_dice=0.7886  lw_hd95=24.10  legacy_dice=0.8014  legacy_hd95=18.69  (n=132)
WT   : lw_dice=0.8796  lw_hd95=13.18  legacy_dice=0.9206  legacy_hd95=2.36   (n=162)
```

Regions absent in both GT and prediction are marked `"absent"` and excluded from averages.

## Time estimates

| Config | Per case | 162 cases |
|--------|----------|-----------|
| `--metrics lw-dice` | ~1 min | ~3 hours |
| `--metrics lw-dice lw-hd95` | ~3 min | ~8 hours |
| All metrics, all regions | ~4 min | ~10 hours |
| `--metrics lw-dice --regions NETC SNFH ET RC` | ~1 min | ~3 hours |

Run on a SLURM compute node for large sets:
```bash
sbatch --partition=normal --time=12:00:00 --cpus-per-task=4 --mem=64G \
  --wrap="python3 ~/bmds260/eval_lesion_dice.py --pred-dir preds --gt-dir gt -o results.json"
```

Or run directly on the rice login node for smaller jobs (no queue wait).
