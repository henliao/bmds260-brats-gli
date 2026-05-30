# Evaluation Findings

## Holdout Set (162 cases, disjoint from training)

### Baseline MedNeXt-B kernel 5x5x5 (5-fold ensemble, 1000 epochs)

| Region | LW-Dice | LW-HD95 (mm) | Legacy Dice | Legacy HD95 (mm) | n |
|--------|---------|--------------|-------------|------------------|---|
| NETC | 0.735 | 36.3 | 0.698 | 41.1 | 81 |
| SNFH | 0.890 | 9.0 | 0.911 | 4.5 | 162 |
| ET | 0.782 | 25.3 | 0.796 | 18.7 | 132 |
| RC | 0.778 | 32.9 | 0.758 | 23.0 | 143 |
| TC | 0.789 | 24.1 | 0.801 | 18.7 | 132 |
| WT | 0.880 | 13.2 | 0.921 | 2.4 | 162 |

### CurriculumGAN MedNeXt-B kernel 5x5x5 (5-fold ensemble, 1000 epochs)

| Region | LW-Dice | LW-HD95 (mm) | Legacy Dice | Legacy HD95 (mm) | n |
|--------|---------|--------------|-------------|------------------|---|
| NETC | 0.738 | 32.3 | 0.702 | 44.7 | 81 |
| SNFH | 0.895 | 7.2 | 0.911 | 4.2 | 162 |
| ET | 0.784 | 23.5 | 0.800 | 18.7 | 132 |
| RC | 0.769 | 33.7 | 0.743 | 31.8 | 145 |
| TC | 0.789 | 23.6 | 0.805 | 18.7 | 132 |
| WT | 0.882 | 12.6 | 0.920 | 2.0 | 162 |

### CurriculumGAN vs Baseline (5-fold, delta)

| Region | LW-Dice | LW-HD95 (mm) | Legacy Dice | Legacy HD95 (mm) |
|--------|---------|--------------|-------------|------------------|
| NETC | **+0.003** | **-4.0** | +0.004 | +3.6 |
| SNFH | **+0.006** | **-1.8** | +0.001 | -0.3 |
| ET | **+0.002** | **-1.8** | +0.004 | 0.0 |
| RC | -0.009 | +0.8 | -0.015 | +8.8 |
| TC | 0.000 | **-0.5** | +0.004 | 0.0 |
| WT | **+0.002** | **-0.6** | -0.001 | -0.4 |

Key observations:
- CurriculumGAN improves LW-Dice on 5/6 regions, with SNFH (+0.006) and NETC (+0.003) showing the largest gains. Only RC regresses (-0.009).
- LW-HD95 (mean) improves on 5/6 regions. NETC sees the largest boundary improvement (-4.0mm), followed by SNFH and ET (-1.8mm each). Only RC is slightly worse (+0.8mm).
- Median LW-HD95 is nearly identical between methods (1.21mm baseline vs 1.25mm CurGAN), indicating the mean HD95 improvements are driven by fewer catastrophic outlier cases rather than uniform boundary gains.
- The adaptive GAN augmentation curriculum helps the model generalize to rare/difficult cases that produce extreme HD95 values in the baseline.
- RC is the only consistently underperforming region. CurriculumGAN may over-regularize resection cavity boundaries, which have highly variable morphology.

### CurriculumGAN preliminary results (partial ensembles)

Earlier runs with incomplete fold ensembles, included for reference:

| Region | 3-fold (folds 1-3) | 4-fold (folds 1-4) | 5-fold (final) | Baseline |
|--------|--------------------|--------------------|----------------|----------|
| NETC | 0.738 | 0.725 | **0.738** | 0.735 |
| SNFH | 0.888 | 0.892 | **0.895** | 0.890 |
| ET | 0.785 | 0.774 | **0.784** | 0.782 |
| RC | 0.769 | 0.778 | 0.769 | **0.778** |

Adding fold 0 to the ensemble recovered NETC (+0.013 vs 4-fold) and ET (+0.010 vs 4-fold). Fold 0 was the weakest fold (NETC EMA 0.751) but its inclusion improved ensemble diversity.

## Cross-Validation (1459 training cases, per-fold)

### Baseline MedNeXt-B kernel 5x5x5 (lesion-wise Dice)

| Fold | NETC | SNFH | ET | RC |
|------|------|------|----|----|
| 0 | 0.657 | 0.854 | 0.752 | 0.753 |
| 1 | 0.616 | 0.879 | 0.752 | 0.777 |
| 2 | 0.724 | 0.859 | 0.751 | 0.787 |
| 3 | 0.653 | 0.873 | 0.754 | 0.777 |
| 4 | 0.630 | 0.869 | 0.765 | 0.761 |
| **Mean** | **0.656** | **0.867** | **0.755** | **0.771** |
| Std | 0.040 | 0.010 | 0.006 | 0.013 |

### CurriculumGAN cross-validation

TBD (requires running eval on each fold's validation_raw predictions).

### HD Loss cross-validation

TBD (training in progress: fold 0 at epoch 658, fold 1 at 494, fold 2 at 590, fold 3 at 238).
