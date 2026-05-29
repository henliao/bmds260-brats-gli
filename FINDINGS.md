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

### CurriculumGAN MedNeXt-B kernel 5x5x5 (4-fold ensemble, folds 1-4, 1000 epochs)

| Region | LW-Dice | LW-HD95 (mm) | Legacy Dice | Legacy HD95 (mm) | n |
|--------|---------|--------------|-------------|------------------|---|
| NETC | 0.725 | 36.3 | 0.698 | 44.7 | 81 |
| SNFH | 0.892 | 8.2 | 0.910 | 4.5 | 162 |
| ET | 0.774 | 25.9 | 0.796 | 18.6 | 132 |
| RC | 0.778 | 28.1 | 0.742 | 29.7 | 145 |
| TC | 0.779 | 26.0 | 0.802 | 18.7 | 132 |
| WT | 0.885 | 11.4 | 0.920 | 2.4 | 162 |

Note: CurriculumGAN uses 4-fold ensemble (fold 0 still training, epoch ~762). Baseline uses 5-fold ensemble. Not a fair comparison until fold 0 completes.

### CurriculumGAN vs Baseline (delta)

| Region | LW-Dice | LW-HD95 | Legacy Dice | Legacy HD95 |
|--------|---------|---------|-------------|-------------|
| NETC | -0.010 | 0.0 | 0.000 | +3.6 |
| SNFH | **+0.002** | **-0.8** | -0.001 | 0.0 |
| ET | -0.008 | +0.6 | 0.000 | -0.1 |
| RC | 0.000 | **-4.8** | -0.016 | +6.7 |
| TC | -0.010 | +1.9 | +0.001 | 0.0 |
| WT | **+0.005** | **-1.8** | -0.001 | 0.0 |

Key observations:
- CurriculumGAN shows HD95 improvements on SNFH (-0.8mm), RC (-4.8mm), and WT (-1.8mm), suggesting better boundary delineation on these regions.
- LW-Dice is mostly flat. NETC and TC slightly down, SNFH and WT slightly up.
- Legacy Dice nearly identical across all regions.
- NETC LW-Dice (-0.010) may recover when fold 0 is included (fold 0 currently shows strong NETC EMA 0.751 with GAN at 0.93).
- RC Legacy HD95 is worse (+6.7mm) despite better LW-HD95 (-4.8mm), indicating the model produces fewer but better-localized RC lesions.

### CurriculumGAN (3-fold ensemble, folds 1-3, v1 eval)

Earlier preliminary run with only 3 folds and v1 metrics (LW-Dice only, individual labels only):

| Region | CG 3-fold | Baseline 5-fold | Delta |
|--------|-----------|-----------------|-------|
| NETC | 0.738 | 0.735 | +0.003 |
| SNFH | 0.888 | 0.890 | -0.002 |
| ET | 0.785 | 0.782 | +0.003 |
| RC | 0.769 | 0.778 | -0.009 |

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

TBD (training in progress: fold 0 at epoch 372, fold 1 at 229, fold 2 at 168, folds 3-4 pending).
