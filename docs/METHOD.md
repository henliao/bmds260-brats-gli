# Performance-Adaptive GAN Augmentation for BraTS Segmentation

## Overview

This document describes the complete method for performance-adaptive GliGAN augmentation in the MedNeXt-B brain tumor segmentation pipeline. The method addresses a core limitation of nnU-Net's static augmentation: it configures data augmentation once at the start of training and keeps it fixed for all 1000 epochs, which is suboptimal for rare classes (NETC: ~2-5% of tumor voxels) that hit a diversity bottleneck before the model converges.

Our approach adapts GAN-based augmentation per-class during training based on smoothed validation Dice trajectories, concentrating synthetic data where the model needs it most.

## Motivation

### The NETC problem

In BraTS 2024, NETC (non-enhancing tumor core, label 1) is the hardest class:
- Smallest region: ~2-5% of tumor voxels
- Most variable: present in only 659 of 1459 training cases (45%)
- Baseline Dice: 0.656 (vs SNFH 0.867, ET 0.755, RC 0.771)

Standard nnU-Net augmentation (rotation, scaling, mirroring, gamma correction) doesn't help NETC specifically. It augments all classes equally.

### Why GAN augmentation

GliGAN (Ferreira et al. 2024) generates realistic synthetic brain tumors by:
1. Generating a random tumor label mask (ConvTranspose3d label generator)
2. Placing it in a healthy brain region
3. Synthesizing MRI appearance for each modality (4x Swin UNETR generators)

This directly increases training diversity for rare classes. The 2024 and 2025 BraTS challenge winners both used GliGAN augmentation.

### Why adaptive scheduling

Prior work uses fixed GAN injection rates for all 1000 epochs. This has three problems:

1. **No class targeting.** A fixed 30% injection rate augments all classes equally. SNFH (already at 0.867) gets the same extra data as NETC (at 0.656).

2. **No response to training dynamics.** The model's learning trajectory varies by class. SNFH converges quickly; NETC plateaus around epoch 400-600. A fixed schedule can't respond to this.

3. **Wasted computation.** Running GAN augmentation at full rate during epochs where all classes are improving rapidly adds noise without benefit.

## Method

### Architecture

Two files implement the method:

- `gligan_augment.py`: GliGAN augmenter with class-weighted label generation
- `nnUNetTrainerV2_MedNeXt_CurriculumGAN.py`: MedNeXt-B trainer subclass with adaptive scheduling

### Component 1: Performance-adaptive injection rate

**EMA-smoothed per-class Dice tracking.** After each epoch, nnU-Net computes online validation Dice per class (from accumulated TP/FP/FN over validation patches). We capture these values before the parent class resets its accumulators, and smooth them with an exponential moving average (alpha=0.1) to filter patch-level noise.

**Why EMA, not raw values.** Online validation Dice is computed on random patches, not full volumes. For NETC (small, sparse), this is very noisy: a single epoch's Dice can swing from 0.15 to 0.35 randomly. Without smoothing, plateau detection would trigger on noise rather than true stagnation.

**Plateau detection.** Every epoch after a 50-epoch warmup period, we compare the current EMA to the EMA from 50 epochs ago:

```
improvement = ema_dice[now] - ema_dice[now - 50]
```

Three outcomes:
- `improvement < plateau_threshold`: class is plateauing, increase GAN rate by 0.05
- `improvement > improvement_threshold`: class is improving, decrease GAN rate by 0.03 (toward baseline)
- Between thresholds: hold steady

**Bidirectional adjustment.** Rates can go up (plateau) and down (improving). This prevents the ratchet problem where noise-triggered plateaus permanently inflate the rate. If NETC ramps up to 0.40 but then starts improving again, the rate decays back toward the 0.15 baseline.

**NETC-specific parameters:**
- Plateau threshold: 0.005 (vs 0.01 for other classes). Triggers earlier.
- Max injection rate: 0.60 (vs 0.50 for other classes). Allows more augmentation.
- Improvement threshold for decay: 0.01 (vs 0.02). Slower to ease off.

The overall injection probability is the maximum across all per-class rates. When any class is plateauing, augmentation increases for all classes (since GliGAN generates whole tumors with all labels), but the label generation is biased toward the plateauing class via threshold modulation.

### Component 2: Class-weighted label generation

**The problem with uniform generation.** GliGAN's label generator outputs a continuous activation volume per class (tanh, range [-1, 1]). The original code thresholds all channels at 0.5 to produce binary masks. This generates tumors with roughly equal proportions of each class, regardless of which class the model is struggling with.

**Threshold modulation.** Instead of a fixed 0.5 threshold for all classes, we modulate the threshold per class based on its current GAN injection rate:

```
threshold = 0.5 - fraction * (0.5 - (-0.2))
         = 0.5 - fraction * 0.7
```

where `fraction = (current_rate - base_rate) / (max_rate - base_rate)`.

At baseline rate (0.15): threshold = 0.5 (normal, matches pretrained behavior).
At max NETC rate (0.60): threshold = -0.2 (roughly 3x larger NETC region).

**Why this works.** The label generator's per-channel activations have a gradient around each class boundary. Voxels near the center of a predicted NETC region have high activations (0.8-0.9); voxels at the edge are medium (0.3-0.5); voxels far outside are low (-0.5 to 0.0). Lowering the threshold expands the class region outward into the medium-confidence zone, producing a physically larger NETC region in the synthetic tumor.

This is strictly better than rejection sampling (generating multiple tumors and picking the one with the most NETC), because:
- One GAN call per augmented sample, not up to 5
- Continuously tunable per class (not binary "has NETC or not")
- The generated tumor shape remains realistic (driven by the pretrained generator), just with larger class-specific regions

**Anatomical enforcement.** After threshold modulation, `_enforce_anatomy()` cleans up:
1. NETC and ET voxels outside the dilated whole-tumor boundary are removed (prevents anatomically impossible floating labels)
2. Tiny isolated components (<5 voxels) per class are filtered

These operations use scipy.ndimage (binary_dilation, connected components) on a 96^3 volume, adding ~2ms per call.

### Component 3: Checkpoint persistence

**The problem.** FarmShare GPU jobs have a 48-hour wall-time limit. At ~4.5 min/epoch, each fold reaches ~640 epochs before timeout. The remaining ~360 epochs require a second submission with `--continue_training`.

Without checkpoint persistence, the adaptive state (EMA history, per-class rates, per-class thresholds) resets to baseline on resume. The lookback window (50 epochs) must refill before any adaptation kicks in. This means epochs 640-690 run with baseline augmentation regardless of where the model was at epoch 639.

**The fix.** We override `save_checkpoint` and `load_checkpoint_ram`:
- `save_checkpoint`: after the parent saves the standard nnU-Net checkpoint, we reload it, append our adaptive state dict (per_class_gan_prob, ema_dice, ema_history), and re-save.
- `load_checkpoint_ram`: after the parent restores model weights and optimizer, we extract and restore our adaptive state from the same checkpoint.

This ensures the augmentation schedule is continuous across wall-time restarts.

**Note on torch.load.** PyTorch 2.6 changed `torch.load` default to `weights_only=True`, which rejects numpy scalars in nnU-Net checkpoints. Our `save_checkpoint` must use `weights_only=False` when reloading the checkpoint for appending.

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gan_base_prob` | 0.15 | Low enough to not dominate early training, high enough for signal |
| `gan_max_prob` | 0.50 | Standard classes |
| `gan_max_prob_netc` | 0.60 | NETC needs more headroom |
| `gan_ramp_step` | 0.05 | Per plateau detection event |
| `gan_decay_step` | 0.03 | Slower decay than ramp (asymmetric, biased toward augmentation) |
| `ema_alpha` | 0.1 | Smoothing factor. Lower = smoother. 0.1 gives ~10-epoch effective window |
| `lookback` | 50 | Epochs to compare for plateau detection |
| `plateau_threshold` | 0.01 | Min EMA improvement over lookback (standard classes) |
| `plateau_threshold_netc` | 0.005 | Triggers NETC ramp earlier |
| `improvement_threshold` | 0.02 | EMA improvement that triggers rate decay |
| `improvement_threshold_netc` | 0.01 | NETC decays slower |
| `DEFAULT_TANH_THRESHOLD` | 0.5 | Matches pretrained label generator operating point |
| `MIN_LABEL_THRESH` | -0.2 | At max rate, threshold drops to -0.2 (~3x larger region) |

## Design decisions and alternatives considered

### Why not rejection sampling for NETC bias?

The first implementation used rejection sampling: generate up to 5 synthetic tumors, keep the one with the most NETC voxels. This was replaced with threshold modulation because:
- Rejection sampling runs 4 Swin UNETR forward passes per attempt (up to 20 total per sample). Threshold modulation runs 4 total.
- Rejection sampling is binary (tumor either has NETC or doesn't). Threshold modulation is continuously tunable.
- Rejection sampling can't control how much NETC. A tumor with 10 NETC voxels "counts" the same as one with 1000.

### Why not a fixed schedule?

The original implementation used three fixed phases:
- Phase 1 (0-299): 0% injection (baseline identical)
- Phase 2 (300-699): 30% injection
- Phase 3 (700-999): 50% injection

Problems:
1. 300 wasted epochs producing no new information
2. Arbitrary phase boundaries with no empirical justification
3. No NETC-specific targeting despite the docstring claiming it
4. No way to separate GAN effect from curriculum effect in ablation

### Why bidirectional rates, not ratchet-only?

The first adaptive version only increased rates (ratchet). The problem: online validation Dice is noisy at the patch level. A random downswing in NETC Dice at epoch 55 triggers plateau detection, bumping the rate from 0.15 to 0.20. If the noise never reverses long enough to clear the lookback window, the rate climbs to 0.60 by epoch 200 and stays there forever, effectively becoming a fixed high-rate schedule.

Bidirectional adjustment means: if NETC starts improving again (EMA delta > 0.01 over 50 epochs), the rate decays back toward 0.15. The augmentation rate tracks the actual need.

### Why EMA and not raw Dice for plateau detection?

Online validation Dice per class, per epoch, on random patches:
- SNFH (large region): relatively stable, std ~0.02 epoch-to-epoch
- NETC (small, sparse): highly variable, std ~0.10 epoch-to-epoch

Without smoothing, NETC Dice looks like noise to the plateau detector. EMA with alpha=0.1 acts as a low-pass filter, preserving the trend while suppressing per-epoch variance.

### Why not per-class independent injection?

`get_gan_probability()` returns `max(per_class_gan_prob)`. This means if NETC rate is 0.45 and SNFH rate is 0.15, every augmented sample gets a whole tumor (all 4 classes). SNFH gets more augmentation than its rate suggests.

This is a deliberate simplification. GliGAN generates whole tumors; you can't synthesize NETC without also synthesizing the surrounding SNFH/ET/RC. The per-class rates don't independently control separate augmentations. They're "votes" for how much total GAN augmentation the training needs, and the highest vote wins.

The class-weighted thresholds partially decouple this: when NETC is the driver, its threshold drops (larger NETC region), while SNFH stays at 0.5 (normal size). The net effect is proportionally more NETC voxels per synthetic tumor even though all classes are present.

A more precise approach would train separate per-class GANs, but GliGAN is a whole-tumor generator and can't be decomposed without retraining.

## Expected training behavior

### Epochs 0-50 (warmup)
All classes at baseline 0.15 injection rate, threshold 0.5. No adaptation yet (lookback window filling). GliGAN loads at epoch 0 (~30 seconds).

### Epochs 50-300 (early adaptation)
Plateau detection begins. SNFH likely improving steadily (no plateau). NETC may show early plateaus due to data scarcity. Expect NETC rate to start climbing first.

### Epochs 300-700 (mid training)
Learning rate decays (nnU-Net cosine schedule). Classes that were improving start to slow. More plateau events. Expect per-class rates to diverge: NETC at 0.30-0.50, others at 0.15-0.25.

### Epochs 700-1000 (late training)
Most classes plateauing as learning rate approaches minimum. GAN rates at or near maximum. Threshold modulation producing large NETC regions. This is where the method should have the most impact: the model sees maximally diverse NETC examples precisely when it has the least capacity to learn from real data alone.

## Ablation structure

| Experiment | Description |
|------------|-------------|
| Baseline | MedNeXt-B kernel 5, no GAN, 1000 epochs |
| Adaptive GAN | MedNeXt-B kernel 5, performance-adaptive GliGAN, 1000 epochs |

Both use the same 5-fold cross-validation split (1459 cases, 292 per fold) and holdout set (162 cases). Lesion-wise Dice is the primary metric (BraTS 2024 challenge standard).

## Literature context

| Component | Precedent | Our contribution |
|-----------|-----------|------------------|
| GliGAN augmentation | Ferreira et al. 2024 (BraTS winner), Jia et al. 2025 (BraTS 2025 winner) | Same GAN, but adaptive scheduling instead of fixed rate |
| Curriculum learning | Bengio et al. 2009 | Applied to GAN injection rate, not sample ordering |
| Adaptive LR scheduling | ReduceLROnPlateau (standard) | Same principle applied to augmentation intensity |
| Class-specific augmentation | nnU-Net foreground oversampling (0.33) | Per-class threshold modulation on GAN label output |
| EMA smoothing | Standard in RL (target networks), GAN training | Applied to validation Dice for noise-robust plateau detection |

The specific combination (EMA-smoothed per-class Dice driving per-class GAN injection rates with threshold-modulated label generation) is novel. Individual components are well-established.
