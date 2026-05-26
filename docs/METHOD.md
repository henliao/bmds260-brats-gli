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

**Bidirectional adjustment.** Rates can go up (plateau) and down (improving). This prevents the ratchet problem where noise-triggered plateaus permanently inflate the rate. If NETC ramps up to 0.40 but then starts improving again, the rate decays back toward the 0.30 baseline.

**NETC-specific parameters:**
- Plateau threshold: 0.005 (vs 0.01 for other classes). Triggers earlier.
- Max injection rate: 0.85 in code, overridable to 0.95 via live tune file (vs 0.50 for other classes). Allows substantially more augmentation headroom.
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

At baseline rate (0.30): threshold = 0.5 (normal, matches pretrained behavior).
At max NETC rate (0.85-0.95): threshold approaches -0.2 (roughly 3x larger NETC region).

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
| `gan_base_prob` | 0.30 | Half the rate used by Jain et al. (2025), leaving headroom for adaptive ramping |
| `gan_max_prob` | 0.50 | Standard classes |
| `gan_max_prob_netc` | 0.85 (code), 0.95 (tune file) | NETC needs substantially more headroom; raised during training based on observed ceiling hits |
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

The first adaptive version only increased rates (ratchet). The problem: online validation Dice is noisy at the patch level. A random downswing in NETC Dice at epoch 55 triggers plateau detection, bumping the rate from 0.30 to 0.35. If the noise never reverses long enough to clear the lookback window, the rate climbs to the ceiling by epoch 200 and stays there forever, effectively becoming a fixed high-rate schedule.

Bidirectional adjustment means: if NETC starts improving again (EMA delta > 0.01 over 50 epochs), the rate decays back toward 0.30. The augmentation rate tracks the actual need.

### Why EMA and not raw Dice for plateau detection?

Online validation Dice per class, per epoch, on random patches:
- SNFH (large region): relatively stable, std ~0.02 epoch-to-epoch
- NETC (small, sparse): highly variable, std ~0.10 epoch-to-epoch

Without smoothing, NETC Dice looks like noise to the plateau detector. EMA with alpha=0.1 acts as a low-pass filter, preserving the trend while suppressing per-epoch variance.

### Why not per-class independent injection?

`get_gan_probability()` returns `max(per_class_gan_prob)`. This means if NETC rate is 0.45 and SNFH rate is 0.30, every augmented sample gets a whole tumor (all 4 classes). SNFH gets more augmentation than its rate suggests.

This is a deliberate simplification. GliGAN generates whole tumors; you can't synthesize NETC without also synthesizing the surrounding SNFH/ET/RC. The per-class rates don't independently control separate augmentations. They're "votes" for how much total GAN augmentation the training needs, and the highest vote wins.

The class-weighted thresholds partially decouple this: when NETC is the driver, its threshold drops (larger NETC region), while SNFH stays at 0.5 (normal size). The net effect is proportionally more NETC voxels per synthetic tumor even though all classes are present.

A more precise approach would train separate per-class GANs, but GliGAN is a whole-tumor generator and can't be decomposed without retraining.

## Expected training behavior

### Epochs 0-50 (warmup)
All classes at baseline 0.30 injection rate, threshold 0.5. No adaptation yet (lookback window filling). GliGAN loads at epoch 0 (~30 seconds). Base rate of 0.30 chosen as half the 60-75% used by Jain et al. (2025 BraTS winner), leaving headroom for adaptive ramping while providing meaningful synthetic diversity from the start.

### Epochs 50-300 (early adaptation)
Plateau detection begins. SNFH likely improving steadily (no plateau). NETC may show early plateaus due to data scarcity. Expect NETC rate to start climbing first.

### Epochs 300-700 (mid training)
Learning rate decays (nnU-Net cosine schedule). Classes that were improving start to slow. More plateau events. Expect per-class rates to diverge: NETC at 0.40-0.85, others at 0.30-0.50.

### Epochs 700-1000 (late training)
Most classes plateauing as learning rate approaches minimum. GAN rates at or near maximum. NETC may reach the 0.85-0.95 ceiling. Threshold modulation producing large NETC regions. This is where the method should have the most impact: the model sees maximally diverse NETC examples precisely when it has the least capacity to learn from real data alone.

## Ablation structure

| Experiment | Description |
|------------|-------------|
| A: Baseline | MedNeXt-B kernel 5, Dice+CE loss, no GAN, 1000 epochs |
| B: HD Boundary Loss | MedNeXt-B kernel 5, Dice+CE loss (epochs 0-799) + Dice+CE+Boundary loss (epochs 800-999), no GAN |
| C: Adaptive GAN | MedNeXt-B kernel 5, Dice+CE loss, performance-adaptive GliGAN, 1000 epochs |

All experiments use the same 5-fold cross-validation split (1459 cases, 292 per fold) and holdout set (162 cases). Lesion-wise Dice is the primary metric (BraTS 2024 challenge standard).

Experiment B (HD boundary loss) tests whether a boundary-aware loss function (Kervadec et al. 2019) improves segmentation of small, boundary-critical regions like NETC without any synthetic data. It uses signed distance transforms from ground truth labels to penalize predictions that are geometrically far from the true boundary, staged late in training (epoch 800+) to avoid destabilizing early convergence.

## Literature context

### Related work in detail

#### GAN-based data augmentation for medical image segmentation

GAN augmentation for brain tumor segmentation has become standard among top BraTS challenge performers:

- **GliGAN (Ferreira et al. 2024)**: The GAN we use. Generates whole synthetic brain tumors (label mask + 4-modality MRI appearance) via a ConvTranspose3d label generator and 4x Swin UNETR image generators. Won the BraTS 2024 Adult Glioma challenge. Uses a fixed injection rate throughout training.

- **Jain et al. (2025)**: BraTS 2025 winner. Used GliGAN at a fixed 60-75% injection rate for the full 1000 epochs. No per-class adaptation, no scheduling. Demonstrated that high fixed rates work well, but did not investigate whether the rate could be optimized.

- **Shin et al. (2018)**: Early work on GAN augmentation for brain lesion segmentation using progressive GAN (Karras et al.). Showed synthetic images improve segmentation when real data is scarce. Fixed augmentation rate.

- **Han et al. (2019)**: GAN augmentation for liver lesion segmentation (progressive GAN). Explored different mixing ratios of real vs. synthetic data but used a fixed ratio during training, not adaptive.

All prior GAN augmentation work in medical imaging uses static injection rates. None adapts the rate based on per-class training dynamics.

#### Curriculum learning

- **Bengio et al. (2009)**: Foundational curriculum learning paper. Key idea: present training samples in order of increasing difficulty. The curriculum is over sample ordering, not augmentation intensity.

- **Self-paced learning (Kumar et al. 2010)**: Extended curriculum learning with automatic difficulty assessment. Samples are weighted by current model loss. Operates on sample selection, not on augmentation parameters.

- **CurriculumNet (Guo et al. 2018)**: Curriculum over data complexity for noisy web data. Again, sample-level ordering.

Our method applies the curriculum principle to augmentation intensity rather than sample ordering. The "difficulty" signal is per-class validation Dice (a class-level signal), not per-sample loss.

#### Adaptive augmentation

- **Adaptive Discriminator Augmentation (ADA, Karras et al. 2020)**: Adapts augmentation probability during GAN training to prevent discriminator overfitting. Uses a single scalar heuristic (ratio of real vs. fake discriminator outputs). Our method: (1) adapts augmentation for the segmentation model, not the GAN's discriminator, (2) uses per-class signals instead of a global scalar, (3) adjusts bidirectionally.

- **Differentiable Augmentation (DiffAug, Zhao et al. 2020)**: Makes augmentation differentiable for end-to-end GAN training. Orthogonal to our approach: DiffAug is about what augmentations to apply during GAN training; ours is about when and how much to apply GAN output during segmentation training.

- **UADA (Uncertainty-Aware Data Augmentation, Li et al. 2021)**: Uses model uncertainty to guide augmentation in domain adaptation. The signal is uncertainty-based, applied to domain shifts, not class-specific performance. Unidirectional: augmentation only increases with uncertainty.

- **Sample-adaptive progressive augmentation**: Some training pipelines increase augmentation over time (e.g., start with weak augmentation, ramp to strong). These use fixed schedules (linear, cosine). No feedback loop from validation metrics.

None of the above combines per-class adaptation with bidirectional rate control based on smoothed validation metrics.

#### Boundary-aware loss functions

- **Kervadec et al. (2019)**: Introduced the boundary loss, which uses signed distance transforms of ground truth labels to provide a differentiable approximation of the Hausdorff distance. Designed for class-imbalanced segmentation where regional losses (Dice, cross-entropy) underweight boundary accuracy for small structures.

- **Isensee et al. (2021)**: nnU-Net uses Dice + cross-entropy as default. No boundary term. The framework was designed for generality across tasks, not optimized for specific class imbalance patterns.

- **Staged loss introduction**: Training with boundary loss from epoch 0 can destabilize convergence because distance-based gradients dominate when predictions are far from ground truth. Common practice is to introduce boundary loss after the model has learned rough region shapes (we use epoch 800 of 1000).

### Novelty assessment

| Component | Established? | Our specific use |
|-----------|-------------|-----------------|
| GAN augmentation for brain tumors | Yes (Ferreira 2024, Jain 2025, Shin 2018) | Same GAN, but with adaptive scheduling |
| Curriculum learning | Yes (Bengio 2009), but for sample ordering | Applied to augmentation intensity, not samples |
| Adaptive augmentation rate | Partially (ADA for GAN discriminators) | Applied to downstream segmentation training, per-class |
| Bidirectional rate adjustment | **Novel** | Rates ramp up on plateau AND decay on improvement |
| Per-class adaptive augmentation | **Novel in this form** | Per-class Dice drives per-class GAN rates and thresholds |
| EMA-smoothed plateau detection | Standard technique, **novel application** | Applied to patch-based validation Dice for augmentation control |
| Threshold modulation on GAN label output | **Novel** | Per-class label generation thresholds driven by adaptive rates |
| Live parameter tuning via file | Engineering contribution | Supports mid-training ceiling adjustments without job restarts |

The specific combination is novel: per-class EMA-smoothed validation Dice driving bidirectional per-class GAN injection rates with coupled threshold-modulated label generation. Individual building blocks are well-established.

The closest existing work is ADA (Karras et al. 2020), which adapts augmentation probability based on a training signal. Key differences: (1) ADA uses a single global signal; ours uses per-class signals. (2) ADA is for GAN training stability; ours is for downstream segmentation. (3) ADA is monotonic within a window; ours is bidirectional across the full training run.

## Strengths and weaknesses

### Strengths (for paper discussion)

1. **Principled, not heuristic.** The method directly connects augmentation intensity to measured learning dynamics per class, rather than relying on hand-tuned schedules. The same mechanism applies across classes with class-specific parameterization only for NETC (justified by its extreme scarcity).

2. **Bidirectional control prevents rate inflation.** Unlike ratchet-only approaches, rates can decay when the model improves. This means the system self-corrects: transient noise in validation Dice doesn't permanently inflate augmentation rates. The training converges to augmentation levels that reflect actual need.

3. **No additional training data or models required.** Uses the same pretrained GliGAN as prior work (Ferreira et al. 2024, Jain et al. 2025). The contribution is entirely in the scheduling and label generation, making it a drop-in replacement for fixed-rate approaches.

4. **Class-specific threshold modulation is cheap.** Compared to rejection sampling (up to 5x GAN forward passes), threshold modulation adds zero extra GAN calls. The per-class label sizes are controlled by a single scalar threshold per class, with negligible compute cost.

5. **Checkpoint persistence enables practical deployment.** Adaptive state survives wall-time restarts (common on shared HPC clusters like FarmShare with 48h limits). Without this, multi-day training runs would lose adaptive state at every restart boundary.

6. **Ablation isolates contributions.** Three-experiment design (baseline, HD loss, adaptive GAN) cleanly separates the effect of boundary-aware loss (Experiment B) from GAN augmentation (Experiment C). Same architecture, same splits, same evaluation.

7. **Live tuning supports rapid iteration.** The tune file mechanism (`curgan_tune.json`) allows parameter adjustments during multi-day training without killing jobs. Useful for correcting ceilings that prove too conservative (as happened: NETC ceiling raised from 0.60 to 0.85 to 0.95 based on observed training dynamics).

### Weaknesses and limitations (for paper discussion)

1. **Online Dice is a noisy signal.** nnU-Net's online validation Dice is computed from random patches, not full-volume inference. For NETC (present in <50% of cases, small when present), the signal has high variance even after EMA smoothing. The plateau detector may still trigger on noise rather than true stagnation, particularly during early training when EMA values are less stable. Mitigation: EMA smoothing and bidirectional control limit the damage, but the signal quality is fundamentally limited.

2. **Cannot disentangle GAN quantity from GAN quality.** When the adaptive system ramps NETC augmentation, it increases both the volume of synthetic data and the bias toward larger NETC regions (via threshold modulation). If results improve, we cannot attribute the gain to "more augmentation" vs. "differently shaped augmentation" without additional ablation (e.g., adaptive rate with fixed threshold, or fixed rate with adaptive threshold). This would require 2 more 5-fold experiments (not feasible in the remaining compute budget).

3. **GliGAN generates whole tumors, limiting per-class independence.** The max-over-classes injection strategy means augmenting NETC also augments SNFH/ET/RC. If SNFH is already converged and additional augmentation introduces noise, the method could hurt high-performing classes while helping low-performing ones. The threshold modulation partially mitigates this (SNFH voxels stay at normal proportions), but the total number of synthetic SNFH voxels still increases.

4. **Parameter sensitivity not fully explored.** The method has 11 tunable parameters (see Parameters table). The NETC-specific values (lower plateau threshold, higher ceiling) were set based on the observed baseline performance gap, not systematic search. A different dataset or class imbalance pattern might require different settings. We report results for one parameter configuration.

5. **No comparison to fixed high-rate GAN.** Jain et al. (2025) used 60-75% fixed injection. Our adaptive approach starts at 30% and ramps. If the optimal strategy is simply "use as much GAN data as possible from epoch 0," the adaptive overhead is wasted. A fixed-60% baseline would test this. Compute constraints prevent running this additional experiment.

6. **Two-experiment ablation (not counting HD loss).** The GAN comparison is binary: baseline vs. adaptive. Without intermediate experiments (fixed-rate GAN, adaptive-rate but uniform threshold, etc.), the relative contribution of each component is unclear. We discuss this as future work.

7. **Wall-time restart introduces a confound.** At ~640 epochs, training is interrupted and resumed from the last checkpoint (every 50 epochs, so up to 49 epochs of work is lost). The adaptive state checkpoints correctly, but the model weights roll back. This means the resumed training re-traverses epochs the model already saw, potentially with different adaptive rates than the first pass. This is a practical limitation of the HPC environment, not the method itself, but it affects reproducibility.

8. **Single architecture, single dataset.** Validated only on MedNeXt-B with BraTS 2024 Adult Glioma. Generalization to other architectures (nnU-Net default, SwinUNETR), other GAN augmenters, or other datasets (BraTS Pediatric, BraTS Meningioma) is untested.

9. **Threshold modulation changes tumor anatomy.** Lowering the NETC threshold from 0.5 to -0.2 roughly triples the NETC region. At aggressive thresholds, the synthetic NETC region may become unrealistically large compared to real NETC distributions. The anatomical enforcement step (dilation boundary, minimum component size) prevents gross violations, but subtler distributional shifts in region size could introduce bias. We do not quantitatively measure synthetic vs. real NETC size distributions.

### Recommendations for paper framing

- Frame the contribution as **"adaptive augmentation scheduling"**, not as a new GAN or a new architecture. The novelty is in the control loop, not the components.
- Acknowledge that the approach is most valuable when class imbalance is severe and one class clearly underperforms. In balanced settings, the adaptive mechanism has less to act on.
- Position the live tuning mechanism as a practical engineering contribution for HPC training, not as part of the core method.
- If NETC Dice improves meaningfully (>0.02 over baseline), highlight the bidirectional mechanism as the key differentiator vs. simply running GliGAN at a higher fixed rate.
- If results are modest or mixed, frame as: "adaptive scheduling concentrates augmentation where needed, but the noisy online Dice signal limits the precision of the adaptation."
