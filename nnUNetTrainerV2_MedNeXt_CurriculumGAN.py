"""
MedNeXt-B kernel5 trainer with performance-adaptive GliGAN augmentation.

Schedule adapts to per-class validation Dice:
  - GAN injection starts at epoch 0 with a baseline rate (15%).
  - Per-class Dice is tracked over a rolling window (50 epochs).
  - When a class plateaus (improvement < threshold over window), GAN injection
    rate increases for that class.
  - NETC (label 1) has a lower plateau threshold and higher max injection rate,
    reflecting its status as the hardest/rarest class.
  - When NETC is in plateau, the label generator biases toward NETC-heavy
    synthetic tumors (rejection sampling).

Paper framing: "Performance-adaptive GAN augmentation: injection probability is
modulated per-class based on validation Dice trajectory, concentrating synthetic
data where the model needs it most."
"""

import os
import numpy as np
import torch
from collections import deque

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_MedNeXt_B_kernel5,
)
from nnunet_mednext.utilities.to_torch import maybe_to_torch, to_cuda

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def autocast():
        yield


DEFAULT_WEIGHTS_DIR = os.path.expanduser("~/bmds260/gligan_weights/brats2024")

# BraTS 2024 class indices: 1=NETC, 2=SNFH, 3=ET, 4=RC
# nnU-Net foreground classes are indexed 0-3 in online eval (0=NETC, 1=SNFH, 2=ET, 3=RC)
CLASS_NAMES = {0: "NETC", 1: "SNFH", 2: "ET", 3: "RC"}


class nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN(nnUNetTrainerV2_MedNeXt_B_kernel5):
    """
    MedNeXt-B kernel5 with performance-adaptive GliGAN augmentation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Adaptive schedule parameters ---
        self.gan_base_prob = 0.15          # baseline injection rate from epoch 0
        self.gan_max_prob = 0.50           # max injection rate for normal classes
        self.gan_max_prob_netc = 0.60      # max injection rate for NETC
        self.gan_ramp_step = 0.05          # how much to increase prob per plateau detection

        # Plateau detection
        self.plateau_window = 50           # rolling window size (epochs)
        self.plateau_threshold = 0.005     # min Dice improvement over window to NOT be plateau
        self.plateau_threshold_netc = 0.003  # lower threshold for NETC (triggers earlier)

        # Per-class state
        self.per_class_gan_prob = [self.gan_base_prob] * 4  # [NETC, SNFH, ET, RC]
        self.per_class_dice_history = [deque(maxlen=self.plateau_window) for _ in range(4)]
        self.netc_plateau = False          # flag for NETC-biased label generation

        # Track per-class Dice each epoch (populated in finish_online_evaluation)
        self._current_epoch_class_dice = None

        # GliGAN augmenter (lazy loaded)
        self._gligan = None
        self._gligan_load_attempted = False
        self.gligan_weights_dir = os.environ.get("GLIGAN_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)

    def _get_gligan(self):
        """Lazy-load GliGAN augmenter on first use."""
        if self._gligan is not None:
            return self._gligan
        if self._gligan_load_attempted:
            return None

        self._gligan_load_attempted = True

        if not os.path.isdir(self.gligan_weights_dir):
            self.print_to_log_file(
                f"WARNING: GliGAN weights dir not found: {self.gligan_weights_dir}. "
                f"Running without GAN augmentation."
            )
            return None

        try:
            from nnunet_mednext.training.network_training.MedNeXt.gligan_augment import GliGANAugmenter
            self._gligan = GliGANAugmenter(
                weights_dir=self.gligan_weights_dir,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                in_channels=5,
                out_channels=1,
                feature_size=48,
                label_out_channels=4,
            )
            self._gligan.load()
            self.print_to_log_file("GliGAN augmenter loaded successfully.")
            return self._gligan
        except Exception as e:
            self.print_to_log_file(f"WARNING: Failed to load GliGAN: {e}. Running without GAN augmentation.")
            return None

    def get_gan_probability(self):
        """
        Get current per-sample GAN injection probability.
        Returns the max across all per-class probabilities (we inject a full
        synthetic tumor, but bias the label generator based on which class needs help).
        """
        return max(self.per_class_gan_prob)

    def _detect_plateaus(self):
        """
        Check each class's Dice history for plateau. If plateaued, bump its
        GAN injection probability.
        """
        if self.epoch < self.plateau_window:
            return  # not enough history yet

        for cls_idx in range(4):
            history = self.per_class_dice_history[cls_idx]
            if len(history) < self.plateau_window:
                continue

            # Improvement = current Dice - Dice from (window) epochs ago
            recent = np.mean(list(history)[-10:])  # last 10 epochs
            earlier = np.mean(list(history)[:10])   # first 10 of window
            improvement = recent - earlier

            threshold = self.plateau_threshold_netc if cls_idx == 0 else self.plateau_threshold
            max_prob = self.gan_max_prob_netc if cls_idx == 0 else self.gan_max_prob

            if improvement < threshold:
                old_prob = self.per_class_gan_prob[cls_idx]
                new_prob = min(old_prob + self.gan_ramp_step, max_prob)
                if new_prob > old_prob:
                    self.per_class_gan_prob[cls_idx] = new_prob
                    self.print_to_log_file(
                        f"  ADAPTIVE: {CLASS_NAMES[cls_idx]} plateaued "
                        f"(improvement={improvement:.4f} < {threshold}). "
                        f"GAN prob: {old_prob:.2f} -> {new_prob:.2f}")

        # Update NETC plateau flag for label generation bias
        self.netc_plateau = self.per_class_gan_prob[0] > self.gan_base_prob

    def finish_online_evaluation(self):
        """
        Override to capture per-class Dice before the parent resets accumulators.
        """
        # Compute per-class Dice from accumulated TP/FP/FN
        tp = np.sum(self.online_eval_tp, 0)
        fp = np.sum(self.online_eval_fp, 0)
        fn = np.sum(self.online_eval_fn, 0)

        per_class_dice = [2 * t / (2 * t + f + n + 1e-8) for t, f, n in zip(tp, fp, fn)]
        self._current_epoch_class_dice = per_class_dice

        # Store in per-class history
        for cls_idx, dice_val in enumerate(per_class_dice):
            if not np.isnan(dice_val):
                self.per_class_dice_history[cls_idx].append(dice_val)

        # Call parent (logs mean Dice, resets accumulators)
        super().finish_online_evaluation()

    def on_epoch_end(self):
        """Override to run plateau detection after online eval."""
        ret = super().on_epoch_end()

        # Log per-class Dice
        if self._current_epoch_class_dice is not None:
            dice_str = ", ".join(
                f"{CLASS_NAMES[i]}={d:.4f}" for i, d in enumerate(self._current_epoch_class_dice)
            )
            self.print_to_log_file(f"  Per-class Dice: {dice_str}")
            prob_str = ", ".join(
                f"{CLASS_NAMES[i]}={p:.2f}" for i, p in enumerate(self.per_class_gan_prob)
            )
            self.print_to_log_file(f"  GAN probs: {prob_str} | NETC-bias: {self.netc_plateau}")

        # Detect plateaus and adapt
        self._detect_plateaus()

        self._current_epoch_class_dice = None
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Override run_iteration to inject GAN augmentation between data loading
        and forward pass, with performance-adaptive probability.
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # Apply GAN augmentation (only during training, not validation)
        if do_backprop:
            gan_prob = self.get_gan_probability()
            gligan = self._get_gligan() if gan_prob > 0 else None

            if gligan is not None and gan_prob > 0:
                batch_size = data.shape[0]
                for b in range(batch_size):
                    if np.random.rand() < gan_prob:
                        try:
                            data_np = data[b].numpy() if isinstance(data, torch.Tensor) else data[b]
                            seg_np = target[b].numpy() if isinstance(target, torch.Tensor) else target[b]

                            data_aug, seg_aug = self._augment_with_bias(
                                gligan, data_np.copy(), seg_np.copy()
                            )

                            if isinstance(data, torch.Tensor):
                                data[b] = torch.from_numpy(data_aug)
                                target[b] = torch.from_numpy(seg_aug)
                            else:
                                data[b] = data_aug
                                target[b] = seg_aug
                        except Exception:
                            pass

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()

    def _augment_with_bias(self, gligan, data_np, seg_np):
        """
        Apply GliGAN augmentation with NETC bias when NETC is plateaued.

        When NETC-biased: rejection-sample the label generator up to 5 times,
        keeping only synthetic tumors that contain NETC (label 1). This increases
        the proportion of NETC voxels in training without changing the GAN itself.
        """
        if not self.netc_plateau:
            return gligan.augment_volume(data_np, seg_np)

        # NETC-biased: try up to 5 times to get a tumor with NETC
        from nnunet_mednext.training.network_training.MedNeXt.gligan_augment import generate_random_label
        best_data, best_seg = None, None
        best_netc_count = 0

        for attempt in range(5):
            data_try, seg_try = gligan.augment_volume(data_np.copy(), seg_np.copy())
            # Count new NETC voxels (label 1 in augmented but not in original)
            new_netc = np.sum((seg_try[0] == 1) & (seg_np[0] != 1))
            if new_netc > best_netc_count:
                best_netc_count = new_netc
                best_data = data_try
                best_seg = seg_try
            if new_netc > 50:  # good enough, stop early
                break

        if best_data is not None:
            return best_data, best_seg
        return data_np, seg_np
