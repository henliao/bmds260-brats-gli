"""
MedNeXt-B kernel5 trainer with performance-adaptive GliGAN augmentation.

Schedule adapts to per-class validation Dice:
  - GAN injection starts at epoch 0 with a baseline rate (15%).
  - Per-class Dice is smoothed via EMA (alpha=0.1) to filter patch-level noise.
  - When a class's smoothed Dice plateaus over a 50-epoch lookback, its GAN
    injection rate increases and its label generation threshold decreases
    (producing larger synthetic regions for that class).
  - When a class improves again, rates decay back toward baseline.
  - Adaptive state (EMA history, per-class rates) is saved/restored in checkpoints
    so the schedule survives wall-time restarts.
"""

import os
import numpy as np
import torch

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

# nnU-Net foreground classes: 0=NETC, 1=SNFH, 2=ET, 3=RC
CLASS_NAMES = {0: "NETC", 1: "SNFH", 2: "ET", 3: "RC"}

# Default label threshold on tanh scale [-1, 1] (matches pretrained generator)
DEFAULT_LABEL_THRESH = 0.5

# Threshold range for class weighting: [min_thresh, DEFAULT_LABEL_THRESH]
# Lower threshold = larger label region. min_thresh=-0.2 roughly triples region size.
MIN_LABEL_THRESH = -0.2


class nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN(nnUNetTrainerV2_MedNeXt_B_kernel5):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Adaptive schedule parameters ---
        self.gan_base_prob = 0.15
        self.gan_max_prob = 0.50
        self.gan_max_prob_netc = 0.60
        self.gan_ramp_step = 0.05
        self.gan_decay_step = 0.03

        # EMA smoothing
        self.ema_alpha = 0.1
        self.ema_dice = [None] * 4

        # Plateau detection
        self.lookback = 50
        self.plateau_threshold = 0.01
        self.plateau_threshold_netc = 0.005
        self.improvement_threshold = 0.02
        self.improvement_threshold_netc = 0.01

        # Per-class state
        self.per_class_gan_prob = [self.gan_base_prob] * 4
        self.ema_history = [[] for _ in range(4)]

        # Per-class Dice from this epoch (set in finish_online_evaluation)
        self._current_epoch_class_dice = None

        # GliGAN augmenter (lazy loaded)
        self._gligan = None
        self._gligan_load_attempted = False
        self.gligan_weights_dir = os.environ.get("GLIGAN_WEIGHTS_DIR", DEFAULT_WEIGHTS_DIR)

    # --- Checkpoint save/restore for adaptive state ---

    def save_checkpoint(self, fname, save_optimizer=True):
        """Save checkpoint with adaptive state appended."""
        super().save_checkpoint(fname, save_optimizer)
        # Append adaptive state to the saved checkpoint
        ckpt = torch.load(fname, map_location='cpu')
        ckpt['adaptive_state'] = {
            'per_class_gan_prob': self.per_class_gan_prob,
            'ema_dice': self.ema_dice,
            'ema_history': [list(h) for h in self.ema_history],
        }
        torch.save(ckpt, fname)

    def load_checkpoint_ram(self, saved, train=True):
        """Restore checkpoint and adaptive state."""
        super().load_checkpoint_ram(saved, train)
        if 'adaptive_state' in saved:
            state = saved['adaptive_state']
            self.per_class_gan_prob = state.get('per_class_gan_prob', [self.gan_base_prob] * 4)
            self.ema_dice = state.get('ema_dice', [None] * 4)
            self.ema_history = [list(h) for h in state.get('ema_history', [[] for _ in range(4)])]
            self.print_to_log_file(
                f"Restored adaptive state: probs={self.per_class_gan_prob}, "
                f"EMA history lengths={[len(h) for h in self.ema_history]}")
        else:
            self.print_to_log_file("No adaptive state in checkpoint, starting fresh.")

    # --- GliGAN management ---

    def _get_gligan(self):
        if self._gligan is not None:
            return self._gligan
        if self._gligan_load_attempted:
            return None
        self._gligan_load_attempted = True

        if not os.path.isdir(self.gligan_weights_dir):
            self.print_to_log_file(
                f"WARNING: GliGAN weights dir not found: {self.gligan_weights_dir}")
            return None

        try:
            from nnunet_mednext.training.network_training.MedNeXt.gligan_augment import GliGANAugmenter
            self._gligan = GliGANAugmenter(
                weights_dir=self.gligan_weights_dir,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                in_channels=5, out_channels=1, feature_size=48, label_out_channels=4)
            self._gligan.load()
            self.print_to_log_file("GliGAN augmenter loaded successfully.")
            return self._gligan
        except Exception as e:
            self.print_to_log_file(f"WARNING: Failed to load GliGAN: {e}")
            return None

    # --- Adaptive schedule ---

    def get_gan_probability(self):
        return max(self.per_class_gan_prob)

    def _prob_to_threshold(self, cls_idx):
        """
        Map per-class GAN probability to label generation threshold.

        At base_prob: threshold = DEFAULT_LABEL_THRESH (0.5, normal)
        At max_prob:  threshold = MIN_LABEL_THRESH (-0.2, ~3x larger region)
        Linear interpolation between.
        """
        prob = self.per_class_gan_prob[cls_idx]
        max_prob = self.gan_max_prob_netc if cls_idx == 0 else self.gan_max_prob
        # Fraction of the way from base to max
        frac = (prob - self.gan_base_prob) / max(max_prob - self.gan_base_prob, 1e-8)
        frac = max(0.0, min(1.0, frac))
        return DEFAULT_LABEL_THRESH - frac * (DEFAULT_LABEL_THRESH - MIN_LABEL_THRESH)

    def _get_class_thresholds(self):
        """Get current per-class label generation thresholds."""
        return {i: self._prob_to_threshold(i) for i in range(4)}

    def _update_ema(self, cls_idx, raw_dice):
        if self.ema_dice[cls_idx] is None:
            self.ema_dice[cls_idx] = raw_dice
        else:
            self.ema_dice[cls_idx] = (
                self.ema_alpha * raw_dice +
                (1 - self.ema_alpha) * self.ema_dice[cls_idx])
        self.ema_history[cls_idx].append(self.ema_dice[cls_idx])

    def _adapt_rates(self):
        """Bidirectional rate adaptation based on smoothed Dice trajectory."""
        for cls_idx in range(4):
            history = self.ema_history[cls_idx]
            if len(history) < self.lookback:
                continue

            improvement = history[-1] - history[-self.lookback]

            plat_thresh = self.plateau_threshold_netc if cls_idx == 0 else self.plateau_threshold
            impr_thresh = self.improvement_threshold_netc if cls_idx == 0 else self.improvement_threshold
            max_prob = self.gan_max_prob_netc if cls_idx == 0 else self.gan_max_prob

            old_prob = self.per_class_gan_prob[cls_idx]

            if improvement < plat_thresh:
                new_prob = min(old_prob + self.gan_ramp_step, max_prob)
                if new_prob > old_prob:
                    self.print_to_log_file(
                        f"  ADAPTIVE: {CLASS_NAMES[cls_idx]} plateau "
                        f"(EMA delta={improvement:.4f} < {plat_thresh}). "
                        f"GAN: {old_prob:.2f} -> {new_prob:.2f}, "
                        f"thresh: {self._prob_to_threshold(cls_idx):.2f} -> {DEFAULT_LABEL_THRESH - (new_prob - self.gan_base_prob) / max(max_prob - self.gan_base_prob, 1e-8) * (DEFAULT_LABEL_THRESH - MIN_LABEL_THRESH):.2f}")
                self.per_class_gan_prob[cls_idx] = new_prob
            elif improvement > impr_thresh:
                new_prob = max(old_prob - self.gan_decay_step, self.gan_base_prob)
                if new_prob < old_prob:
                    self.print_to_log_file(
                        f"  ADAPTIVE: {CLASS_NAMES[cls_idx]} improving "
                        f"(EMA delta={improvement:.4f} > {impr_thresh}). "
                        f"GAN: {old_prob:.2f} -> {new_prob:.2f}")
                self.per_class_gan_prob[cls_idx] = new_prob

    # --- Online evaluation override ---

    def finish_online_evaluation(self):
        tp = np.sum(self.online_eval_tp, 0)
        fp = np.sum(self.online_eval_fp, 0)
        fn = np.sum(self.online_eval_fn, 0)

        per_class_dice = [2 * t / (2 * t + f + n + 1e-8) for t, f, n in zip(tp, fp, fn)]
        self._current_epoch_class_dice = per_class_dice

        for cls_idx, dice_val in enumerate(per_class_dice):
            if not np.isnan(dice_val):
                self._update_ema(cls_idx, float(dice_val))

        super().finish_online_evaluation()

    def on_epoch_end(self):
        ret = super().on_epoch_end()

        if self._current_epoch_class_dice is not None:
            raw_str = ", ".join(f"{CLASS_NAMES[i]}={d:.4f}"
                                for i, d in enumerate(self._current_epoch_class_dice))
            ema_str = ", ".join(
                f"{CLASS_NAMES[i]}={e:.4f}" if e is not None else f"{CLASS_NAMES[i]}=N/A"
                for i, e in enumerate(self.ema_dice))
            prob_str = ", ".join(f"{CLASS_NAMES[i]}={p:.2f}"
                                for i, p in enumerate(self.per_class_gan_prob))
            thresh_str = ", ".join(f"{CLASS_NAMES[i]}={self._prob_to_threshold(i):.2f}"
                                  for i in range(4))
            self.print_to_log_file(f"  Per-class Dice (raw): {raw_str}")
            self.print_to_log_file(f"  Per-class Dice (EMA): {ema_str}")
            self.print_to_log_file(f"  GAN probs: {prob_str}")
            self.print_to_log_file(f"  Label thresholds: {thresh_str}")

        self._adapt_rates()
        self._current_epoch_class_dice = None
        return ret

    # --- Training iteration with GAN augmentation ---

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        if do_backprop:
            gan_prob = self.get_gan_probability()
            gligan = self._get_gligan() if gan_prob > 0 else None

            if gligan is not None and gan_prob > 0:
                class_thresholds = self._get_class_thresholds()
                batch_size = data.shape[0]
                for b in range(batch_size):
                    if np.random.rand() < gan_prob:
                        try:
                            data_np = data[b].numpy() if isinstance(data, torch.Tensor) else data[b]
                            seg_np = target[b].numpy() if isinstance(target, torch.Tensor) else target[b]

                            data_aug, seg_aug = gligan.augment_volume(
                                data_np.copy(), seg_np.copy(),
                                class_thresholds=class_thresholds)

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
