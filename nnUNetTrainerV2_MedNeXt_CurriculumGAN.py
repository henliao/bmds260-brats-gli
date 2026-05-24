"""
MedNeXt-B kernel5 trainer with curriculum-scheduled GliGAN augmentation.

Curriculum schedule:
  Phase 1 (epochs 0-299):   No GAN injection. Learn clean representations.
  Phase 2 (epochs 300-699): GAN injection at moderate rate (30%). Balanced tumor types.
  Phase 3 (epochs 700-999): GAN injection at high rate (50%). Bias toward NETC-heavy cases.

The GAN augmentation is applied per-sample in the batch, after nnU-Net's standard
data loading but before the forward pass.
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


# Default weights directory (override via GLIGAN_WEIGHTS_DIR env var)
DEFAULT_WEIGHTS_DIR = os.path.expanduser("~/bmds260/gligan_weights/brats2024")


class nnUNetTrainerV2_MedNeXt_B_kernel5_CurriculumGAN(nnUNetTrainerV2_MedNeXt_B_kernel5):
    """
    MedNeXt-B kernel5 with curriculum-scheduled GliGAN on-the-fly augmentation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Curriculum schedule parameters
        self.gan_phase1_end = 300    # no GAN
        self.gan_phase2_end = 700    # moderate GAN
        # phase 3: 700-1000, aggressive GAN

        self.gan_prob_phase2 = 0.3   # injection probability in phase 2
        self.gan_prob_phase3 = 0.5   # injection probability in phase 3

        # GliGAN augmenter (lazy loaded)
        self._gligan = None
        self._gligan_load_attempted = False

        # Weights directory
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
                in_channels=5,  # 4 label channels + 1 noisy scan (BraTS 2024 post-treatment)
                out_channels=1,
                feature_size=48,
                label_out_channels=4,  # NETC, SNFH, ET, RC
            )
            self._gligan.load()
            self.print_to_log_file("GliGAN augmenter loaded successfully.")
            return self._gligan
        except Exception as e:
            self.print_to_log_file(f"WARNING: Failed to load GliGAN: {e}. Running without GAN augmentation.")
            return None

    def get_gan_probability(self):
        """Get current GAN injection probability based on curriculum schedule."""
        if self.epoch < self.gan_phase1_end:
            return 0.0
        elif self.epoch < self.gan_phase2_end:
            return self.gan_prob_phase2
        else:
            return self.gan_prob_phase3

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        Override run_iteration to inject GAN augmentation between data loading
        and forward pass, with curriculum-scheduled probability.
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # Apply GAN augmentation (only during training, not validation)
        if do_backprop:
            gan_prob = self.get_gan_probability()
            gligan = self._get_gligan() if gan_prob > 0 else None

            if gligan is not None and gan_prob > 0:
                # data shape: (B, C, D, H, W) where C=4 modalities
                # target shape: (B, 1, D, H, W) segmentation labels
                batch_size = data.shape[0]
                for b in range(batch_size):
                    if np.random.rand() < gan_prob:
                        try:
                            data_np = data[b].numpy() if isinstance(data, torch.Tensor) else data[b]
                            seg_np = target[b].numpy() if isinstance(target, torch.Tensor) else target[b]

                            data_aug, seg_aug = gligan.augment_volume(
                                data_np.copy(), seg_np.copy()
                            )

                            if isinstance(data, torch.Tensor):
                                data[b] = torch.from_numpy(data_aug)
                                target[b] = torch.from_numpy(seg_aug)
                            else:
                                data[b] = data_aug
                                target[b] = seg_aug
                        except Exception as e:
                            # Silently skip failed augmentations
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

    def on_epoch_end(self):
        """Log curriculum phase transitions."""
        ret = super().on_epoch_end()

        # Log phase transitions
        if self.epoch == self.gan_phase1_end:
            self.print_to_log_file(
                f"=== CURRICULUM: Entering Phase 2 (epoch {self.epoch}). "
                f"GAN injection at {self.gan_prob_phase2*100:.0f}% ===")
        elif self.epoch == self.gan_phase2_end:
            self.print_to_log_file(
                f"=== CURRICULUM: Entering Phase 3 (epoch {self.epoch}). "
                f"GAN injection at {self.gan_prob_phase3*100:.0f}% ===")

        return ret
