"""
MedNeXt-B kernel5 trainer with staged Hausdorff Distance (boundary) loss.

Training schedule:
  - Epochs 0-799: standard Dice+CE loss (identical to baseline MedNeXt)
  - Epochs 800-999: Dice+CE + boundary loss, with boundary weight ramping
    linearly from 0 to max_boundary_weight (0.5)

The boundary loss uses signed distance maps computed from ground truth
segmentations (Kervadec et al., 2019). Distance maps are computed on-the-fly
per batch using scipy's distance_transform_edt on CPU, then transferred to GPU.

This staged approach leverages learned tumor region representations from the
first 80% of training, then refines boundary precision in the final 20%.

Separate results directory (RESULTS_FOLDER_HDLOSS) prevents overwriting
baseline or CurriculumGAN checkpoints.
"""

import numpy as np
import torch

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_MedNeXt_B_kernel5,
)
from nnunet_mednext.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet_mednext.utilities.to_torch import maybe_to_torch, to_cuda

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def autocast():
        yield

from nnunet_mednext.training.network_training.MedNeXt.boundary_loss import (
    BoundaryLoss,
    DC_CE_and_Boundary_loss,
    MultipleOutputLoss2_WithBoundary,
    compute_signed_distance_map,
    seg_to_onehot,
)

# BraTS-GLI foreground classes
NUM_FG_CLASSES = 4
CLASS_NAMES = {0: "NETC", 1: "SNFH", 2: "ET", 3: "RC"}


class nnUNetTrainerV2_MedNeXt_B_kernel5_HDLoss(nnUNetTrainerV2_MedNeXt_B_kernel5):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # HD loss schedule: final 20% of training
        self.hd_switch_epoch = 800
        self.hd_max_epochs = 1000
        self.hd_max_boundary_weight = 0.5
        self._hd_active = False

    def initialize(self, training=True, force_load_plans=False):
        """Override to replace loss with combined DC+CE+Boundary loss."""
        super().initialize(training, force_load_plans)

        if training:
            # self.loss at this point is MultipleOutputLoss2(DC_and_CE_loss, ds_weights)
            # We need to unwrap, wrap with boundary, then re-wrap with deep supervision

            # Get the inner DC+CE loss and deep supervision weights
            inner_loss = self.loss.loss  # DC_and_CE_loss instance
            ds_weights = self.loss.weight_factors

            # Wrap with boundary loss
            combined = DC_CE_and_Boundary_loss(
                dc_ce_loss=inner_loss,
                num_classes=NUM_FG_CLASSES,
                switch_epoch=self.hd_switch_epoch,
                max_epochs=self.hd_max_epochs,
                max_boundary_weight=self.hd_max_boundary_weight,
            )

            # Re-wrap with deep supervision
            self.loss = MultipleOutputLoss2_WithBoundary(combined, ds_weights)

            self.print_to_log_file(
                f"HD Loss trainer initialized. Boundary loss activates at epoch {self.hd_switch_epoch}, "
                f"ramping to weight {self.hd_max_boundary_weight} by epoch {self.hd_max_epochs}.")

    def _compute_dist_maps_for_target(self, target_np: np.ndarray) -> np.ndarray:
        """
        Compute signed distance maps for a batch of segmentation targets.

        Args:
            target_np: (B, 1, D, H, W) integer segmentation

        Returns:
            (B, NUM_FG_CLASSES, D, H, W) signed distance maps
        """
        batch_size = target_np.shape[0]
        spatial_shape = target_np.shape[2:]  # (D, H, W)
        dist_maps = np.zeros(
            (batch_size, NUM_FG_CLASSES) + spatial_shape, dtype=np.float32)

        for b in range(batch_size):
            seg = target_np[b]  # (1, D, H, W)
            onehot = seg_to_onehot(seg, NUM_FG_CLASSES)  # (4, D, H, W)
            sdm = compute_signed_distance_map(onehot)  # (4, D, H, W)
            dist_maps[b] = sdm.astype(np.float32)

        return dist_maps

    def on_epoch_end(self):
        """Log boundary loss status at each epoch."""
        ret = super().on_epoch_end()

        # Update epoch in loss
        if hasattr(self.loss, 'loss') and hasattr(self.loss.loss, 'set_epoch'):
            self.loss.loss.set_epoch(self.epoch + 1)
            bw = self.loss.loss.get_boundary_weight()
            was_active = self._hd_active
            self._hd_active = bw > 0

            if self._hd_active:
                self.print_to_log_file(
                    f"  HD boundary loss weight: {bw:.3f}")
            elif self.epoch + 1 == self.hd_switch_epoch:
                self.print_to_log_file(
                    f"  HD boundary loss ACTIVATING next epoch (epoch {self.hd_switch_epoch})")

        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """Override to compute and pass distance maps when boundary loss is active."""
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # Compute distance maps only for full-resolution target (index 0).
        # Lower deep supervision levels use one-hot soft pooling, not integer
        # segmentations, so boundary loss is only applied at full resolution.
        dist_maps_list = None
        need_dist_maps = (hasattr(self.loss, 'loss') and
                          hasattr(self.loss.loss, 'get_boundary_weight') and
                          self.loss.loss.get_boundary_weight() > 0)

        if need_dist_maps and do_backprop:
            if isinstance(target, (list, tuple)):
                # target[0] = full-res integer segmentation (B, D, H, W) torch tensor
                t0 = target[0]
                t0_np = t0.numpy() if isinstance(t0, torch.Tensor) else t0
                # Need shape (B, 1, D, H, W) for _compute_dist_maps_for_target
                if t0_np.ndim == 4:
                    t0_np = t0_np[:, np.newaxis]
                dm = self._compute_dist_maps_for_target(t0_np)
                dm_tensor = torch.from_numpy(dm)
                if torch.cuda.is_available():
                    dm_tensor = to_cuda(dm_tensor)
                # Only full-res gets dist maps, rest get None
                dist_maps_list = [dm_tensor] + [None] * (len(target) - 1)
            else:
                t_np = target.numpy() if isinstance(target, torch.Tensor) else target
                if t_np.ndim == 4:
                    t_np = t_np[:, np.newaxis]
                dm = self._compute_dist_maps_for_target(t_np)
                dm_tensor = torch.from_numpy(dm)
                if torch.cuda.is_available():
                    dm_tensor = to_cuda(dm_tensor)
                dist_maps_list = [dm_tensor]

        # Set distance maps on the loss wrapper
        if hasattr(self.loss, 'set_dist_maps'):
            self.loss.set_dist_maps(dist_maps_list)

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
