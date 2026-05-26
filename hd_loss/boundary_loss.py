"""
Boundary loss (Kervadec et al., 2019) for medical image segmentation.

Computes a differentiable approximation to surface distance using signed
distance maps from ground truth. Per-class, compatible with deep supervision.

Reference: https://proceedings.mlr.press/v102/kervadec19a.html
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_signed_distance_map(seg_onehot: np.ndarray) -> np.ndarray:
    """
    Compute signed distance maps for each class in a one-hot segmentation.

    Args:
        seg_onehot: (C, D, H, W) binary one-hot encoded segmentation

    Returns:
        (C, D, H, W) signed distance maps. Positive outside, negative inside.
    """
    num_classes = seg_onehot.shape[0]
    sdm = np.zeros_like(seg_onehot, dtype=np.float64)

    for c in range(num_classes):
        mask = seg_onehot[c].astype(bool)
        if mask.any() and not mask.all():
            dist_out = distance_transform_edt(~mask)
            dist_in = distance_transform_edt(mask)
            sdm[c] = dist_out - dist_in
        elif mask.all():
            sdm[c] = -distance_transform_edt(mask)
        else:
            # No foreground for this class: large positive distance everywhere
            sdm[c] = distance_transform_edt(~mask)

    return np.clip(sdm, -1000.0, 1000.0)


def seg_to_onehot(seg: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer segmentation to one-hot encoding.

    BraTS-GLI nnU-Net classes: background=0, NETC=1, SNFH=2, ET=3, RC=4
    We skip background (index 0), so foreground classes are indices 1..num_classes.

    Args:
        seg: (1, D, H, W) or (D, H, W) integer segmentation
        num_classes: number of foreground classes (4 for BraTS)

    Returns:
        (num_classes, D, H, W) binary one-hot (foreground only)
    """
    if seg.ndim == 4:
        seg = seg[0]  # remove channel dim
    onehot = np.zeros((num_classes,) + seg.shape, dtype=np.float32)
    for c in range(num_classes):
        onehot[c] = (seg == (c + 1)).astype(np.float32)
    return onehot


class BoundaryLoss(nn.Module):
    """
    Boundary loss using precomputed signed distance maps from ground truth.

    For each sample in the batch:
      L_boundary = sum over classes of: mean(softmax_pred_c * dist_map_c)

    Lower loss = predictions align with GT boundaries.
    """

    def __init__(self, num_classes: int = 4, do_bg: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.do_bg = do_bg

    def forward(self, net_output: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            net_output: (B, C, D, H, W) raw logits from the network
            dist_maps: (B, num_classes, D, H, W) signed distance maps

        Returns:
            Scalar boundary loss
        """
        # Softmax to get probabilities
        probs = torch.softmax(net_output, dim=1)

        # Skip background channel (class 0 in network output)
        # Foreground classes are channels 1..num_classes
        start_idx = 0 if self.do_bg else 1
        fg_probs = probs[:, start_idx:start_idx + self.num_classes]

        # Boundary loss: inner product of predicted probability and distance map
        # Minimizing this pushes predictions toward GT boundaries
        loss = (fg_probs * dist_maps).mean()

        return loss


class DC_CE_and_Boundary_loss(nn.Module):
    """
    Combined Dice+CE+Boundary loss with epoch-dependent scheduling.

    For epochs < switch_epoch: loss = DC_and_CE_loss (standard nnU-Net)
    For epochs >= switch_epoch: loss = alpha * DC_and_CE_loss + (1-alpha) * BoundaryLoss

    The boundary loss weight ramps linearly from 0 to max_boundary_weight
    over the range [switch_epoch, max_epochs].
    """

    def __init__(self, dc_ce_loss: nn.Module, num_classes: int = 4,
                 switch_epoch: int = 800, max_epochs: int = 1000,
                 max_boundary_weight: float = 0.5):
        super().__init__()
        self.dc_ce = dc_ce_loss
        self.boundary = BoundaryLoss(num_classes=num_classes)
        self.switch_epoch = switch_epoch
        self.max_epochs = max_epochs
        self.max_boundary_weight = max_boundary_weight
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def get_boundary_weight(self) -> float:
        if self._current_epoch < self.switch_epoch:
            return 0.0
        progress = (self._current_epoch - self.switch_epoch) / max(
            self.max_epochs - self.switch_epoch, 1)
        return min(self.max_boundary_weight * progress, self.max_boundary_weight)

    def forward(self, net_output: torch.Tensor,
                target: torch.Tensor,
                dist_maps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            net_output: raw logits (B, C, D, H, W)
            target: ground truth segmentation for DC+CE
            dist_maps: signed distance maps for boundary loss (optional)
        """
        dc_ce_loss = self.dc_ce(net_output, target)
        bw = self.get_boundary_weight()

        if bw > 0 and dist_maps is not None:
            bd_loss = self.boundary(net_output, dist_maps)
            return (1 - bw) * dc_ce_loss + bw * bd_loss

        return dc_ce_loss


class MultipleOutputLoss2_WithBoundary(nn.Module):
    """
    Deep supervision wrapper that passes distance maps to the combined loss.

    Same interface as nnU-Net's MultipleOutputLoss2 but additionally accepts
    distance maps for the boundary loss component.
    """

    def __init__(self, loss: DC_CE_and_Boundary_loss, weight_factors=None):
        super().__init__()
        self.loss = loss
        self.weight_factors = weight_factors
        self._dist_maps = None  # set per-iteration by the trainer

    def set_dist_maps(self, dist_maps_list):
        """Set distance maps for all deep supervision levels."""
        self._dist_maps = dist_maps_list

    def forward(self, x, y):
        assert isinstance(x, (tuple, list))
        assert isinstance(y, (tuple, list))
        weights = self.weight_factors if self.weight_factors is not None else [1] * len(x)

        dm = self._dist_maps
        dm0 = dm[0] if dm is not None and len(dm) > 0 else None
        l = weights[0] * self.loss(x[0], y[0], dm0)

        for i in range(1, len(x)):
            if weights[i] != 0:
                dmi = dm[i] if dm is not None and i < len(dm) else None
                l += weights[i] * self.loss(x[i], y[i], dmi)
        return l
