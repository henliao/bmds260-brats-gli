"""
On-the-fly GliGAN augmentation for nnU-Net v1 training.

Loads pretrained GliGAN generators (4 modality-specific Swin UNETRs + 1 label GAN)
and injects synthetic tumors into training batches. All operations vectorized.

All heavy imports (monai, scipy) are deferred to load() time so that nnU-Net's
trainer discovery doesn't fail when these packages are missing.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelGenerator(nn.Module):
    """Small ConvTranspose3d network that generates tumor label masks from noise."""
    def __init__(self, noise=100, channel=64, out_channels=3):
        super().__init__()
        _c = channel
        self.leaky_relu = nn.LeakyReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*8, 4, 1, 0, bias=False)
        self.bn1 = nn.InstanceNorm3d(_c*8)
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, 3, 1, 1, bias=False)
        self.bn2 = nn.InstanceNorm3d(_c*4)
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, 3, 1, 1, bias=False)
        self.bn3 = nn.InstanceNorm3d(_c*2)
        self.tp_conv4 = nn.Conv3d(_c*2, _c, 3, 1, 1, bias=False)
        self.bn4 = nn.InstanceNorm3d(_c)
        self.tp_conv5 = nn.Conv3d(_c, out_channels, 3, 1, 1, bias=False)

    def forward(self, noise):
        noise = noise.view(-1, self.noise, 1, 1, 1)
        h = self.leaky_relu(self.bn1(self.tp_conv1(noise)))
        h = F.interpolate(h, scale_factor=2)
        h = self.leaky_relu(self.bn2(self.tp_conv2(h)))
        h = F.interpolate(h, scale_factor=2)
        h = self.leaky_relu(self.bn3(self.tp_conv3(h)))
        h = F.interpolate(h, scale_factor=2)
        h = self.leaky_relu(self.bn4(self.tp_conv4(h)))
        h = F.interpolate(h, scale_factor=2)
        h = self.tp_conv5(h)
        return torch.tanh(h)


def rescale_array(arr, minv=0.0, maxv=1.0):
    """Rescale array to [minv, maxv]."""
    mina = arr.min()
    maxa = arr.max()
    if mina == maxa:
        return np.full_like(arr, minv)
    return (arr - mina) / (maxa - mina) * (maxv - minv) + minv


def add_gaussian_noise_vectorized(scan, label_mask):
    """Vectorized: replace tumor region with Gaussian noise, rescale to [-1,1]."""
    scan_noisy = scan.copy()
    tumor_mask = label_mask != 0
    noise = np.random.randn(*scan.shape).astype(np.float32)
    scan_noisy[tumor_mask] = noise[tumor_mask]
    scan_noisy = rescale_array(scan_noisy, -1, 1)
    return scan_noisy, tumor_mask.astype(np.float32)


def correct_label_vectorized(label, brain_mask, original_label):
    """Vectorized: zero out label voxels outside brain or overlapping existing tumor."""
    invalid = (brain_mask == 0) | (original_label != 0)
    label[invalid] = 0
    return label


def correct_background_vectorized(healthy_crop_norm, imgs_recon):
    """Vectorized: clamp background to -1, values to [-1, 1]."""
    result = imgs_recon.copy()
    result[healthy_crop_norm == -1] = -1
    return np.clip(result, -1, 1)


def linear_interpolation_vectorized(recon, healthy_crop, original_label_crop, noise_mask):
    """Vectorized linear intensity correction using boundary voxels."""
    from scipy.ndimage import binary_dilation
    dilated_noise = binary_dilation(noise_mask > 0, iterations=5)
    boundary = (healthy_crop != 0) & (original_label_crop == 0) & (~dilated_noise)

    if boundary.sum() < 2:
        return recon

    x_vals = recon[boundary].ravel()
    y_vals = healthy_crop[boundary].ravel()

    # Sample up to 200 points for fitting
    if len(x_vals) > 200:
        idx = np.random.choice(len(x_vals), 200, replace=False)
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]

    x_all = np.concatenate([[-1.0], x_vals])
    y_all = np.concatenate([[0.0], y_vals])

    coeffs = np.polyfit(x_all, y_all, 1)
    result = coeffs[0] * recon + coeffs[1]
    result[healthy_crop == 0] = 0
    result[result < 0] = 0
    return result


def generate_random_label(label_gen, device, out_channels=4):
    """Generate a random tumor label mask using the label GAN."""
    with torch.no_grad():
        z = torch.randn(1, 100, device=device)
        x = label_gen(z)

    x_cpu = x.squeeze(0).cpu().numpy()

    channels = []
    for i in range(min(x_cpu.shape[0], out_channels)):
        ch_tensor = torch.from_numpy(x_cpu[i:i+1]).unsqueeze(0).float()  # [1,1,D,H,W]
        ch_resized = torch.nn.functional.interpolate(ch_tensor, size=(96, 96, 96), mode='trilinear', align_corners=False)
        ch = ch_resized.squeeze(0).squeeze(0).numpy()
        ch = (ch > 0.5).astype(np.float32)
        channels.append(ch)

    combined = np.zeros((96, 96, 96), dtype=np.float32)
    if len(channels) >= 3:
        combined[channels[1] == 1] = 2  # SNFH
        combined[channels[0] == 1] = 1  # NETC
        combined[channels[2] == 1] = 3  # ET
    if len(channels) >= 4:
        combined[channels[3] == 1] = 4  # RC

    return combined


def convert_label_to_multichannel_brats2024(label):
    """Convert single-channel label to multi-channel for GliGAN input."""
    return np.stack([
        ((label == 1) | (label == 3)).astype(np.float32),
        ((label == 1) | (label == 2) | (label == 3)).astype(np.float32),
        (label == 3).astype(np.float32),
        (label == 4).astype(np.float32),
    ], axis=0)


class GliGANAugmenter:
    """
    On-the-fly GliGAN tumor augmentation.

    Loads 4 modality generators (Swin UNETR) + 1 label generator.
    Injects synthetic tumors into training volumes.
    """

    def __init__(self, weights_dir, device='cuda',
                 in_channels=5, out_channels=1, feature_size=48,
                 label_out_channels=4):
        self.device = device
        self.weights_dir = weights_dir
        self.generators = {}
        self.label_gen = None
        self.label_out_channels = label_out_channels
        self._loaded = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size

    def load(self):
        """Load all generators. Call once after init."""
        if self._loaded:
            return

        # Lazy import
        from monai.networks.nets.swin_unetr import SwinUNETR

        modalities = ['t1ce', 't1', 't2', 'flair']
        for mod in modalities:
            mod_dir = os.path.join(self.weights_dir, mod, 'weights')
            if not os.path.isdir(mod_dir):
                raise FileNotFoundError(f"GliGAN weights not found: {mod_dir}")

            pt_files = sorted([f for f in os.listdir(mod_dir)
                               if f.startswith('generator_') and f.endswith('.pt')])
            if not pt_files:
                raise FileNotFoundError(f"No generator weights in {mod_dir}")
            weight_path = os.path.join(mod_dir, pt_files[-1])

            gen = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                feature_size=self.feature_size,
                use_checkpoint=False,
            )
            state = torch.load(weight_path, map_location='cpu')['state_dict']
            gen.load_state_dict(state)
            gen.eval()
            gen.to(self.device)
            self.generators[mod] = gen
            print(f"Loaded GliGAN generator: {mod} from {weight_path}")

        # Load label generator
        label_dir = os.path.join(self.weights_dir, 'label', 'weights')
        if os.path.isdir(label_dir):
            label_files = sorted([f for f in os.listdir(label_dir)
                                  if f.startswith('G_iter') and
                                  (f.endswith('.pth') or f.endswith('.pt'))])
            if label_files:
                label_path = os.path.join(label_dir, label_files[-1])
                self.label_gen = LabelGenerator(
                    noise=100, out_channels=self.label_out_channels)
                state = torch.load(label_path, map_location='cpu')
                if 'state_dict' in state:
                    state = state['state_dict']
                self.label_gen.load_state_dict(state)
                self.label_gen.eval()
                self.label_gen.to(self.device)
                print(f"Loaded GliGAN label generator from {label_path}")

        self._loaded = True

    def augment_volume(self, data_np, seg_np):
        """
        Apply GliGAN augmentation to a single training volume.

        Args:
            data_np: (4, D, H, W) numpy array, 4 MRI modalities
            seg_np: (1, D, H, W) numpy array, segmentation labels

        Returns:
            data_np, seg_np: augmented (same shape)
        """
        if not self._loaded:
            self.load()

        C, D, H, W = data_np.shape
        seg = seg_np[0]

        # Brain mask from first modality
        brain_mask = np.abs(data_np[0]) > 0.01

        # Find random 96^3 patch location in brain, away from existing tumor
        for _ in range(50):
            brain_coords = np.argwhere(brain_mask)
            if len(brain_coords) < 100:
                return data_np, seg_np
            idx = np.random.randint(len(brain_coords))
            cz, cy, cx = brain_coords[idx]

            z0, z1 = max(0, cz - 48), max(0, cz - 48) + 96
            y0, y1 = max(0, cy - 48), max(0, cy - 48) + 96
            x0, x1 = max(0, cx - 48), max(0, cx - 48) + 96

            # Clamp
            if z1 > D: z0, z1 = D - 96, D
            if y1 > H: y0, y1 = H - 96, H
            if x1 > W: x0, x1 = W - 96, W
            if z0 < 0 or y0 < 0 or x0 < 0:
                continue

            if seg[z0:z1, y0:y1, x0:x1].sum() == 0:
                break
        else:
            return data_np, seg_np

        # Generate random tumor label
        if self.label_gen is None:
            return data_np, seg_np
        new_label = generate_random_label(
            self.label_gen, self.device, self.label_out_channels)

        # Correct label
        brain_crop = brain_mask[z0:z1, y0:y1, x0:x1].astype(np.float32)
        orig_seg_crop = seg[z0:z1, y0:y1, x0:x1]
        new_label = correct_label_vectorized(new_label, brain_crop, orig_seg_crop)

        if new_label.sum() < 10:
            return data_np, seg_np

        label_mc = convert_label_to_multichannel_brats2024(new_label)

        # nnU-Net modality order: T1, T1ce, T2, FLAIR
        mod_map = {0: 't1', 1: 't1ce', 2: 't2', 3: 'flair'}

        for mod_idx, mod_name in mod_map.items():
            if mod_name not in self.generators:
                continue

            scan_crop = data_np[mod_idx, z0:z1, y0:y1, x0:x1].copy()
            scan_norm = rescale_array(scan_crop, -1, 1)
            noisy_scan, noise_mask = add_gaussian_noise_vectorized(scan_norm, new_label)

            noisy_t = torch.from_numpy(noisy_scan).float().reshape(1, 1, 96, 96, 96)
            label_t = torch.from_numpy(label_mc).float().unsqueeze(0)
            gen_input = torch.cat([noisy_t, label_t], dim=1).to(self.device)

            with torch.no_grad():
                recon = self.generators[mod_name](gen_input)
            recon_np = recon.squeeze().cpu().numpy()

            recon_corrected = correct_background_vectorized(scan_norm, recon_np)
            final_crop = linear_interpolation_vectorized(
                recon_corrected, scan_crop, orig_seg_crop, noise_mask)

            data_np[mod_idx, z0:z1, y0:y1, x0:x1] = final_crop

        # Update segmentation
        combined_seg = seg.copy()
        combined_seg[z0:z1, y0:y1, x0:x1] = np.maximum(
            combined_seg[z0:z1, y0:y1, x0:x1], new_label)
        seg_np[0] = combined_seg

        return data_np, seg_np
