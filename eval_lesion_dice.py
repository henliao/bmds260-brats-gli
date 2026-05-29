#!/usr/bin/env python3
"""
BraTS 2024 Evaluation Tool (v2)

Computes official BraTS 2024 metrics:
  - Lesion-wise Dice (LW-Dice)
  - Lesion-wise HD95 (LW-HD95)
  - Legacy Dice (standard voxel-level)
  - Legacy HD95 (standard voxel-level)

Regions: NETC(1), SNFH(2), ET(3), RC(4), TC(1+3), WT(1+2+3)

Note: WT = labels 1+2+3 (NETC+SNFH+ET), does NOT include RC(4).
This matches BraTS 2024 official definition.

Usage:
  python3 eval_lesion_dice.py --pred-dir <preds> --gt-dir <gt>
  python3 eval_lesion_dice.py --pred-dir <preds> --gt-dir <gt> --metrics lw-dice lw-hd95
  python3 eval_lesion_dice.py --pred-dir <preds> --gt-dir <gt> --regions NETC ET TC WT
  python3 eval_lesion_dice.py --task 501 --fold 0
  python3 eval_lesion_dice.py --task 501 --fold 0 --metrics lw-dice --regions NETC SNFH ET RC

Flags:
  --metrics    Which metrics to compute (default: all)
               Options: lw-dice, lw-hd95, legacy-dice, legacy-hd95
  --regions    Which regions to evaluate (default: all)
               Options: NETC, SNFH, ET, RC, TC, WT

Requirements:
  pip install connected-components-3d nibabel scipy numpy

Authors: Henry Liao / BMDS 260 Team
Reference: https://github.com/rachitsaluja/BraTS-2024-Metrics
"""
import os, sys, json, glob, argparse
import numpy as np
import nibabel as nib
import cc3d
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, distance_transform_edt
from collections import defaultdict

# Individual labels
LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}

# Composite regions: name -> list of label IDs to OR together
COMPOSITE = {
    "TC": [1, 3],       # Tumor Core = NETC + ET
    "WT": [1, 2, 3],    # Whole Tumor = NETC + SNFH + ET (no RC)
}

ALL_REGIONS = ["NETC", "SNFH", "ET", "RC", "TC", "WT"]
ALL_METRICS = ["lw-dice", "lw-hd95", "legacy-dice", "legacy-hd95"]

STRUCT = generate_binary_structure(3, 2)
HD95_FP_PENALTY = 337.0  # BraTS 2024 official FP penalty for HD95


def get_dil_factor(region):
    """BraTS-GLI dilation factors: 5 for NETC/SNFH/RC, 3 for ET.
    Composite regions use ET's factor (3) if ET is included, else 5."""
    if region == "ET":
        return 3
    if region in COMPOSITE:
        return 3 if 3 in COMPOSITE[region] else 5  # ET is label 3
    return 5


def get_vol_thresh(region):
    """BraTS-GLI volume thresholds: 10 for ET, 20 for others.
    Composite regions use ET's threshold (10) if ET is included, else 20."""
    if region == "ET":
        return 10
    if region in COMPOSITE:
        return 10 if 3 in COMPOSITE[region] else 20
    return 20


def dice_binary(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2.0 * inter / total


def hd95_binary(gt_bin, pred_bin, voxel_spacing=(1.0, 1.0, 1.0)):
    """Compute HD95 using scipy EDT. Returns distance in mm."""
    gt_bin = gt_bin.astype(bool)
    pred_bin = pred_bin.astype(bool)

    if not np.any(gt_bin) and not np.any(pred_bin):
        return 0.0
    if not np.any(gt_bin) or not np.any(pred_bin):
        return HD95_FP_PENALTY

    # Surface voxels (boundary)
    gt_surface = gt_bin ^ binary_erosion(gt_bin, structure=STRUCT)
    pred_surface = pred_bin ^ binary_erosion(pred_bin, structure=STRUCT)

    if not np.any(gt_surface) or not np.any(pred_surface):
        return HD95_FP_PENALTY

    # Distance from gt surface to nearest pred voxel
    dt_pred = distance_transform_edt(~pred_bin, sampling=voxel_spacing)
    d_gt_to_pred = dt_pred[gt_surface]

    # Distance from pred surface to nearest gt voxel
    dt_gt = distance_transform_edt(~gt_bin, sampling=voxel_spacing)
    d_pred_to_gt = dt_gt[pred_surface]

    all_distances = np.concatenate([d_gt_to_pred, d_pred_to_gt])
    return float(np.percentile(all_distances, 95))


def combine_by_dilation(binary_mask, dil_factor):
    """Dilate binary mask, find CCs on dilated version, map back to original voxels.
    This merges nearby small blobs into single lesions."""
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.int32)
    orig_cc = cc3d.connected_components(binary_mask.astype(np.uint8), connectivity=26)
    dilated = binary_dilation(binary_mask, structure=STRUCT, iterations=dil_factor)
    dilated_cc = cc3d.connected_components(dilated.astype(np.uint8), connectivity=26)
    combined = np.zeros_like(orig_cc)
    for dil_id in range(1, dilated_cc.max() + 1):
        dil_mask = (dilated_cc == dil_id)
        overlapping = orig_cc * dil_mask
        orig_ids = np.unique(overlapping)
        orig_ids = orig_ids[orig_ids != 0]
        for oid in orig_ids:
            combined[orig_cc == oid] = dil_id
    return combined


def remove_small_pred(pred_combined, vol_thresh):
    """Remove predicted components with volume <= threshold."""
    result = pred_combined.copy()
    labels, counts = np.unique(result, return_counts=True)
    for lab, cnt in zip(labels, counts):
        if lab != 0 and cnt <= vol_thresh:
            result[result == lab] = 0
    return result


def reindex(arr):
    """Reindex component labels to sequential 1..N."""
    unique = np.unique(arr)
    unique = unique[unique != 0]
    mapping = {old: new for new, old in enumerate(unique, 1)}
    tmp = np.zeros_like(arr)
    for old, new in mapping.items():
        tmp[arr == old] = new
    arr[:] = tmp
    return arr


def get_binary_masks(gt_vol, pred_vol, region):
    """Get binary masks for a region (individual or composite)."""
    if region in COMPOSITE:
        label_ids = COMPOSITE[region]
        gt_bin = np.isin(gt_vol, label_ids).astype(np.uint8)
        pred_bin = np.isin(pred_vol, label_ids).astype(np.uint8)
    else:
        label_id = {v: k for k, v in LABELS.items()}[region]
        gt_bin = (gt_vol == label_id).astype(np.uint8)
        pred_bin = (pred_vol == label_id).astype(np.uint8)
    return gt_bin, pred_bin


def lesion_wise_eval(gt_bin, pred_bin, region, voxel_spacing, compute_hd95=False):
    """Compute lesion-wise Dice and optionally HD95."""
    dil_factor = get_dil_factor(region)
    vol_thresh = get_vol_thresh(region)

    if not np.any(gt_bin) and not np.any(pred_bin):
        return None  # region absent

    gt_combined = reindex(combine_by_dilation(gt_bin, dil_factor))
    pred_combined = reindex(remove_small_pred(combine_by_dilation(pred_bin, dil_factor), vol_thresh))

    n_gt = int(gt_combined.max())
    n_pred = int(pred_combined.max())

    info = {"n_gt": n_gt, "n_pred": n_pred}

    # No GT, only predictions -> all FP
    if n_gt == 0 and n_pred > 0:
        info.update({"tp": 0, "fp": n_pred, "fn": 0})
        result = {"lw_dice": 0.0}
        if compute_hd95:
            result["lw_hd95"] = HD95_FP_PENALTY
        return {**result, **info}

    # GT present, no predictions -> all FN
    if n_gt > 0 and n_pred == 0:
        fn = sum(1 for gid in range(1, n_gt + 1) if (gt_combined == gid).sum() > vol_thresh)
        info.update({"tp": 0, "fp": 0, "fn": fn})
        result = {"lw_dice": 1.0 if fn == 0 else 0.0}
        if compute_hd95:
            result["lw_hd95"] = HD95_FP_PENALTY if fn > 0 else 0.0
        return {**result, **info}

    # Both present: match lesions
    tp_pred = set()
    dice_scores = []
    hd95_scores = []
    fn_count = 0

    for gid in range(1, n_gt + 1):
        gt_mask = (gt_combined == gid)
        gt_vox = gt_mask.sum()
        gt_dilated = binary_dilation(gt_mask, structure=STRUCT, iterations=dil_factor)
        intersecting = np.unique(pred_combined[gt_dilated])
        intersecting = intersecting[intersecting != 0]

        if len(intersecting) > 0:
            pred_match = np.isin(pred_combined, intersecting)
            d = dice_binary(pred_match, gt_mask)
            if gt_vox > vol_thresh:
                dice_scores.append(d)
                if compute_hd95:
                    h = hd95_binary(gt_mask, pred_match, voxel_spacing)
                    hd95_scores.append(h)
            for pid in intersecting:
                tp_pred.add(int(pid))
        else:
            if gt_vox > vol_thresh:
                fn_count += 1

    fp_count = len(set(range(1, n_pred + 1)) - tp_pred)
    denom = len(dice_scores) + fp_count

    lw_dice = sum(dice_scores) / denom if denom > 0 else 1.0

    result = {"lw_dice": round(float(lw_dice), 4)}
    info.update({"tp": len(dice_scores), "fp": fp_count, "fn": fn_count})

    if compute_hd95:
        if denom > 0:
            lw_hd95 = (sum(hd95_scores) + fp_count * HD95_FP_PENALTY) / denom
        else:
            lw_hd95 = 0.0
        result["lw_hd95"] = round(float(lw_hd95), 4)

    return {**result, **info}


def legacy_eval(gt_bin, pred_bin, voxel_spacing, compute_hd95=False):
    """Compute legacy (voxel-level) Dice and optionally HD95."""
    result = {}
    result["legacy_dice"] = round(float(dice_binary(gt_bin, pred_bin)), 4)
    if compute_hd95:
        result["legacy_hd95"] = round(float(hd95_binary(gt_bin, pred_bin, voxel_spacing)), 4)
    return result


def run_evaluation(pred_dir, gt_dir, output_path=None, metrics=None, regions=None):
    """Run evaluation on all NIfTI files in pred_dir."""
    if metrics is None:
        metrics = ALL_METRICS
    if regions is None:
        regions = ALL_REGIONS

    do_lw_dice = "lw-dice" in metrics
    do_lw_hd95 = "lw-hd95" in metrics
    do_legacy_dice = "legacy-dice" in metrics
    do_legacy_hd95 = "legacy-hd95" in metrics
    do_lw = do_lw_dice or do_lw_hd95
    do_legacy = do_legacy_dice or do_legacy_hd95

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    pred_files = [f for f in pred_files if "summary" not in os.path.basename(f) and "plans" not in os.path.basename(f)]
    print(f"Found {len(pred_files)} predictions in {pred_dir}", flush=True)
    print(f"Metrics: {metrics}", flush=True)
    print(f"Regions: {regions}", flush=True)

    results = {"pred_dir": pred_dir, "gt_dir": gt_dir, "metrics": metrics, "regions": regions, "per_case": []}
    region_scores = {r: defaultdict(list) for r in regions}

    for i, pred_path in enumerate(pred_files):
        case_id = os.path.basename(pred_path).replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
        if not os.path.exists(gt_path):
            continue

        pred_nii = nib.load(pred_path)
        pred_vol = pred_nii.get_fdata().astype(np.int16)
        gt_vol = nib.load(gt_path).get_fdata().astype(np.int16)
        voxel_spacing = tuple(float(x) for x in pred_nii.header.get_zooms()[:3])

        case_result = {"case": case_id}

        for region in regions:
            gt_bin, pred_bin = get_binary_masks(gt_vol, pred_vol, region)
            region_result = {}

            # Skip if region absent in both
            if not np.any(gt_bin) and not np.any(pred_bin):
                case_result[region] = "absent"
                continue

            if do_lw:
                lw = lesion_wise_eval(gt_bin, pred_bin, region, voxel_spacing, compute_hd95=do_lw_hd95)
                if lw is None:
                    case_result[region] = "absent"
                    continue
                region_result.update(lw)

            if do_legacy:
                leg = legacy_eval(gt_bin, pred_bin, voxel_spacing, compute_hd95=do_legacy_hd95)
                region_result.update(leg)

            case_result[region] = region_result

            # Accumulate scores
            for key in ["lw_dice", "lw_hd95", "legacy_dice", "legacy_hd95"]:
                if key in region_result:
                    region_scores[region][key].append(region_result[key])

        results["per_case"].append(case_result)
        if (i + 1) % 25 == 0 or (i + 1) == len(pred_files):
            print(f"  Processed {i+1}/{len(pred_files)}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    summary = {}
    for region in regions:
        region_summary = {}
        for key, scores in region_scores[region].items():
            if scores:
                region_summary[key] = {
                    "mean": round(float(np.mean(scores)), 4),
                    "std": round(float(np.std(scores)), 4),
                    "median": round(float(np.median(scores)), 4),
                    "n_cases": len(scores),
                }
        if region_summary:
            summary[region] = region_summary
            parts = []
            for key in ["lw_dice", "lw_hd95", "legacy_dice", "legacy_hd95"]:
                if key in region_summary:
                    s = region_summary[key]
                    parts.append(f"{key}={s['mean']:.4f}+/-{s['std']:.4f}")
            n = region_summary[list(region_summary.keys())[0]]["n_cases"]
            print(f"  {region:5s}: {', '.join(parts)}  (n={n})", flush=True)

    results["summary"] = summary

    if output_path:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="BraTS 2024 Evaluation Tool (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All metrics, all regions:
  python3 eval_lesion_dice.py --pred-dir ./predictions --gt-dir ./ground_truth -o results.json

  # Lesion-wise only, individual labels only (v1 behavior):
  python3 eval_lesion_dice.py --pred-dir ./predictions --gt-dir ./ground_truth --metrics lw-dice --regions NETC SNFH ET RC

  # Just TC and WT with HD95:
  python3 eval_lesion_dice.py --pred-dir ./predictions --gt-dir ./ground_truth --metrics lw-dice lw-hd95 --regions TC WT

  # nnU-Net shortcut:
  python3 eval_lesion_dice.py --task 501 --fold 0

  # nnU-Net shortcut with specific metrics:
  python3 eval_lesion_dice.py --task 501 --fold 0 --metrics lw-dice legacy-dice --regions NETC ET TC
        """)
    parser.add_argument("--pred-dir", help="Directory with prediction NIfTI files (.nii.gz)")
    parser.add_argument("--gt-dir", help="Directory with ground truth segmentation NIfTI files (.nii.gz)")
    parser.add_argument("--task", type=int, help="nnU-Net task ID (e.g., 500, 501)")
    parser.add_argument("--fold", type=int, help="nnU-Net fold number (0-4)")
    parser.add_argument("--trainer", default="nnUNetTrainerV2_MedNeXt_B_kernel5",
                        help="nnU-Net trainer name (default: nnUNetTrainerV2_MedNeXt_B_kernel5)")
    parser.add_argument("--plans", default="nnUNetPlansv2.1",
                        help="nnU-Net plans name (default: nnUNetPlansv2.1)")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS,
                        choices=ALL_METRICS,
                        help="Metrics to compute (default: all). Options: lw-dice, lw-hd95, legacy-dice, legacy-hd95")
    parser.add_argument("--regions", nargs="+", default=ALL_REGIONS,
                        choices=ALL_REGIONS,
                        help="Regions to evaluate (default: all). Options: NETC, SNFH, ET, RC, TC, WT")
    args = parser.parse_args()

    if args.task is not None and args.fold is not None:
        base = os.path.expanduser("~/bmds260/nnunet")
        results_base = os.path.join(base, "results", "nnUNet", "3d_fullres")
        task_dirs = [d for d in os.listdir(results_base) if d.startswith(f"Task{args.task}")]
        if not task_dirs:
            results_base_v1 = os.path.join(base, "results_v1_1350cases", "nnUNet", "3d_fullres")
            if os.path.exists(results_base_v1):
                task_dirs = [d for d in os.listdir(results_base_v1) if d.startswith(f"Task{args.task}")]
                if task_dirs:
                    results_base = results_base_v1
        if not task_dirs:
            print(f"No task directory found for Task{args.task}")
            sys.exit(1)
        task_name = task_dirs[0]
        pred_dir = os.path.join(results_base, task_name,
                                f"{args.trainer}__{args.plans}",
                                f"fold_{args.fold}", "validation_raw")
        preproc_base = os.path.join(base, "preprocessed")
        gt_dir = os.path.join(preproc_base, task_name, "gt_segmentations")
        if not os.path.exists(gt_dir):
            preproc_base = os.path.join(base, "preprocessed_v1_1350cases")
            gt_dir = os.path.join(preproc_base, task_name, "gt_segmentations")
        if not args.output:
            args.output = os.path.expanduser(
                f"~/bmds260/lesion_wise_dice/task{args.task}_fold{args.fold}_lesion_dice.json")
    else:
        pred_dir = args.pred_dir
        gt_dir = args.gt_dir

    if not pred_dir or not gt_dir:
        parser.print_help()
        sys.exit(1)

    if args.output:
        d = os.path.dirname(args.output)
        if d:
            os.makedirs(d, exist_ok=True)
    run_evaluation(pred_dir, gt_dir, args.output, args.metrics, args.regions)


if __name__ == "__main__":
    main()
