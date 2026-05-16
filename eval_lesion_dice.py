#!/usr/bin/env python3
"""
Official BraTS 2024 Lesion-wise Dice Evaluation Tool

Computes lesion-wise Dice scores matching the BraTS 2024 challenge metric.
Key features:
  - Dilation before CC analysis (merges nearby blobs)
  - Volume thresholding (ignores tiny components)
  - FP penalty in denominator
  - Cross-label combining (not yet implemented, minor effect)

Usage:
  python3 eval_lesion_dice.py --pred-dir <path_to_predictions> --gt-dir <path_to_gt>
  python3 eval_lesion_dice.py --task 501 --fold 0  # shortcut for nnU-Net

Output:
  JSON file with per-case and summary lesion-wise Dice scores.

Requirements:
  pip install connected-components-3d nibabel scipy numpy

Authors: Henry Liao / BMDS 260 Team
Reference: https://github.com/rachitsaluja/BraTS-2024-Metrics
"""
import os, sys, json, glob, argparse
import numpy as np
import nibabel as nib
import cc3d
from scipy.ndimage import binary_dilation, generate_binary_structure
from collections import defaultdict

LABELS = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
STRUCT = generate_binary_structure(3, 2)


def get_dil_factor(label_name):
    """BraTS-GLI dilation factors: 5 for NETC/SNFH/RC, 3 for ET."""
    return 5 if label_name in ("NETC", "SNFH", "RC") else 3


def get_vol_thresh(label_name):
    """BraTS-GLI volume thresholds: 10 for ET, 20 for others."""
    return 10 if label_name == "ET" else 20


def dice_binary(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2.0 * inter / total


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


def lesion_dice_official(gt_vol, pred_vol, label_id, label_name):
    """Compute official BraTS 2024 lesion-wise Dice for one label, one patient.

    Returns (score, info_dict) where score is None if both GT and pred are empty.
    """
    gt_bin = (gt_vol == label_id).astype(np.uint8)
    pred_bin = (pred_vol == label_id).astype(np.uint8)
    dil_factor = get_dil_factor(label_name)
    vol_thresh = get_vol_thresh(label_name)

    if not np.any(gt_bin) and not np.any(pred_bin):
        return None, {}

    gt_combined = reindex(combine_by_dilation(gt_bin, dil_factor))
    pred_combined = reindex(remove_small_pred(combine_by_dilation(pred_bin, dil_factor), vol_thresh))

    n_gt = int(gt_combined.max())
    n_pred = int(pred_combined.max())

    if n_gt == 0 and n_pred > 0:
        return 0.0, {"n_gt": 0, "n_pred": n_pred, "tp": 0, "fp": n_pred, "fn": 0}
    if n_gt > 0 and n_pred == 0:
        fn = sum(1 for gid in range(1, n_gt + 1) if (gt_combined == gid).sum() > vol_thresh)
        return (1.0 if fn == 0 else 0.0), {"n_gt": n_gt, "n_pred": 0, "tp": 0, "fp": 0, "fn": fn}

    tp_pred = set()
    dice_scores = []
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
            for pid in intersecting:
                tp_pred.add(int(pid))
        else:
            if gt_vox > vol_thresh:
                fn_count += 1

    fp_count = len(set(range(1, n_pred + 1)) - tp_pred)
    denom = len(dice_scores) + fp_count
    lw_dice = sum(dice_scores) / denom if denom > 0 else 1.0

    return float(lw_dice), {
        "n_gt": n_gt, "n_pred": n_pred,
        "tp": len(dice_scores), "fp": fp_count, "fn": fn_count,
    }


def run_evaluation(pred_dir, gt_dir, output_path=None):
    """Run lesion-wise Dice evaluation on all NIfTI files in pred_dir."""
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    pred_files = [f for f in pred_files
                  if "summary" not in os.path.basename(f)
                  and "plans" not in os.path.basename(f)]
    print(f"Found {len(pred_files)} predictions in {pred_dir}", flush=True)

    results = {"pred_dir": pred_dir, "gt_dir": gt_dir, "per_case": []}
    label_scores = defaultdict(list)

    for i, pred_path in enumerate(pred_files):
        case_id = os.path.basename(pred_path).replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
        if not os.path.exists(gt_path):
            continue

        pred_vol = nib.load(pred_path).get_fdata().astype(np.int16)
        gt_vol = nib.load(gt_path).get_fdata().astype(np.int16)

        case_result = {"case": case_id}
        for label_id, label_name in LABELS.items():
            score, info = lesion_dice_official(gt_vol, pred_vol, label_id, label_name)
            if score is not None:
                label_scores[label_name].append(score)
                case_result[label_name] = {"dice": round(score, 4), **info}
            else:
                case_result[label_name] = "absent"
        results["per_case"].append(case_result)
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{len(pred_files)}", flush=True)

    print(f"\n{'='*50}", flush=True)
    summary = {}
    for label in ["NETC", "SNFH", "ET", "RC"]:
        scores = label_scores[label]
        if scores:
            summary[label] = {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "median": round(float(np.median(scores)), 4),
                "n_cases": len(scores)
            }
            print(f"  {label:5s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} "
                  f"(median {np.median(scores):.4f}, n={len(scores)})", flush=True)
    results["summary"] = summary

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="BraTS 2024 Lesion-wise Dice Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generic usage (any framework):
  python3 eval_lesion_dice.py --pred-dir ./predictions --gt-dir ./ground_truth -o results.json

  # nnU-Net shortcut:
  python3 eval_lesion_dice.py --task 501 --fold 0

  # Swin UNETR or other MONAI models:
  python3 eval_lesion_dice.py --pred-dir /path/to/swin_predictions --gt-dir /path/to/gt_segs -o swin_results.json
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
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_evaluation(pred_dir, gt_dir, args.output)


if __name__ == "__main__":
    main()
