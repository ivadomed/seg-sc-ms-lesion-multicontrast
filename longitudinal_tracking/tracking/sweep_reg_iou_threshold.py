"""
Sweeps the IoU threshold for the registered-IoU lesion tracking method and evaluates
performance at each threshold value.

Efficiency: the warp-back (sct_apply_transfo) and IoU matrix computation are performed
ONCE per subject; only the cheap threshold-dependent mapping step is repeated.

Thresholds: 1e-5, 1e-3, 1%, 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%

Inputs:
    -i  / --input-msd    : Path to the MSD dataset JSON file.
    -pred / --pred        : Path to the folder containing predicted segmentations
                            (must contain pre-computed registration files).
    -g  / --gt-mapping    : Path to the folder containing ground-truth lesion mappings.
    -o  / --output-folder : Base output folder (one sub-folder per threshold is created).
    --set                 : Optional dataset split filter (train / val / test).

Outputs:
    {output}/{threshold_label}/{subject}/lesion_mapping.json  — per-subject mappings
    {output}/{threshold_label}/eval/overall_results.json      — per-threshold evaluation
    {output}/summary_iou_sweep.json                           — aggregated P/R/F1 table
    {output}/sweep_iou_threshold.log

Author: Pierre-Louis Benveniste
"""
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# Make sibling and eval_tracking modules importable
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, file_path)
sys.path.insert(0, os.path.abspath(os.path.join(file_path, "..", "eval_tracking")))

from run_evaluation import evaluate_lesion_mapping


THRESHOLDS = [1e-5, 1e-3, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep IoU threshold for registered-IoU lesion tracking."
    )
    parser.add_argument('-i', '--input-msd', type=str, required=True,
                        help='Path to the MSD dataset JSON file')
    parser.add_argument('-pred', '--pred', type=str, required=True,
                        help='Path to the folder containing predicted segmentations')
    parser.add_argument('-g', '--gt-mapping', type=str, required=True,
                        help='Path to the folder containing ground-truth lesion mapping JSONs')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Base output folder (sub-folders per threshold will be created here)')
    parser.add_argument('--set', type=str, choices=['train', 'val', 'test'], default=None,
                        help="Dataset split filter (train / val / test); default: all subjects")
    return parser.parse_args()


def threshold_label(t):
    """Return a filesystem-safe string label for a threshold value."""
    if t < 0.01:
        return f"thresh_{t:.0e}"
    return f"thresh_{t:.2f}"


def compute_IoU_matrix(data_lesion_1, data_lesion_2):
    """Compute the pairwise IoU matrix between labeled lesion maps."""
    n1 = len(np.unique(data_lesion_1)) - 1  # exclude background
    n2 = len(np.unique(data_lesion_2)) - 1
    iou = np.zeros((n1, n2))
    for i in range(1, n1 + 1):
        mask1 = (data_lesion_1 == i)
        for j in range(1, n2 + 1):
            mask2 = (data_lesion_2 == j)
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou[i - 1, j - 1] = intersection / union if union > 0 else 0.0
    return iou


def apply_threshold(iou_matrix, threshold):
    """Return a dict mapping lesion index (1-based) → list of matched lesion indices."""
    n1, n2 = iou_matrix.shape
    mapping = {}
    for i in range(n1):
        mapping[i + 1] = [j + 1 for j in range(n2) if iou_matrix[i, j] >= threshold]
    return mapping


def aggregate_metrics(results):
    """Compute global TP/FP/FN and derived metrics from evaluate_lesion_mapping output."""
    total_tp = sum(sum(r['TP']) for r in results.values())
    total_fp = sum(sum(r['FP']) for r in results.values())
    total_fn = sum(sum(r['FN']) for r in results.values())
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        'total_TP': int(total_tp),
        'total_FP': int(total_fp),
        'total_FN': int(total_fn),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }


def precompute_subject_iou_matrices(data, pred_folder, temp_folder):
    """
    For each subject, compute:
      - IoU matrix between labeled lesion seg at T1 and registered labeled lesion seg of T2
      - Warp the registered labeled lesion seg of T2 back to T2 space (sct_apply_transfo, once)
      - IoU matrix between the warped-back seg and the labeled lesion seg at T2

    Returns a dict: {subject_id: {'iou_fwd': array, 'iou_bwd': array}}
    """
    subject_matrices = {}

    for subject in tqdm(data, desc="Precomputing IoU matrices"):
        input_image1 = data[subject]["ses-M0"][0]
        input_image2 = data[subject]["ses-M12"][0]
        image1_name = Path(input_image1).name
        image2_name = Path(input_image2).name
        subject_pred = os.path.join(pred_folder, subject)

        labeled_seg_1 = os.path.join(subject_pred, image1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_seg_2 = os.path.join(subject_pred, image2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        labeled_seg_2_reg = os.path.join(subject_pred, image2_name.replace('.nii.gz', '_lesion-seg-reg-labeled.nii.gz'))
        inv_warp = os.path.join(subject_pred, image2_name.replace('.nii.gz', '_inv_warp_to_' + image1_name))

        # Verify required files exist
        for path, desc in [
            (labeled_seg_1,      "labeled lesion seg T1"),
            (labeled_seg_2,      "labeled lesion seg T2"),
            (labeled_seg_2_reg,  "registered labeled lesion seg T2"),
            (inv_warp,           "inverse warp field"),
        ]:
            if not os.path.isfile(path):
                logger.warning(f"[{subject}] Missing {desc}: {path} — skipping subject")
                break
        else:
            # --- Forward IoU matrix: T1 vs registered T2 ---
            data_seg_1 = nib.load(labeled_seg_1).get_fdata()
            data_seg_2_reg = nib.load(labeled_seg_2_reg).get_fdata()
            iou_fwd = compute_IoU_matrix(data_seg_1, data_seg_2_reg)

            # --- Warp registered T2 back to T2 space (once per subject) ---
            subject_temp = os.path.join(temp_folder, subject)
            os.makedirs(subject_temp, exist_ok=True)
            warpback_path = os.path.join(
                subject_temp,
                image2_name.replace('.nii.gz', '_lesion-seg-reg-back-labeled.nii.gz')
            )
            if not os.path.isfile(warpback_path):
                ret = os.system(
                    f"sct_apply_transfo -i {labeled_seg_2_reg} -d {input_image2} "
                    f"-w {inv_warp} -o {warpback_path} -x nn"
                )
                if ret != 0:
                    logger.warning(f"[{subject}] sct_apply_transfo failed — skipping subject")
                    continue

            # --- Backward IoU matrix: warped-back T2 vs T2 ---
            data_seg_2_back = nib.load(warpback_path).get_fdata()
            data_seg_2 = nib.load(labeled_seg_2).get_fdata()
            iou_bwd = compute_IoU_matrix(data_seg_2_back, data_seg_2)

            subject_matrices[subject] = {'iou_fwd': iou_fwd, 'iou_bwd': iou_bwd}
            logger.info(
                f"[{subject}] IoU matrices computed — "
                f"forward: {iou_fwd.shape}, backward: {iou_bwd.shape}"
            )

    return subject_matrices


def run_tracking_for_threshold(subject_matrices, threshold, thresh_output_folder):
    """Apply the IoU threshold to all subjects and save lesion mappings."""
    for subject, matrices in subject_matrices.items():
        subject_dir = os.path.join(thresh_output_folder, subject)
        os.makedirs(subject_dir, exist_ok=True)

        mapping_1_to_reg = apply_threshold(matrices['iou_fwd'], threshold)
        mapping_reg_back_to_2 = apply_threshold(matrices['iou_bwd'], threshold)

        # Combine: T1 → registered T2 → T2
        full_mapping = {}
        for les1, mapped_reg in mapping_1_to_reg.items():
            full_mapped = []
            for les_reg in mapped_reg:
                if les_reg in mapping_reg_back_to_2:
                    full_mapped.extend(mapping_reg_back_to_2[les_reg])
            full_mapping[les1] = full_mapped

        mapping_path = os.path.join(subject_dir, 'lesion_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(full_mapping, f, indent=4)


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'sweep_iou_threshold.log'))

    # Load dataset
    with open(args.input_msd, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # Apply optional split filter (mirrors run_evaluation.py logic)
    if args.set == 'test':
        data = {k: v for k, v in data.items() if 'tor' in k}
    elif args.set == 'train':
        data = {k: v for k, v in data.items() if 'edm' in k or 'van' in k}
    elif args.set == 'val':
        data = {k: v for k, v in data.items() if 'cal' in k or 'mon' in k}

    logger.info(f"Subjects to process: {len(data)}")
    logger.info(f"Thresholds to sweep: {THRESHOLDS}")

    # Phase 1 — precompute IoU matrices (once per subject)
    temp_folder = os.path.join(args.output_folder, '_temp')
    subject_matrices = precompute_subject_iou_matrices(data, args.pred, temp_folder)
    logger.info(f"Precomputed matrices for {len(subject_matrices)} subjects")

    summary = {}

    # Phase 2 — sweep thresholds
    for threshold in THRESHOLDS:
        label = threshold_label(threshold)
        logger.info(f"{'='*60}")
        logger.info(f"IoU threshold = {threshold}  (label: {label})")

        thresh_output = os.path.join(args.output_folder, label)

        # 2a. Compute and save lesion mappings for this threshold
        run_tracking_for_threshold(subject_matrices, threshold, thresh_output)

        # 2b. Evaluate mappings against GT
        eval_output = os.path.join(thresh_output, 'eval')
        results = evaluate_lesion_mapping(
            args.input_msd,
            args.pred,        # pred_seg (for lesion volumes)
            thresh_output,    # pred_mapping folder (contains {subject}/lesion_mapping.json)
            args.gt_mapping,
            eval_output,
            set=args.set,
        )

        # 2c. Aggregate and log
        metrics = aggregate_metrics(results)
        metrics['threshold'] = threshold
        logger.info(
            f"threshold={threshold}  "
            f"TP={metrics['total_TP']}  FP={metrics['total_FP']}  FN={metrics['total_FN']}  "
            f"Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )
        summary[label] = metrics

    # Save cross-threshold summary
    summary_path = os.path.join(args.output_folder, 'summary_iou_sweep.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")

    # Report best threshold per metric
    best_f1_key = max(summary, key=lambda k: summary[k]['f1'])
    best_p_key  = max(summary, key=lambda k: summary[k]['precision'])
    best_r_key  = max(summary, key=lambda k: summary[k]['recall'])
    logger.info(
        f"Best threshold by F1        : {best_f1_key}  "
        f"(F1={summary[best_f1_key]['f1']:.4f}  "
        f"P={summary[best_f1_key]['precision']:.4f}  "
        f"R={summary[best_f1_key]['recall']:.4f})"
    )
    logger.info(
        f"Best threshold by Precision : {best_p_key}  "
        f"(P={summary[best_p_key]['precision']:.4f})"
    )
    logger.info(
        f"Best threshold by Recall    : {best_r_key}  "
        f"(R={summary[best_r_key]['recall']:.4f})"
    )


if __name__ == "__main__":
    main()
