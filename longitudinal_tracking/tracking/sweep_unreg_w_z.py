"""
This script evaluates the unregistered lesion tracking method (track_lesion_unreg) across a
range of w_z_over_disk values (10 to 50 by steps of 5).

For each w value it:
  1. Creates a dedicated sub-folder inside the base output folder.
  2. Runs map_lesions_unregistered on every subject in the MSD dataset.
  3. Evaluates the predicted mappings against the ground-truth mappings.
  4. Saves per-subject results and an aggregate summary.

A final JSON comparing precision / recall / F1 across all w values is written to
<output_folder>/summary_w_sweep.json.

Inputs:
    -i  / --input-msd   : Path to the MSD dataset JSON file.
    -pred / --pred       : Path to the folder containing predicted segmentations.
    -g  / --gt-mapping   : Path to the folder containing ground-truth lesion mapping JSONs.
    -o  / --output-folder: Base output folder (one sub-folder per w value will be created).

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger

# Make sibling modules importable
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, file_path)
sys.path.insert(0, os.path.abspath(os.path.join(file_path, "..", "eval_tracking")))

from track_lesion_unreg import map_lesions_unregistered
from run_evaluation import evaluate_lesion_mapping


W_VALUES = list(range(10, 51, 5))  # [10, 15, 20, 25, 30, 35, 40, 45, 50]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True,
                        help='Path to the MSD dataset JSON file')
    parser.add_argument('-pred', '--pred', type=str, required=True,
                        help='Path to the folder containing predicted segmentations')
    parser.add_argument('-g', '--gt-mapping', type=str, required=True,
                        help='Path to the folder containing ground-truth lesion mapping JSONs')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Base output folder (sub-folders per w value will be created here)')
    return parser.parse_args()


def run_tracking_for_w(input_msd_dataset, pred_folder, output_folder, w):
    """Run map_lesions_unregistered for all subjects with the given w_z_over_disk value."""
    os.makedirs(output_folder, exist_ok=True)

    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    for subject in tqdm(data, desc=f"Tracking  w_z={w}"):
        input_image1 = data[subject]["ses-M0"][0]
        input_image2 = data[subject]["ses-M12"][0]
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)
        subject_pred_folder = os.path.join(pred_folder, subject)

        lesion_mapping = map_lesions_unregistered(
            input_image1, input_image2,
            subject_pred_folder, subject_output_folder,
            w_z_over_disk=w
        )

        mapping_path = os.path.join(subject_output_folder, 'lesion_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(lesion_mapping, f, indent=4)


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


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'sweep_w_z_over_disk.log'))

    summary = {}

    for w in W_VALUES:
        logger.info(f"{'='*50}")
        logger.info(f"w_z_over_disk = {w}")

        w_output_folder = os.path.join(args.output_folder, f'w_z_{w}')

        # Step 1 — run tracking for all subjects
        run_tracking_for_w(args.input_msd, args.pred, w_output_folder, w)

        # Step 2 — evaluate against GT mappings
        eval_output_folder = os.path.join(w_output_folder, 'eval')
        results = evaluate_lesion_mapping(
            args.input_msd,
            args.pred,           # pred_seg (for lesion volumes)
            w_output_folder,     # pred_mapping (lesion_mapping.json per subject)
            args.gt_mapping,
            eval_output_folder
        )

        # Step 3 — aggregate metrics and log
        metrics = aggregate_metrics(results)
        metrics['w_z_over_disk'] = w
        logger.info(
            f"w={w}  TP={metrics['total_TP']}  FP={metrics['total_FP']}  FN={metrics['total_FN']}  "
            f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
        )
        summary[str(w)] = metrics

    # Save cross-w summary
    summary_path = os.path.join(args.output_folder, 'summary_w_sweep.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")

    # Report best w by F1
    best_w = max(summary, key=lambda k: summary[k]['f1'])
    logger.info(
        f"Best w_z_over_disk = {best_w}  "
        f"F1 = {summary[best_w]['f1']:.4f}  "
        f"Precision = {summary[best_w]['precision']:.4f}  "
        f"Recall = {summary[best_w]['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
