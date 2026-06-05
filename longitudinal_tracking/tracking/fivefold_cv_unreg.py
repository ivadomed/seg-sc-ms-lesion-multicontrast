"""
5-fold nested cross-validation for the unregistered lesion tracking method.

Outer loop : 5-fold split on subjects (KFold, shuffle, seed 42).
Inner loop : sweep w_z_over_disk in [10, 15, ..., 50] on the 4/5 training subjects;
             select the w that maximises F1.
             Test subjects are held out for both the w selection and the final evaluation.

Output per fold (fold_<N>/):
    w_optimization/w_z_<w>/<subject>/lesion_mapping.json
    w_optimization_results.json   — F1 per w on training subjects
    test/<subject>/lesion_mapping.json
    test_results.json             — best_w + TP/FP/FN/P/R/F1 for this fold

Global output:
    cv_summary.json               — per-fold + global aggregate
    fivefold_cv_unreg.log

Inputs:
    -i  / --input-msd   : Path to the MSD dataset JSON file.
    -pred / --pred       : Folder containing predicted segmentations.
    -g  / --gt-mapping   : Folder containing ground-truth lesion mapping JSONs.
    -o  / --output-folder: Base output folder.

Author: Pierre-Louis Benveniste
"""
import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.model_selection import KFold

file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, file_path)
sys.path.insert(0, os.path.abspath(os.path.join(file_path, "..", "eval_tracking")))

from track_lesion_unreg import map_lesions_unregistered
from run_evaluation import compare_2_lesion_mappings


W_VALUES = list(range(5, 36, 5))  # [10, 15, 20, 25, 30, 35, 40, 45, 50]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True,
                        help='Path to the MSD dataset JSON file')
    parser.add_argument('-pred', '--pred', type=str, required=True,
                        help='Folder containing predicted segmentations')
    parser.add_argument('-g', '--gt-mapping', type=str, required=True,
                        help='Folder containing ground-truth lesion mapping JSONs')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Base output folder')
    return parser.parse_args()


def track_and_evaluate(subject, data, pred_folder, gt_mapping_folder, output_folder, w):
    """Run tracking for one subject at a given w and return per-GT-lesion TP/FP/FN lists."""
    input_image1 = data[subject]["ses-M0"][0]
    input_image2 = data[subject]["ses-M12"][0]
    subject_output_folder = os.path.join(output_folder, subject)
    os.makedirs(subject_output_folder, exist_ok=True)

    pred_mapping = map_lesions_unregistered(
        input_image1, input_image2,
        os.path.join(pred_folder, subject),
        subject_output_folder,
        w_z_over_disk=w,
    )
    with open(os.path.join(subject_output_folder, 'lesion_mapping.json'), 'w') as f:
        json.dump(pred_mapping, f, indent=4)

    with open(os.path.join(gt_mapping_folder, subject, 'lesion_mapping.json'), 'r') as f:
        gt_mapping = json.load(f)

    return compare_2_lesion_mappings(pred_mapping, gt_mapping)


def aggregate_metrics(tp_lists, fp_lists, fn_lists):
    """Pool per-subject TP/FP/FN lists and compute P/R/F1."""
    total_tp = sum(sum(x) for x in tp_lists)
    total_fp = sum(sum(x) for x in fp_lists)
    total_fn = sum(sum(x) for x in fn_lists)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return int(total_tp), int(total_fp), int(total_fn), round(precision, 4), round(recall, 4), round(f1, 4)


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'fivefold_cv_unreg.log'))

    with open(args.input_msd, 'r') as f:
        msd_data = json.load(f)
    data         = msd_data['data']
    all_subjects = np.array(list(data.keys()))

    logger.info(f"Total subjects: {len(all_subjects)}")
    logger.info(f"W values: {W_VALUES}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results  = []
    fold_tp, fold_fp, fold_fn             = [], [], []
    fold_precision, fold_recall, fold_f1  = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(all_subjects), total=5, desc="5-fold CV")):
        train_subjects = all_subjects[train_idx]
        test_subjects  = all_subjects[test_idx]
        fold_folder    = os.path.join(args.output_folder, f'fold_{fold_idx+1}')
        os.makedirs(fold_folder, exist_ok=True)

        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_idx+1}/5 — {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")

        # ------------------------------------------------------------------
        # Inner loop: select best w on training subjects
        # ------------------------------------------------------------------
        w_metrics = {}
        for w in W_VALUES:
            w_folder = os.path.join(fold_folder, 'w_optimization', f'w_z_{w}')
            tp_all, fp_all, fn_all = [], [], []

            for subject in train_subjects:
                tp, fp, fn = track_and_evaluate(
                    subject, data, args.pred, args.gt_mapping, w_folder, w,
                )
                tp_all.append(tp); fp_all.append(fp); fn_all.append(fn)

            total_tp, total_fp, total_fn, precision, recall, f1 = aggregate_metrics(tp_all, fp_all, fn_all)
            w_metrics[w] = {
                'total_TP': total_tp, 'total_FP': total_fp, 'total_FN': total_fn,
                'precision': precision, 'recall': recall, 'f1': f1,
            }
            logger.info(
                f"  w={w:3d}: TP={total_tp}  FP={total_fp}  FN={total_fn}  "
                f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}"
            )

        with open(os.path.join(fold_folder, 'w_optimization_results.json'), 'w') as f:
            json.dump({str(k): v for k, v in w_metrics.items()}, f, indent=4)

        best_w = max(w_metrics, key=lambda k: w_metrics[k]['f1'])
        logger.info(f"Fold {fold_idx+1} best w: {best_w}  (F1={w_metrics[best_w]['f1']:.4f})")

        # ------------------------------------------------------------------
        # Outer evaluation: test subjects with best w
        # ------------------------------------------------------------------
        test_folder    = os.path.join(fold_folder, 'test')
        tp_all, fp_all, fn_all = [], [], []

        for subject in test_subjects:
            tp, fp, fn = track_and_evaluate(
                subject, data, args.pred, args.gt_mapping, test_folder, best_w,
            )
            tp_all.append(tp); fp_all.append(fp); fn_all.append(fn)

        total_tp, total_fp, total_fn, precision, recall, f1 = aggregate_metrics(tp_all, fp_all, fn_all)
        logger.info(
            f"Fold {fold_idx+1} test (w={best_w}): TP={total_tp}  FP={total_fp}  FN={total_fn}  "
            f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}"
        )

        fold_tp.append(total_tp); fold_fp.append(total_fp); fold_fn.append(total_fn)
        fold_precision.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1)

        fold_result = {
            'fold':           fold_idx + 1,
            'train_subjects': list(train_subjects),
            'test_subjects':  list(test_subjects),
            'best_w':         best_w,
            'total_TP': total_tp, 'total_FP': total_fp, 'total_FN': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'w_optimization': {str(k): v for k, v in w_metrics.items()},
        }
        fold_results.append(fold_result)
        with open(os.path.join(fold_folder, 'test_results.json'), 'w') as f:
            json.dump(fold_result, f, indent=4)

    # Global aggregate (pool predictions across all test folds)
    global_tp = sum(fold_tp)
    global_fp = sum(fold_fp)
    global_fn = sum(fold_fn)
    global_p  = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_r  = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) > 0 else 0.0

    logger.info(f"{'='*60}")
    logger.info(
        f"5-fold CV (mean ± std) — "
        f"Precision: {np.mean(fold_precision):.2f} ± {np.std(fold_precision):.2f}  "
        f"Recall: {np.mean(fold_recall):.2f} ± {np.std(fold_recall):.2f}  "
        f"F1: {np.mean(fold_f1):.2f} ± {np.std(fold_f1):.2f}"
    )
    logger.info(
        f"Global aggregate — TP={global_tp}  FP={global_fp}  FN={global_fn}  "
        f"P={global_p:.4f}  R={global_r:.4f}  F1={global_f1:.4f}"
    )

    summary = {
        'folds':          fold_results,
        'mean_precision': round(float(np.mean(fold_precision)), 4),
        'std_precision':  round(float(np.std(fold_precision)), 4),
        'mean_recall':    round(float(np.mean(fold_recall)), 4),
        'std_recall':     round(float(np.std(fold_recall)), 4),
        'mean_f1':        round(float(np.mean(fold_f1)), 4),
        'std_f1':         round(float(np.std(fold_f1)), 4),
        'global': {
            'total_TP': global_tp, 'total_FP': global_fp, 'total_FN': global_fn,
            'precision': round(global_p, 4),
            'recall':    round(global_r, 4),
            'f1':        round(global_f1, 4),
        },
    }
    summary_path = os.path.join(args.output_folder, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
