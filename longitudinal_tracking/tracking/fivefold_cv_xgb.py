"""
5-fold nested cross-validation for XGBoost lesion tracking.

Outer loop : 5-fold split on subjects (KFold, shuffle, seed 42).
Inner loop : BayesSearchCV with cv=3 on the training fold selects hyperparameters.
             Test subjects are fully held out during this selection — this constitutes
             the nested structure that prevents optimism bias.

Inputs:
    -d / --data          : Path to the csv dataset containing lesion features over time.
    -o / --output-folder : Output folder where results will be stored.

Output:
    cross_validation_results.json   — per-fold metrics + global aggregate
    fivefold_cv_xgb.log

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from loguru import logger
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import KFold
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to the csv dataset containing lesion features over time')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Path to the output folder where results will be stored')
    return parser.parse_args()


def distance_cylindrical(coord1, coord2, w_z=1.0, w_disk=1.0):
    distance_disk = np.sqrt(
        coord1['r']**2 + coord2['r']**2
        - 2 * coord1['r'] * coord2['r'] * np.cos(np.radians(coord1['theta'] - coord2['theta']))
    )
    z_dist = coord1['z'] - coord2['z']
    return np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)


def serialize_params(params):
    """Convert numpy scalar types to plain Python types for JSON serialisation."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'fivefold_cv_xgb.log'))

    df = pd.read_csv(args.data)

    # Build one row per lesion pair (same subject, across exactly 2 timepoints)
    lesion_pairs = []
    cv_subjects  = []

    for subject in df['subject'].unique():
        df_subject = df[df['subject'] == subject]
        timepoints = df_subject['timepoint'].unique()
        if len(timepoints) != 2:
            continue
        cv_subjects.append(subject)
        tp1, tp2 = timepoints
        df_tp1 = df_subject[df_subject['timepoint'] == tp1]
        df_tp2 = df_subject[df_subject['timepoint'] == tp2]

        for _, lesion1 in df_tp1.iterrows():
            for _, lesion2 in df_tp2.iterrows():
                pair_features = {'subject': subject}
                for col in df.columns:
                    if col not in ['subject', 'timepoint', 'group']:
                        pair_features[f'{col}1'] = lesion1[col]
                        pair_features[f'{col}2'] = lesion2[col]
                pair_features['dz']     = lesion1['z'] - lesion2['z']
                pair_features['dr']     = lesion1['r'] - lesion2['r']
                pair_features['dtheta'] = lesion1['theta'] - lesion2['theta']
                pair_features['dvol']   = abs(lesion1['volume'] - lesion2['volume'])
                pair_features['dist']   = distance_cylindrical(
                    {'r': lesion1['r'], 'theta': lesion1['theta'], 'z': lesion1['z']},
                    {'r': lesion2['r'], 'theta': lesion2['theta'], 'z': lesion2['z']},
                )
                pair_features['label'] = 1 if lesion1['group'] == lesion2['group'] else 0
                lesion_pairs.append(pair_features)

    df_pairs = pd.DataFrame(lesion_pairs)
    # Fix a known label error (see https://github.com/ivadomed/ms-lesion-agnostic/issues/98)
    df_pairs.at[684, 'label'] = 0

    X = df_pairs.drop(columns=['label'])
    y = df_pairs[['subject', 'label']]

    logger.info(f"Total subjects with 2 timepoints: {len(cv_subjects)}")
    logger.info(f"Total lesion pairs: {len(df_pairs)}")

    subjects_array = np.array(cv_subjects)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results  = []
    fold_tp, fold_fp, fold_fn             = [], [], []
    fold_precision, fold_recall, fold_f1  = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(subjects_array), total=5, desc="5-fold CV")):
        train_subjects = subjects_array[train_idx]
        test_subjects  = subjects_array[test_idx]
        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_idx+1}/5 — {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")

        train_mask = df_pairs['subject'].isin(train_subjects)
        test_mask  = df_pairs['subject'].isin(test_subjects)

        X_train = X[train_mask].drop(columns=['subject'])
        y_train = y[train_mask]['label']
        X_test  = X[test_mask].drop(columns=['subject'])
        y_test  = y[test_mask]['label']

        # Inner loop: BayesSearchCV with cv=3 — hyperparameters selected on training fold only
        search_spaces = {
            'max_depth':        Integer(3, 10),
            'min_child_weight': Integer(1, 10),
            'subsample':        Real(0.5, 1),
            'colsample_bytree': Real(0.001, 1),
            'learning_rate':    Real(0.01, 0.5, prior='log-uniform'),
            'scale_pos_weight': Integer(5, 10, prior='log-uniform'),
        }
        model  = XGBClassifier(seed=42, eval_metric='logloss')
        search = BayesSearchCV(
            model, search_spaces, n_iter=50, n_jobs=1, cv=3,
            random_state=42, scoring='average_precision', verbose=0,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        logger.info(f"Fold {fold_idx+1} best params: {search.best_params_}")

        # Evaluate on held-out test fold
        y_pred    = model.predict(X_test)
        tp        = int(sum((y_test == 1) & (y_pred == 1)))
        fp        = int(sum((y_test == 0) & (y_pred == 1)))
        fn        = int(sum((y_test == 1) & (y_pred == 0)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(
            f"Fold {fold_idx+1}: TP={tp}  FP={fp}  FN={fn}  "
            f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}"
        )

        fold_tp.append(tp); fold_fp.append(fp); fold_fn.append(fn)
        fold_precision.append(round(precision, 4))
        fold_recall.append(round(recall, 4))
        fold_f1.append(round(f1, 4))

        fold_results.append({
            'fold':            fold_idx + 1,
            'train_subjects':  list(train_subjects),
            'test_subjects':   list(test_subjects),
            'best_params':     serialize_params(search.best_params_),
            'total_TP': tp, 'total_FP': fp, 'total_FN': fn,
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'f1':        round(f1, 4),
        })

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

    results = {
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
    results_path = os.path.join(args.output_folder, 'cross_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
