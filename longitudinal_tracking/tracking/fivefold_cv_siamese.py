"""
5-fold cross-validation for the Siamese network lesion tracking.

Outer loop : 5-fold split on subjects (KFold, shuffle, seed 42).
             The Siamese network has a fixed architecture with no inner hyperparameter
             search, so the nested structure is implicit: the scaler is fit and the model
             is trained exclusively on the 4/5 training fold; test subjects are never seen.

Inputs:
    -d / --data          : Path to the csv dataset containing lesion features over time.
    -o / --output-folder : Output folder where results will be stored.

Output:
    cross_validation_results.json   — per-fold metrics + global aggregate
    fivefold_cv_siamese.log

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
from loguru import logger
from keras import layers, models, Input, optimizers
from keras import ops
import keras
import tensorflow.keras.backend as K
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.model_selection import KFold

keras.utils.set_random_seed(812)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True,
                        help='Path to the csv dataset containing lesion features over time')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Path to the output folder where results will be stored')
    return parser.parse_args()


def contrastive_loss_margin(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred  = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - y_pred, 0))
        return ops.mean((1 - y_true) * square_pred + y_true * margin_square)
    return contrastive_loss


def build_siamese_network(num_features_per_lesion):
    input_enc = Input(shape=(num_features_per_lesion,))
    x         = layers.Dense(128, activation='relu')(input_enc)
    x         = layers.BatchNormalization()(x)
    x         = layers.Dense(64, activation='relu')(x)
    embedding = layers.Dense(32, activation='linear')(x)
    encoder   = models.Model(inputs=input_enc, outputs=embedding)

    input_a     = Input(shape=(num_features_per_lesion,))
    input_b     = Input(shape=(num_features_per_lesion,))
    processed_a = encoder(input_a)
    processed_b = encoder(input_b)

    L2_layer   = layers.Lambda(
        lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True))
    )
    L2_distance = L2_layer([processed_a, processed_b])
    prediction  = layers.Dense(1, activation='sigmoid')(L2_distance)
    return models.Model(inputs=[input_a, input_b], outputs=prediction)


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'fivefold_cv_siamese.log'))

    df = pd.read_csv(args.data)

    # Build one row per lesion in each timepoint (separate inputs for the Siamese network)
    lesions_1, lesions_2, lesion_pairs_labels = [], [], []
    cv_subjects = []

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
                l1 = {'subject': subject}
                l2 = {'subject': subject}
                lp = {'subject': subject}
                for col in df.columns:
                    if col not in ['subject', 'timepoint', 'group']:
                        l1[col] = lesion1[col]
                        l2[col] = lesion2[col]
                lp['label'] = 1 if lesion1['group'] == lesion2['group'] else 0
                lesions_1.append(l1)
                lesions_2.append(l2)
                lesion_pairs_labels.append(lp)

    df_pairs    = pd.DataFrame(lesion_pairs_labels)
    lesion_1_df = pd.DataFrame(lesions_1)
    lesion_2_df = pd.DataFrame(lesions_2)

    # Fix a known label error (see https://github.com/ivadomed/ms-lesion-agnostic/issues/98)
    df_pairs.at[684, 'label'] = 0

    feature_cols   = [col for col in lesion_1_df.columns if col != 'subject']
    subjects_array = np.array(cv_subjects)
    kf             = KFold(n_splits=5, shuffle=True, random_state=41)

    logger.info(f"Total subjects with 2 timepoints: {len(cv_subjects)}")
    logger.info(f"Total lesion pairs: {len(df_pairs)}")

    fold_results  = []
    fold_tp, fold_fp, fold_fn             = [], [], []
    fold_precision, fold_recall, fold_f1  = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(subjects_array)):
        train_subjects = subjects_array[train_idx]
        test_subjects  = subjects_array[test_idx]
        logger.info(f"{'='*60}")
        logger.info(f"Fold {fold_idx+1}/5 — {len(train_subjects)} train subjects, {len(test_subjects)} test subjects")

        train_mask = lesion_1_df['subject'].isin(train_subjects)
        test_mask  = lesion_1_df['subject'].isin(test_subjects)

        X_train_1_raw = lesion_1_df[train_mask][feature_cols]
        X_train_2_raw = lesion_2_df[train_mask][feature_cols]
        y_train       = df_pairs[train_mask]['label'].reset_index(drop=True)

        X_test_1_raw  = lesion_1_df[test_mask][feature_cols]
        X_test_2_raw  = lesion_2_df[test_mask][feature_cols]
        y_test        = df_pairs[test_mask]['label'].reset_index(drop=True)

        # Fit scaler on training data only — never touch test fold
        scaler    = StandardScaler()
        X_train_1 = scaler.fit_transform(X_train_1_raw)
        X_train_2 = scaler.transform(X_train_2_raw)
        X_test_1  = scaler.transform(X_test_1_raw)
        X_test_2  = scaler.transform(X_test_2_raw)

        # Build and compile a fresh model for each fold
        model     = build_siamese_network(X_train_1.shape[1])
        optimizer = optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=contrastive_loss_margin(margin=1), metrics=['AUC'])

        weights = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train,
        )
        class_weights_dict = {0: weights[0], 1: weights[1]}
        logger.info(f"Fold {fold_idx+1} class weights: {class_weights_dict}")

        model.fit(
            [X_train_1, X_train_2], y_train,
            epochs=500, batch_size=20,
            class_weight=class_weights_dict,
            callbacks=[TqdmCallback(verbose=0)],
            verbose=0,
        )

        # Evaluate on held-out test fold
        y_pred        = model.predict([X_test_1, X_test_2], verbose=0)
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        tp        = int(sum((y_test == 1) & (y_pred_labels == 1)))
        fp        = int(sum((y_test == 0) & (y_pred_labels == 1)))
        fn        = int(sum((y_test == 1) & (y_pred_labels == 0)))
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
            'fold':           fold_idx + 1,
            'train_subjects': list(train_subjects),
            'test_subjects':  list(test_subjects),
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
