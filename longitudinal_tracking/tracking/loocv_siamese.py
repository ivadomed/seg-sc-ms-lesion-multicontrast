"""
In this script, we perform lesion mapping between two timepoints using a Siamese network to predict lesion
correspondences based on features extracted from the lesions.
The specificity of this script is that it uses LOOCV to estimate the performance of the model on the entire dataset.

Inputs:
    - dataset_csv: Path to the csv dataset containing lesion features over time.
    - output_folder: Path to the folder where model and results will be saved.

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import pandas as pd
import argparse
from loguru import logger
import numpy as np
from keras import layers, models, Input, optimizers
from keras import ops
import keras
import tensorflow.keras.backend as K
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import json

keras.utils.set_random_seed(812)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to the csv dataset containing lesion features over time')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where model and results will be stored')
    return parser.parse_args()


def distance_cylindrical(coord1, coord2, w_z=1.0, w_disk=1.0):
    """
    Computes a weighted Euclidean distance between two points in cylindrical coordinates.
    """
    distance_disk = np.sqrt(coord1['r']**2 + coord2['r']**2 - 2 * coord1['r'] * coord2['r'] * np.cos(np.radians(coord1['theta'] - coord2['theta'])))
    z_dist = coord1['z'] - coord2['z']
    return np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)


def contrastive_loss_margin(margin=1):
    def contrastive_loss(y_true, y_pred):
        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)
    return contrastive_loss


def build_siamese_network(num_features_per_lesion):
    input_enc = Input(shape=(num_features_per_lesion,))
    x = layers.Dense(128, activation='relu')(input_enc)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    embedding = layers.Dense(32, activation='linear')(x)
    encoder = models.Model(inputs=input_enc, outputs=embedding)

    input_a = Input(shape=(num_features_per_lesion,))
    input_b = Input(shape=(num_features_per_lesion,))
    processed_a = encoder(input_a)
    processed_b = encoder(input_b)

    L2_layer = layers.Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))
    L2_distance = L2_layer([processed_a, processed_b])

    prediction = layers.Dense(1, activation='sigmoid')(L2_distance)
    model = models.Model(inputs=[input_a, input_b], outputs=prediction)
    return model


def main():
    args = parse_args()
    dataset_csv = args.data
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)
    logger.add(os.path.join(output_folder, 'lesion_mapping_siamese_loocv.log'))

    df = pd.read_csv(dataset_csv)

    # Build one row per lesion pair (same subject, across exactly 2 timepoints)
    lesions_1 = []
    lesions_2 = []
    lesion_pairs_labels = []
    loocv_subjects = []

    for subject in df['subject'].unique():
        df_subject = df[df['subject'] == subject]
        timepoints = df_subject['timepoint'].unique()
        if len(timepoints) != 2:
            continue
        loocv_subjects.append(subject)
        tp1, tp2 = timepoints
        df_tp1 = df_subject[df_subject['timepoint'] == tp1]
        df_tp2 = df_subject[df_subject['timepoint'] == tp2]

        for _, lesion1 in df_tp1.iterrows():
            for _, lesion2 in df_tp2.iterrows():
                lesion1_features = {'subject': subject}
                lesion2_features = {'subject': subject}
                lesion_pair = {'subject': subject}
                for col in df.columns:
                    if col not in ['subject', 'timepoint', 'group']:
                        lesion1_features[col] = lesion1[col]
                        lesion2_features[col] = lesion2[col]
                lesion_pair['label'] = 1 if lesion1['group'] == lesion2['group'] else 0
                lesions_1.append(lesion1_features)
                lesions_2.append(lesion2_features)
                lesion_pairs_labels.append(lesion_pair)

    df_pairs = pd.DataFrame(lesion_pairs_labels)
    lesion_1_df = pd.DataFrame(lesions_1)
    lesion_2_df = pd.DataFrame(lesions_2)

    # Fix a known label error (see https://github.com/ivadomed/ms-lesion-agnostic/issues/98)
    df_pairs.at[684, 'label'] = 0

    feature_cols = [col for col in lesion_1_df.columns if col != 'subject']

    logger.info(f"Total subjects with 2 timepoints: {len(loocv_subjects)}")
    logger.info(f"Total lesion pairs: {len(df_pairs)}")

    cv_tp_scores = []
    cv_fp_scores = []
    cv_fn_scores = []
    cv_precision_scores = []
    cv_recall_scores = []
    cv_f1_scores = []
    cv_subjects = []

    for subject in tqdm(loocv_subjects):
        train_mask = lesion_1_df['subject'] != subject
        test_mask = lesion_1_df['subject'] == subject

        X_train_1_raw = lesion_1_df[train_mask][feature_cols]
        X_train_2_raw = lesion_2_df[train_mask][feature_cols]
        y_train = df_pairs[train_mask]['label'].reset_index(drop=True)

        X_test_1_raw = lesion_1_df[test_mask][feature_cols]
        X_test_2_raw = lesion_2_df[test_mask][feature_cols]
        y_test = df_pairs[test_mask]['label'].reset_index(drop=True)

        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train_1 = scaler.fit_transform(X_train_1_raw)
        X_train_2 = scaler.transform(X_train_2_raw)
        X_test_1 = scaler.transform(X_test_1_raw)
        X_test_2 = scaler.transform(X_test_2_raw)

        # Build and compile a fresh model for each fold
        model = build_siamese_network(X_train_1.shape[1])
        optimizer = optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=contrastive_loss_margin(margin=1), metrics=['AUC'])

        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = {0: weights[0], 1: weights[1]}
        logger.info(f"Subject {subject} — class weights: {class_weights_dict}")

        model.fit(
            [X_train_1, X_train_2], y_train,
            epochs=500, batch_size=20,
            class_weight=class_weights_dict,
            callbacks=[TqdmCallback(verbose=0)],
            verbose=0
        )

        # Evaluate on held-out subject
        y_pred = model.predict([X_test_1, X_test_2], verbose=0)
        y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        tp = int(sum((y_test == 1) & (y_pred_labels == 1)))
        fp = int(sum((y_test == 0) & (y_pred_labels == 1)))
        fn = int(sum((y_test == 1) & (y_pred_labels == 0)))
        logger.info(f"Subject {subject} — TP: {tp}, FP: {fp}, FN: {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        logger.info(f"Subject {subject} — Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        cv_tp_scores.append(tp)
        cv_fp_scores.append(fp)
        cv_fn_scores.append(fn)
        cv_precision_scores.append(precision)
        cv_recall_scores.append(recall)
        cv_f1_scores.append(f1)
        cv_subjects.append(subject)

    logger.info(
        f"LOOCV — Precision: {np.mean(cv_precision_scores):.2f} ± {np.std(cv_precision_scores):.2f}, "
        f"Recall: {np.mean(cv_recall_scores):.2f} ± {np.std(cv_recall_scores):.2f}, "
        f"F1: {np.mean(cv_f1_scores):.2f} ± {np.std(cv_f1_scores):.2f}"
    )

    results = {
        'subjects': cv_subjects,
        'tp_scores': cv_tp_scores,
        'fp_scores': cv_fp_scores,
        'fn_scores': cv_fn_scores,
        'precision_scores': cv_precision_scores,
        'recall_scores': cv_recall_scores,
        'f1_scores': cv_f1_scores, 
        'CV_scores': {
            'TP': int(np.sum(cv_tp_scores)),
            'FP': int(np.sum(cv_fp_scores)),
            'FN': int(np.sum(cv_fn_scores)),
            'precision': float(np.mean(cv_precision_scores)),
            'recall': float(np.mean(cv_recall_scores)),
            'f1_score': float(np.mean(cv_f1_scores))
        }
    }
    results_path = os.path.join(output_folder, 'cross_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return None


if __name__ == "__main__":
    main()
