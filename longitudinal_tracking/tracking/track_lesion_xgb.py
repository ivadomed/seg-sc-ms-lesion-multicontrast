"""
In this script, we perform lesion mapping between two timepoints using an XGBoost model to predict lesion correspondences based on features extracted from the lesions.

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
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import numpy as np
import json
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to the csv dataset containing lesion features over time')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where model and results will be stored')
    return parser.parse_args()


def distance_cylindrical(coord1, coord2, w_z=1.0, w_disk=1.0):
    """
    Computes a weighted Euclidean distance between two points in cylindrical coordinates.
    
    Parameters:
        coord1 (dict): Coordinates of the first point with keys 'r', 'theta', and 'z'.
        coord2 (dict): Coordinates of the second point with keys 'r', 'theta', and 'z'.
        w_z (float): Weight for the z-axis distance.
        w_disk (float): Weight for the distance in the disk plane (r, theta).
    
    Returns:
        float: The weighted Euclidean distance.
    """
    distance_disk = np.sqrt(coord1['r']**2 + coord2['r']**2 - 2 * coord1['r'] * coord2['r'] * np.cos(np.radians(coord1['theta'] - coord2['theta'])))
    z_dist = coord1['z'] - coord2['z']
    return np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)


def main():
    args = parse_args()
    dataset_csv = args.data
    output_folder = args.output_folder

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a logger
    logger.add(os.path.join(output_folder, 'lesion_mapping_XGB.log'))

    # Load dataset
    df = pd.read_csv(dataset_csv)

    # We transform the df to create one row per lesion pair in timepoint1 and timepoint2 (pairing only within the same subject)
    ## Columns are : subject,timepoint,group,z,r,theta,volume
    lesion_pairs = []
    subjects = df['subject'].unique()

    for subject in subjects:
        df_subject = df[df['subject'] == subject]
        timepoints = df_subject['timepoint'].unique()
        if len(timepoints) != 2:
            continue  # We only process subjects with exactly 2 timepoints
        tp1, tp2 = timepoints

        df_tp1 = df_subject[df_subject['timepoint'] == tp1]
        df_tp2 = df_subject[df_subject['timepoint'] == tp2]

        for _, lesion1 in df_tp1.iterrows():
            for _, lesion2 in df_tp2.iterrows():
                pair_features = {
                    'subject': subject,
                }
                # Add features from lesion1 and lesion2
                for col in df.columns:
                    if col not in ['subject', 'timepoint', 'group']:
                        pair_features[f'{col}1'] = lesion1[col]
                        pair_features[f'{col}2'] = lesion2[col]
                # Calculate explicit differences
                dz = lesion1['z'] - lesion2['z']
                dr = lesion1['r'] - lesion2['r']
                dtheta = lesion1['theta'] - lesion2['theta']
                d_vol = abs(lesion1['volume'] - lesion2['volume'])
                dist = distance_cylindrical({'r': lesion1['r'], 'theta': lesion1['theta'], 'z': lesion1['z']},
                                            {'r': lesion2['r'], 'theta': lesion2['theta'], 'z': lesion2['z']})
                # Add these to pair_features
                pair_features['dz'] = dz
                pair_features['dr'] = dr
                pair_features['dtheta'] = dtheta
                pair_features['dvol'] = d_vol
                pair_features['dist'] = dist
                # Add label if available (1 if they are the same lesion, else 0)
                pair_features['label'] = 1 if lesion1['group'] == lesion2['group'] else 0
                lesion_pairs.append(pair_features)

    # Create a DataFrame from lesion pairs
    df_pairs = pd.DataFrame(lesion_pairs)

    # I manually set to false index 684 in lesion_pairs (because of this: https://github.com/ivadomed/ms-lesion-agnostic/issues/98)
    df_pairs.at[684, 'label'] = 0

    # Split the dataset into training and testing sets (done on subject level to avoid data leakage)
    subjects = df_pairs['subject'].unique()
    ## We arbitrarily choose that subjects from the Toronto site are used for testing
    test_subjects = [s for s in subjects if 'tor' in s]
    train_subjects = [s for s in subjects if s not in test_subjects]
    train_df = df_pairs[df_pairs['subject'].isin(train_subjects)]
    test_df = df_pairs[df_pairs['subject'].isin(test_subjects)]
    logger.info(f"Total subjects: {len(subjects)}, Training subjects: {len(train_subjects)}, Testing subjects: {len(test_subjects)}")

    # Split the dataset into features and labels
    feature_cols = [col for col in df_pairs.columns if col not in ['subject', 'label']]
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']
    logger.info(f"Training set size: {X_train.shape[0]} pairs")
    logger.info(f"Testing set size: {X_test.shape[0]} pairs")

    # Train an XGBoost model
    search_spaces = {
        'max_depth': Integer(3, 10), # Lower values prevent overfitting
        'min_child_weight': Integer(1, 10), # Higher values prevent overfitting # Suggested to go as high as 5 by LeChat
        'subsample': Real(0.5, 1), # Lower values prevent overfitting
        'colsample_bytree': Real(0.001, 1), # Lower values prevent overfitting # Suggested to go as low as 0.5 by LeChat
        'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
        'scale_pos_weight': Integer(5, 10, prior='log-uniform')  # To handle class imbalance
    }
    # We define the model
    model = XGBClassifier(seed=42, eval_metric='logloss')  # We set scale_pos_weight to the ratio of negative to positive samples
    # We define the search
    search = BayesSearchCV(model, search_spaces, n_iter=50, n_jobs=1, cv=3, random_state=42, scoring='average_precision', verbose=1)
    # We fit the search
    search.fit(X_train, y_train)
    model = search.best_estimator_
    # print best hyperparameters
    logger.info(f"Best hyperparameters: {search.best_params_}")

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    logger.info(f"Training accuracy: {train_accuracy:.4f}")
    logger.info(f"Testing accuracy: {test_accuracy:.4f}")
    logger.info("--------------------------------------------------")

    # Compute TP, FP, FN on the training set
    y_train_pred = model.predict(X_train)
    tp = sum((y_train == 1) & (y_train_pred == 1))
    fp = sum((y_train == 0) & (y_train_pred == 1))
    fn = sum((y_train == 1) & (y_train_pred == 0))
    logger.info(f"Training set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    logger.info(f"Training set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
    logger.info("--------------------------------------------------")
    # Compute TP, FP, FN on the test set
    y_pred = model.predict(X_test)
    tp = sum((y_test == 1) & (y_pred == 1))
    fp = sum((y_test == 0) & (y_pred == 1))
    fn = sum((y_test == 1) & (y_pred == 0))
    logger.info(f"Test set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    logger.info(f"Test set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    # We want to build a per-subject results dictionary
    results = {}
    for subject in subjects:
        results[subject] = {'TP': [], 'FP': [], 'FN': []}
        if subject not in test_subjects:
            subject_df = train_df[train_df['subject'] == subject]
        else:
            subject_df = test_df[test_df['subject'] == subject]
        if subject_df.empty:
            continue
        X_subj = subject_df[feature_cols]
        y_subj = subject_df['label']
        y_subj_pred = model.predict(X_subj)
        for true_label, pred_label in zip(y_subj, y_subj_pred):
            if true_label == 1 and pred_label == 1:
                results[subject]['TP'].append(1)
                results[subject]['FP'].append(0)
                results[subject]['FN'].append(0)
            elif true_label == 0 and pred_label == 1:
                results[subject]['TP'].append(0)
                results[subject]['FP'].append(1)
                results[subject]['FN'].append(0)
            elif true_label == 1 and pred_label == 0:
                results[subject]['TP'].append(0)
                results[subject]['FP'].append(0)
                results[subject]['FN'].append(1)
            else:
                results[subject]['TP'].append(0)
                results[subject]['FP'].append(0)
                results[subject]['FN'].append(0)
        # Sum up the counts
        results[subject]['TP'] = sum(results[subject]['TP'])
        results[subject]['FP'] = sum(results[subject]['FP'])
        results[subject]['FN'] = sum(results[subject]['FN'])

    # Save the results file
    output_results_path = os.path.join(output_folder, 'lesion_mapping_XGB_results.json')
    with open(output_results_path, 'w') as f:
        json.dump(results, f, indent=4)

    # save the model
    output_model_path = os.path.join(output_folder, 'lesion_mapping_XGB_model.pkl')
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)

    return None


if __name__ == "__main__":

    main()