"""
In this script, we perform lesion mapping between two timepoints using an XGBoost model to predict lesion correspondences based on features extracted from the lesions.
The specificity of this script is that it uses cross-validation to estimate the performance of the model on the entire dataset.

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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from tqdm import tqdm


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

    # we split the dataset into X (features) and y (labels)
    X = df_pairs.drop(columns=['label'])
    y = df_pairs[['subject','label']]

    # Initialize the cv scores
    cv_tp_scores = []
    cv_fp_scores = []
    cv_fn_scores = []
    cv_precision_scores = []
    cv_recall_scores = []
    cv_f1_scores = []
    cv_subjects = []

    # We perform leave-one-subject-out cross-validation
    for subject in tqdm(subjects):
        X_train = X[X['subject'] != subject].drop(columns=['subject'])
        y_train = y[X['subject'] != subject]['label']
        X_test = X[X['subject'] == subject].drop(columns=['subject'])
        y_test = y[X['subject'] == subject]['label']

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

        # Evaluate the model on the test set only
        test_accuracy = model.score(X_test, y_test)
        logger.info(f"Testing accuracy: {test_accuracy:.4f}")
        
        # Compute TP, FP, FN on the test set
        y_pred = model.predict(X_test)
        logger.info(f"Subject in the test set: {subject}")
        tp = sum((y_test == 1) & (y_pred == 1))
        fp = sum((y_test == 0) & (y_pred == 1))
        fn = sum((y_test == 1) & (y_pred == 0))
        logger.info(f"Test set - TP: {tp}, FP: {fp}, FN: {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        logger.info(f"Test set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        cv_tp_scores.append(tp)
        cv_fp_scores.append(fp)
        cv_fn_scores.append(fn)
        cv_precision_scores.append(precision)
        cv_recall_scores.append(recall)
        cv_f1_scores.append(f1_score)
        cv_subjects.append(subject)

    # Log average scores across folds
    logger.info(f"Cross-Validation - Average Precision: {np.mean(cv_precision_scores):.2f} ± {np.std(cv_precision_scores):.2f}, Average Recall: {np.mean(cv_recall_scores):.2f} ± {np.std(cv_recall_scores):.2f}, Average F1 Score: {np.mean(cv_f1_scores):.2f} ± {np.std(cv_f1_scores):.2f}")

    # We save the results
    results = {
        'subjects': cv_subjects,
        'tp_scores': cv_tp_scores,
        'fp_scores': cv_fp_scores,
        'fn_scores': cv_fn_scores,
        'precision_scores': cv_precision_scores,
        'recall_scores': cv_recall_scores,
        'f1_scores': cv_f1_scores
    }
    results_path = os.path.join(output_folder, 'cross_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    return None


if __name__ == "__main__":

    main()