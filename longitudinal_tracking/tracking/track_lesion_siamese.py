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
from keras import layers, models, regularizers, Input
from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras import regularizers
from keras import metrics
import keras
from tqdm.keras import TqdmCallback
import tensorflow as tf
import tensorflow.keras.backend as K
keras.utils.set_random_seed(812)
from keras import ops
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import json


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


def load_data():

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
    lesions_1 = []
    lesions_2 = []
    lesion_pairs_labels = []
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
                lesion1_features = {
                    'subject': subject,
                }
                lesion2_features = {
                    'subject': subject,
                }
                lesion_pair = {
                    'subject': subject,
                }
                # Add features from lesion1 and lesion2
                for col in df.columns:
                    if col not in ['subject', 'timepoint', 'group']:
                        lesion1_features[f'{col}'] = lesion1[col]
                        lesion2_features[f'{col}'] = lesion2[col]
                
                # Add label if available (1 if they are the same lesion, else 0)
                lesion_pair['label'] = 1 if lesion1['group'] == lesion2['group'] else 0
                lesion_pairs_labels.append(lesion_pair)
                lesions_1.append(lesion1_features)
                lesions_2.append(lesion2_features)

    # Create a DataFrame from lesion pairs
    df_pairs = pd.DataFrame(lesion_pairs_labels)
    lesion_1_df = pd.DataFrame(lesions_1)
    lesion_2_df = pd.DataFrame(lesions_2)

    # Split the dataset into training and testing sets (done on subject level to avoid data leakage)
    subjects = df_pairs['subject'].unique()
    ## We arbitrarily choose that subjects from the Toronto site are used for testing
    test_subjects = [s for s in subjects if 'tor' in s]
    # val_subjects = [s for s in subjects if 'cal' in s or 'mon' in s]
    train_subjects = [s for s in subjects if s not in test_subjects]

    # I manually set to false index 684 in lesion_pairs (because of this: https://github.com/ivadomed/ms-lesion-agnostic/issues/98)
    df_pairs.at[684, 'label'] = 0

    train_lesion_1_df = lesion_1_df[lesion_1_df['subject'].isin(train_subjects)]
    train_lesion_2_df = lesion_2_df[lesion_2_df['subject'].isin(train_subjects)]
    train_df_pairs = df_pairs[df_pairs['subject'].isin(train_subjects)]

    # val_lesion_1_df = lesion_1_df[lesion_1_df['subject'].isin(val_subjects)]
    # val_lesion_2_df = lesion_2_df[lesion_2_df['subject'].isin(val_subjects)]
    # val_df_pairs = df_pairs[df_pairs['subject'].isin(val_subjects)]

    test_lesion_1_df = lesion_1_df[lesion_1_df['subject'].isin(test_subjects)]
    test_lesion_2_df = lesion_2_df[lesion_2_df['subject'].isin(test_subjects)]
    test_df_pairs = df_pairs[df_pairs['subject'].isin(test_subjects)]

    logger.info(f"Total subjects: {len(subjects)}, Training subjects: {len(train_subjects)}, Testing subjects: {len(test_subjects)}")

    # Split the dataset into features and labels
    feature_cols = [col for col in train_lesion_1_df.columns if col not in ['subject', 'label']]
    X_train_1 = train_lesion_1_df[feature_cols]
    X_train_2 = train_lesion_2_df[feature_cols]
    y_train = train_df_pairs['label']
    # X_val_1 = val_lesion_1_df[feature_cols]
    # X_val_2 = val_lesion_2_df[feature_cols]
    # y_val = val_df_pairs['label']
    X_test_1 = test_lesion_1_df[feature_cols]
    X_test_2 = test_lesion_2_df[feature_cols]
    y_test = test_df_pairs['label']
    logger.info(f"Training set size: {X_train_1.shape[0]} pairs")
    # logger.info(f"Validation set size: {X_val_1.shape[0]} pairs")
    logger.info(f"Testing set size: {X_test_1.shape[0]} pairs")

    # Extract mean and std for each feature in the training set
    means = X_train_1.mean()
    stds = X_train_1.std()
    logger.info("Feature means and stds computed from training set.")
    logger.debug(f"Feature means: {np.mean(means)}")
    logger.debug(f"Feature stds: {np.mean(stds)}")

    # Initialize Scaler
    scaler = StandardScaler()

    # FIT on Training data only, then Transform everything
    X_train_1 = scaler.fit_transform(X_train_1)
    X_train_2 = scaler.transform(X_train_2) # Use same scaler!

    # X_val_1   = scaler.transform(X_val_1)
    # X_val_2   = scaler.transform(X_val_2)

    X_test_1  = scaler.transform(X_test_1)
    X_test_2  = scaler.transform(X_test_2)

    logger.info("Data normalized using StandardScaler.")

    return X_train_1, X_train_2, y_train, _, _, _, X_test_1, X_test_2, y_test, output_folder, logger, means, stds


def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_in(inputs, targets):
        # Binary Cross-Entropy loss calculation
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Convert BCE loss to probability
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss  # Apply focal adjustment
        return focal_loss.mean()
    return focal_loss_in


def contrastive_loss_margin(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = ops.square(y_pred)
        margin_square = ops.square(ops.maximum(margin - (y_pred), 0))
        return ops.mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss


def build_siamese_network(num_features_per_lesion):
        
    # Define the encoder of the network
    input_enc = Input(shape=(num_features_per_lesion,))
    x = layers.Dense(128, activation='relu')(input_enc)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    embedding = layers.Dense(32, activation='linear')(x)
    
    encoder = models.Model(inputs=input_enc, outputs=embedding)
    
    # Define Inputs
    input_a = Input(shape=(num_features_per_lesion,))
    input_b = Input(shape=(num_features_per_lesion,))
    
    # Process both through the same encoder
    processed_a = encoder(input_a)
    processed_b = encoder(input_b)
    
    # # Compute Distance Layer (L1 Distance)
    # L1_layer = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
    # L1_distance = L1_layer([processed_a, processed_b])
    # Let's try L2 distance instead
    L2_layer = layers.Lambda(lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))
    L2_distance = L2_layer([processed_a, processed_b])
    
    # Classification on top of the distance
    prediction = layers.Dense(1, activation='sigmoid')(L2_distance)
    # x = layers.Dense(16, activation='relu')(L2_distance)
    # prediction = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=[input_a, input_b], outputs=prediction)
    return model


def train(X_train_1, X_train_2, y_train, X_val_1, X_val_2, y_val, X_test_1, X_test_2, y_test, output_folder, logger):

    model = build_siamese_network(X_train_1.shape[1])
    optimizer = optimizers.RMSprop(learning_rate=0.001)
    # optimizer = optimizers.AdamW(learning_rate=0.001)

    
    # model.compile(optimizer='rmsprop' ,loss='binary_crossentropy' ,metrics=['AUC'])
    # model.compile(optimizer='rmsprop', loss='BinaryFocalCrossentropy', metrics=['AUC'])
    model.compile(optimizer=optimizer, loss=contrastive_loss_margin(margin=1), metrics=['AUC'])
    
    # fit the model
    # Calculate weights
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {0: weights[0], 1: weights[1]}
    logger.info(f"Class weights: {class_weights_dict}")
    model.fit([X_train_1, X_train_2], y_train, epochs=500, batch_size=20, class_weight=class_weights_dict, callbacks=[TqdmCallback(verbose=0)], verbose=0)
    # model.fit([X_train_1, X_train_2], y_train, epochs=500, batch_size=20, validation_data=([X_val_1, X_val_2], y_val), callbacks=[TqdmCallback(verbose=0)], verbose=0)

    # # Compute evaluation metrics on the validation set
    # val_results = model.predict([X_val_1, X_val_2], verbose=0)
    # val_pred_labels = (val_results > 0.5).astype(int).flatten()
    # tp = sum((y_val == 1) & (val_pred_labels == 1))
    # fp = sum((y_val == 0) & (val_pred_labels == 1))
    # fn = sum((y_val == 1) & (val_pred_labels == 0))
    # logger.info(f"Validation set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # logger.info(f"Validation set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    # Compute TP, FP, FN on the test set
    y_pred = model.predict([X_test_1, X_test_2])
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    tp = sum((y_test == 1) & (y_pred_labels == 1))
    fp = sum((y_test == 0) & (y_pred_labels == 1))
    fn = sum((y_test == 1) & (y_pred_labels == 0))
    logger.info(f"Test set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    logger.info(f"Test set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    # Compute TP, FP, FN on the training set
    y_train_pred = model.predict([X_train_1, X_train_2])
    y_train_pred_labels = (y_train_pred > 0.5).astype(int).flatten()
    tp = sum((y_train == 1) & (y_train_pred_labels == 1))
    fp = sum((y_train == 0) & (y_train_pred_labels == 1))
    fn = sum((y_train == 1) & (y_train_pred_labels == 0))
    logger.info(f"Training set - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    logger.info(f"Training set - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    return model
    

if __name__ == "__main__":
    # Load the traing and testing data
    X_train_1, X_train_2, y_train, X_val_1, X_val_2, y_val, X_test_1, X_test_2, y_test, output_folder, logger, means, stds = load_data()
    # Train the model and evaluate it
    model = train(X_train_1, X_train_2, y_train, X_val_1, X_val_2, y_val, X_test_1, X_test_2, y_test, output_folder, logger)
    # Save the model
    model_path = os.path.join(output_folder, 'lesion_mapping_siamese_model.keras')
    model.save(model_path)
    logger.info(f"Model saved at {model_path}")