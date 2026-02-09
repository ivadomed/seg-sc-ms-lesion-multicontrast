"""
This script runs the full experiment.

Input:
    -i: Path to input msd file
    --model-path: Path to the nnUNet model to use for prediction
    --five-folds: Whether to use 5-folds ensemble for prediction (instead of just fold 2 which is the best fold on the validation set)
    -o: Path to output results folder

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from downsample import main_downsample
from run_pred_without_sct import main_run_pred
from thresholding import main_thresholding
from compute_volumes import main_volume
from analyze_lesion_precision import analyze_precision
from evaluate_segs import main_dice_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run full experiment: prediction, thresholding, downsampling.")
    parser.add_argument("-i", "--input-msd", type=str, required=True, help="Path to input msd file")
    parser.add_argument("--path-model", type=str, required=True, help="Path to the nnUNet model to use for prediction")
    parser.add_argument("--five-folds", action="store_true", help="Whether to use 5-folds ensemble for prediction (instead of just fold 2 which is the best fold on the validation set)")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output results folder")
    return parser.parse_args()


def main_full_experiment():
    # Parse arguments
    args = parse_args()
    input_msd = args.input_msd
    output_folder = args.output_folder
    path_model = args.path_model
    five_folds = args.five_folds

    # Build output folder
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Downsampling:
    ## Build downsampling output folder
    downsample_output_folder = os.path.join(output_folder, "downsampled_images")
    os.makedirs(downsample_output_folder, exist_ok=True)
    print("Starting downsampling of images...")
    # main_downsample(input_msd, downsample_output_folder)

    # Step 2: Run predictions
    ## Build prediction output folder
    prediction_output_folder = os.path.join(output_folder, "predictions")
    if five_folds:
        prediction_output_folder = os.path.join(output_folder, "predictions_5_folds")
    os.makedirs(prediction_output_folder, exist_ok=True)
    print("Starting predictions...")
    # main_run_pred(downsample_output_folder, prediction_output_folder, path_model, five_folds=five_folds)

    # Step 3: Thresholding
    ## Build thresholding output folder
    thresholding_output_folder = os.path.join(output_folder, "thresholded_segmentations")
    if five_folds:
        thresholding_output_folder = os.path.join(output_folder, "thresholded_segmentations_5_folds")
    os.makedirs(thresholding_output_folder, exist_ok=True)
    print("Starting thresholding of soft segmentations...")
    # main_thresholding(prediction_output_folder, thresholding_output_folder, high_thresh=True)

    # Step 4: Compute volumes and analyze results
    ## Build volumes output folder
    print("Starting computation of lesion volumes and analysis of precision...")
    volumes_output_folder = os.path.join(output_folder, "volumes")
    if five_folds:
        volumes_output_folder = os.path.join(output_folder, "volumes_5_folds")
    os.makedirs(volumes_output_folder, exist_ok=True)
    ## For each folder in thresholding_output_folder, compute volumes and save to a csv file
    for thresh_folder in os.listdir(thresholding_output_folder):
        thresh_folder_path = os.path.join(thresholding_output_folder, thresh_folder)
        if not os.path.isdir(thresh_folder_path):
            # return error if not a folder
            raise ValueError(f"{thresh_folder_path} is not a folder")
        thresh_volumes_output_folder = os.path.join(volumes_output_folder, thresh_folder)
        os.makedirs(thresh_volumes_output_folder, exist_ok=True)
        # volumes_csv = main_volume(thresh_folder_path, thresh_volumes_output_folder)
        # Analyze the precision 
        volumes_csv = os.path.join(thresh_volumes_output_folder, "lesion_volumes.csv")
        # analyze_precision(volumes_csv, thresh_volumes_output_folder)
        # Compute the dice score:
        main_dice_eval(input_msd, thresh_folder_path, thresh_volumes_output_folder)

    # Step 5: also compute volumes and analyze results for the original soft segmentations (without thresholding)
    original_volumes_output_folder = os.path.join(volumes_output_folder, "no_thresholding")
    os.makedirs(original_volumes_output_folder, exist_ok=True)
    # volumes_csv = main_volume(prediction_output_folder, original_volumes_output_folder)
    # analyze_precision(volumes_csv, original_volumes_output_folder)
    # Compute the dice score
    main_dice_eval(input_msd, prediction_output_folder, original_volumes_output_folder)

    return None


if __name__ == "__main__":
    main_full_experiment()