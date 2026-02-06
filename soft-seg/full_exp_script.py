"""
This script runs the full experiment.

Input:
    -i: Path to input msd file
    --model-path: Path to the nnUNet model to use for prediction
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run full experiment: prediction, thresholding, downsampling.")
    parser.add_argument("-i", "--input-msd", type=str, required=True, help="Path to input msd file")
    parser.add_argument("--path-model", type=str, required=True, help="Path to the nnUNet model to use for prediction")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output results folder")
    return parser.parse_args()


def main_full_experiment():
    # Parse arguments
    args = parse_args()
    input_msd = args.input_msd
    output_folder = args.output_folder
    path_model = args.path_model

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
    os.makedirs(prediction_output_folder, exist_ok=True)
    print("Starting predictions...")
    main_run_pred(downsample_output_folder, prediction_output_folder, path_model)

    # Step 3: Thresholding
    ## Build thresholding output folder
    thresholding_output_folder = os.path.join(output_folder, "thresholded_segmentations")
    os.makedirs(thresholding_output_folder, exist_ok=True)
    print("Starting thresholding of soft segmentations...")
    main_thresholding(prediction_output_folder, thresholding_output_folder, high_thresh=True)

    # Step 4: Compute volumes and analyze results
    ## Build volumes output folder
    volumes_output_folder = os.path.join(output_folder, "volumes")
    os.makedirs(volumes_output_folder, exist_ok=True)
    ## For each folder in thresholding_output_folder, compute volumes and save to a csv file
    for thresh_folder in os.listdir(thresholding_output_folder):
        thresh_folder_path = os.path.join(thresholding_output_folder, thresh_folder)
        if not os.path.isdir(thresh_folder_path):
            # return error if not a folder
            raise ValueError(f"{thresh_folder_path} is not a folder")
        thresh_volumes_output_folder = os.path.join(volumes_output_folder, thresh_folder)
        os.makedirs(thresh_volumes_output_folder, exist_ok=True)
        volumes_csv = main_volume(thresh_folder_path, thresh_volumes_output_folder)
        # Analyze the precision 
        analyze_precision(volumes_csv, thresh_volumes_output_folder)


if __name__ == "__main__":
    main_full_experiment()