"""
This file is used to evaluate the predictions of the model on the test set.

Input:
    -pred-folder: Folder containing the predictions of the model on the test set
    -label-folder: Folder containing the images of the test set
    -image-folder: Folder containing the images of the test set
    -conversion-dict: Dictionary containing the conversion of the predictions to the original labels
    -output-folder: Folder to save the evaluation results

Output:
    None

Example: 
    python evaluate_predictions.py -pred-folder /path/to/predictions -image-folder /path/to/images -conversion-dict /path/to/dict -output-folder /path/to/output


Author: Pierre-Louis Benveniste
"""

import os
import numpy as np
import argparse
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm
from utils import dice_score, lesion_ppv, lesion_f1_score, lesion_sensitivity
from panoptica import InputType, Panoptica_Evaluator
from panoptica.metrics import Metric
from panoptica.instance_matcher import NaiveThresholdMatching  # or IoUMatching


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred-folder", required=True, type=str, help="Folder containing the predictions of the model on the test set")
    parser.add_argument("-label-folder", required=True, type=str, help="Folder containing the images of the test set")
    parser.add_argument("-image-folder", required=True, type=str, help="Folder containing the images of the test set")
    parser.add_argument("-conversion-dict", required=True, type=str, help="Dictionary containing the conversion of the predictions to the original labels")
    parser.add_argument("-output-folder", required=True, type=str, help="Folder to save the evaluation results")
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    pred_folder = args.pred_folder
    label_folder = args.label_folder
    image_folder = args.image_folder
    conversion_dict = args.conversion_dict
    output_folder = args.output_folder

    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all the predictions (with rglob)
    predictions = list(Path(pred_folder).rglob("*.nii.gz"))

    # Open the conversion dictionary (its a json file)
    with open(conversion_dict, "r") as f:
        conversion_dict = json.load(f)

    # Dict of dice score
    dice_scores = {}
    ppv_scores = {}
    f1_scores = {}
    sensitivity_scores = {}

    # Initialize the panoptica evaluator
    evaluator = Panoptica_Evaluator(
        expected_input = InputType.UNMATCHED_INSTANCE,
        instance_matcher = NaiveThresholdMatching(
            matching_metric = Metric.IOU,
            matching_threshold = 0.1,
            allow_many_to_one = True)
    )

    results_panoptica = {}

    # Iterate over the results
    for pred in tqdm(predictions):
        # Get the corresponding image
        label = os.path.join(label_folder, pred.name)
        image = os.path.join(image_folder, pred.name).replace(".nii.gz", "_0000.nii.gz")

        # Load the predictions and the label
        pred_data = nib.load(str(pred)).get_fdata()
        label_data = nib.load(str(label)).get_fdata()

        # Compute dice score
        dice = dice_score(pred_data, label_data)
        ppv = lesion_ppv(label_data, pred_data)
        f1 = lesion_f1_score(label_data, pred_data)
        sensitivity = lesion_sensitivity(label_data, pred_data)

        # Get initial image name from conversion dict
        image_name = None
        for original_image in conversion_dict:
            
            if conversion_dict[original_image] == image:
                image_name = original_image
                break

        # Evaluate the predictions with panoptica
        eval_result = evaluator.evaluate(str(pred), str(label))
        # Key varies by panoptica version; take the first available value
        result_panoptica = next(iter(eval_result.values()))

        metric_names = [
            "n_ref_instances", "n_pred_instances", "tp", "fp", "fn", "rq",
            "global_bin_dsc",
            "sq", "pq", "sq_dsc", "pq_dsc", "sq_assd", "sq_rvd",
        ]
        results_panoptica[image_name] = {
            m: float(getattr(result_panoptica, m, np.nan))
            for m in metric_names
        }

        # Save the dice score
        dice_scores[image_name] = dice
        ppv_scores[image_name] = ppv
        f1_scores[image_name] = f1
        sensitivity_scores[image_name] = sensitivity

    with open(os.path.join(output_folder, "results_panoptica.json"), "w") as f:
        json.dump(results_panoptica, f, indent=2)

    # Save the results
    with open(os.path.join(output_folder, "dice_scores.txt"), "w") as f:
        for key, value in dice_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "ppv_scores.txt"), "w") as f:
        for key, value in ppv_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "f1_scores.txt"), "w") as f:
        for key, value in f1_scores.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(output_folder, "sensitivity_scores.txt"), "w") as f:
        for key, value in sensitivity_scores.items():
            f.write(f"{key}: {value}\n")

    return None


if __name__ == "__main__":
    main()