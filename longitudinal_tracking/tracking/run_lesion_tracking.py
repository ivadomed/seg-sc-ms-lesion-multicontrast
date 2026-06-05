"""
This file runs one of the 3 methods for lesion matching between two timepoints.
It runs on all images of the msd dataset provided as input.

Input:
    -i: msd dataset path
    -pred: the pred folder containing the lesion segmentations and other segmentations
    -o: output folder where to store the lesion matching results
    -m: method to use for lesion matching. Choices are:
        - registered_with_CoM
        - registered_with_IoU
        - unregistered
    -w_z_over_disk: weight for the z-axis distance relative to the disk plane when computing lesion matching (default: 25.0)

Output:
    None

Authors: Pierre-Louis Benveniste
"""
import os
import argparse
import json
from tqdm import tqdm
from track_lesion_reg_com import map_lesions_registered_with_CoM
from track_lesion_reg_com_m0_to_m12 import map_lesions_registered_with_CoM_m0_to_m12
from track_lesion_reg_iou import map_lesions_registered_with_IoU
from track_lesion_reg_iou_m0_to_m12 import map_lesions_registered_with_IoU_m0_to_m12
from track_lesion_unreg import map_lesions_unregistered


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-pred', '--pred', type=str, required=True, help='Path to the folder containing the predicted segmentations')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where lesion matching results will be stored')
    parser.add_argument('-m', '--method', type=str, required=True, choices=['registered_with_CoM', 'registered_with_IoU', 'registered_with_CoM_m0_to_m12', 'registered_with_IoU_m0_to_m12', 'unregistered'], help='Method to use for lesion matching')
    parser.add_argument('--w_z_over_disk', type=float, default=25.0, help='Weight for the z-axis distance relative to the disk plane when computing lesion matching (default: 25.0)')
    return parser.parse_args()


def main(input_msd_dataset, pred_folder, output_folder, method, w_z_over_disk):

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    # We run lesion matching now on all the data
    for subject in tqdm(data):
        subject_id = subject
        # Initialize the timepoints and images
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"
        input_image1 = data[subject][timepoint1][0]
        input_image2 = data[subject][timepoint2][0]
        # Build subject output folder
        subject_output_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)
        # Build the path to the subject pred folder
        subject_pred_folder = os.path.join(pred_folder, subject_id)

        if method == 'registered_with_CoM':
            lesion_mapping = map_lesions_registered_with_CoM(input_image1, input_image2, subject_pred_folder, subject_output_folder)
        elif method == 'registered_with_CoM_m0_to_m12':
            lesion_mapping = map_lesions_registered_with_CoM_m0_to_m12(input_image1, input_image2, subject_pred_folder, subject_output_folder)
        elif method == 'registered_with_IoU':
            lesion_mapping = map_lesions_registered_with_IoU(input_image1, input_image2, subject_pred_folder, subject_output_folder)
        elif method == 'registered_with_IoU_m0_to_m12':
            lesion_mapping = map_lesions_registered_with_IoU_m0_to_m12(input_image1, input_image2, subject_pred_folder, subject_output_folder)
        elif method == 'unregistered':
            lesion_mapping = map_lesions_unregistered(input_image1, input_image2, subject_pred_folder, subject_output_folder, w_z_over_disk=w_z_over_disk)
        
        # Save the lesion mapping in a json file
        mapping_output_file = os.path.join(subject_output_folder, 'lesion_mapping.json')
        with open(mapping_output_file, 'w') as f:
            json.dump(lesion_mapping, f, indent=4)

    return None


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    input_msd_dataset = args.input_msd
    output_folder = args.output_folder
    method = args.method
    pred_folder = args.pred
    w_z_over_disk = args.w_z_over_disk

    main(input_msd_dataset, pred_folder, output_folder, method, w_z_over_disk)