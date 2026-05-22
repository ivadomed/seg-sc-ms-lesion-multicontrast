"""
This code builds a dataset for training a model to predict lesion mapping between two timepoints.

Input:
    -i: msd dataset path
    -pred: path to the folder containing the predicted segmentations (SC, lesions, centerline, levels ...)
    -gt_mappings: path to the folder containing the ground truth lesion mappings
    -o: output folder where to store the lesion matching results
    --for-gt-lesion-seg: if specified, the labeled lesion seg come from the gt_mapping folder instead of the pred_folder

Output:
    None

Authors: Pierre-Louis Benveniste
"""
import os
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import date
from tqdm import tqdm
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions, get_centerline, get_levels, label_lesion_seg
from tracking.track_lesion_unreg import label_centerline, analyze_lesions, compute_lesion_location

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-pred', '--pred', type=str, required=True, help='Path to the folder containing the predicted segmentations')
    parser.add_argument('-gt-mapping', '--gt-mapping', type=str, required=True, help='Path to the folder containing the ground truth lesion mappings')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where lesion matching results will be stored')
    parser.add_argument('--for-gt-lesion-seg', action='store_true', help='If specified, the labeled lesion seg come from the gt_mapping folder instead of the pred_folder')
    return parser.parse_args()


def group_lesions(lesion_analysis_1, lesion_analysis_2, lesion_mapping_path):
    """
    This function creates group for lesions from the lesion mapping.
    If lesion mapping is:
         B1 -> F1,F2
         B2 -> F2
         B3 -> F3
         B4 -> F3
         B5 -> None
    Then groups are:
        G1: B1, B2, F1, F2
        G2: B3, B4, F3
        G3: B5

    Inputs:
        lesion_analysis_1 : dictionary containing lesion analysis for timepoint1
        lesion_analysis_2 : dictionary containing lesion analysis for timepoint2
        lesion_mapping_path : path to the lesion mapping json file
    
    Outputs:
        grouped_lesions_1 : updated lesion_analysis_1 with group information
        grouped_lesions_2 : updated lesion_analysis_2 with group information
    """
    # Load the lesion mapping
    with open(lesion_mapping_path, 'r') as f:
        lesion_mapping = json.load(f)

    print("Lesion Mapping Loaded:")
    for lesion_1, mapped_lesions_2 in lesion_mapping.items():
        print(f"  Lesion {lesion_1} -> Mapped Lesions {mapped_lesions_2}")

    # Initialize the group ID
    group_id = 0
    
    # Initialize a list of lesions at baseline and follow-up
    lesions_baselines = set(lesion_analysis_1.keys())
    lesions_followups = set(lesion_analysis_2.keys())
    # Iterate through the lesion mapping to create groups
    for lesion_1 in list(lesions_baselines):
        if 'group' in lesion_analysis_1[lesion_1]:
            # Already assigned to a group
            continue
        group_id += 1
        # The lesion is assigned to a group
        lesion_analysis_1[lesion_1]['group'] = group_id
        # We check all mapped lesions too
        for lesion_2 in lesion_mapping[lesion_1]: 
            lesion_analysis_2[str(lesion_2)]['group'] = group_id
        # Now we check if all other lesions in baseline have any mapped lesions in common with the current lesion_1
        for other_lesion_1 in list(lesions_baselines):
            if other_lesion_1 == lesion_1:
                continue
            if 'group' in lesion_analysis_1[other_lesion_1]:
                # Already assigned to a group
                continue
            other_mapped_lesions_2 = lesion_mapping[other_lesion_1]
            # If there is an intersection between other_mapped_lesions_2 and mapped_lesions_2, we add other_lesion_1 to the same group
            if set(other_mapped_lesions_2).intersection(set(lesion_mapping[lesion_1])):
                lesion_analysis_1[other_lesion_1]['group'] = group_id
                # And all mapped lesions too
                for lesion_2 in other_mapped_lesions_2:
                    lesion_analysis_2[str(lesion_2)]['group'] = group_id
    
    # For lesions of follow-up that were not mapped to any baseline lesion, we create new groups
    for lesion_2 in lesions_followups:
        # If already assigned to a group, we skip
        if 'group' in lesion_analysis_2[lesion_2]:
            continue
        group_id += 1
        lesion_analysis_2[lesion_2]['group'] = group_id

    return lesion_analysis_1, lesion_analysis_2


def build_dataset(input_msd_dataset, pred_folder, gt_mappings, output_folder, for_gt_lesion_seg=False):
    """
    This function is used to build the dataset for training the model and lesion mapping.
    For each lesion in the baseline scan and follow-up scan, we add them to the dataset alongside their coordinates and volume.
    Input:
        - input_msd_dataset: Path to the MSD dataset folder.
        - output_folder: Path to the folder where the dataset will be saved.

    Output:
        None
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Add a logger
    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']

    dataset = []

    # Loop through each subject
    for subject in tqdm(data):

        # Build subject pred folder
        subject_pred_folder = os.path.join(pred_folder, subject)

        # Build path to lesion mapping file
        if for_gt_lesion_seg:
            lesion_mapping_path = os.path.join(gt_mappings, subject, "lesion-mapping_M0-M12.json")
        else:
            lesion_mapping_path = os.path.join(gt_mappings, subject, "lesion_mapping.json")

        # Build subject output folder
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)
        # Build a temp folder for intermediate results
        temp_folder = os.path.join(subject_output_folder, 'temp')
        os.makedirs(temp_folder, exist_ok=True)
        # Build a qc folder
        qc_folder = os.path.join(temp_folder, 'qc')
        os.makedirs(qc_folder, exist_ok=True)

        # Define timepoints
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"

        ## We extract image name
        image1 = data[subject][timepoint1][0]
        image2 = data[subject][timepoint2][0]

        # Initialize file names for lesions, sc and disc levels at both timepoints
        image_1_name = Path(image1).name
        image_2_name = Path(image2).name
        # Initialize the names
        sc_seg_1 = os.path.join(subject_pred_folder, image_1_name.replace('.nii.gz', '_sc-seg.nii.gz'))
        sc_seg_2 = os.path.join(subject_pred_folder, image_2_name.replace('.nii.gz', '_sc-seg.nii.gz'))
        centerline_1 = os.path.join(subject_pred_folder, image_1_name.replace('.nii.gz', '_centerline.nii.gz'))
        centerline_2 = os.path.join(subject_pred_folder, image_2_name.replace('.nii.gz', '_centerline.nii.gz'))
        levels_1 = os.path.join(subject_pred_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
        levels_2 = os.path.join(subject_pred_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))
        # The lesion segmentations are different for pred or manual lesion segs
        if for_gt_lesion_seg:
            labeled_lesion_seg_1 = os.path.join(gt_mappings, subject, "ses-M0", "anat",  image_1_name.replace('.nii.gz', '_lesion-manual-labeled.nii.gz'))
            labeled_lesion_seg_2 = os.path.join(gt_mappings, subject, "ses-M12", "anat", image_2_name.replace('.nii.gz', '_lesion-manual-labeled.nii.gz'))
        else:
            labeled_lesion_seg_1 = os.path.join(subject_pred_folder, image_1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
            labeled_lesion_seg_2 = os.path.join(subject_pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))

        # We label the centerline
        labeled_centerline_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_labeled-centerline.nii.gz'))
        labeled_centerline_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_labeled-centerline.nii.gz'))
        label_centerline(centerline_1, levels_1, labeled_centerline_1)
        label_centerline(centerline_2, levels_2, labeled_centerline_2)

        # For each lesion in the labeled mask at timepoint1, we get its coordinates and volume
        lesion_analysis_1 = analyze_lesions(labeled_lesion_seg_1)
        lesion_analysis_1 = compute_lesion_location(lesion_analysis_1, labeled_lesion_seg_1, sc_seg_1, labeled_centerline_1, levels_1)
        lesion_analysis_2 = analyze_lesions(labeled_lesion_seg_2)
        lesion_analysis_2 = compute_lesion_location(lesion_analysis_2, labeled_lesion_seg_2, sc_seg_2, labeled_centerline_2, levels_2)

        # Now we group the lesions based on the lesion mapping
        lesion_analysis_1, lesion_analysis_2  = group_lesions(lesion_analysis_1, lesion_analysis_2, lesion_mapping_path)

        logger.info(f"Subject {subject} - Grouped Lesions Timepoint {timepoint1}:")
        for lesion_id, lesion_info in lesion_analysis_1.items():
            logger.info(f"  Lesion {lesion_id}: {lesion_info}")
        logger.info("")
        logger.info(f"Subject {subject} - Grouped Lesions Timepoint {timepoint2}:")
        for lesion_id, lesion_info in lesion_analysis_2.items():
            logger.info(f"  Lesion {lesion_id}: {lesion_info}")
        
        
        # Now we add the lesions to the dataset
        for lesion_id, lesion_info in lesion_analysis_1.items():
            dataset.append({
                'subject': subject,
                'timepoint': timepoint1,
                'group': lesion_info['group'],
                'lesion_id': lesion_id,
                'z': lesion_info['centerline_z'],
                'r': lesion_info['radius_mm'],
                'theta': lesion_info['theta'],
                'volume': lesion_info['volume_mm3'],
                'diameter_RL': lesion_info['diameter_RL_mm'],
                'diameter_AP': lesion_info['diameter_AP_mm'],
                'diameter_SI': lesion_info['diameter_SI_mm']
            })
    
        for lesion_id, lesion_info in lesion_analysis_2.items():
            dataset.append({
                'subject': subject,
                'timepoint': timepoint2,
                'group': lesion_info['group'],
                'lesion_id': lesion_id,
                'z': lesion_info['centerline_z'],
                'r': lesion_info['radius_mm'],
                'theta': lesion_info['theta'],
                'volume': lesion_info['volume_mm3'],
                'diameter_RL': lesion_info['diameter_RL_mm'],
                'diameter_AP': lesion_info['diameter_AP_mm'],
                'diameter_SI': lesion_info['diameter_SI_mm']
            })
        
    # Convert to pandas dataframe
    dataset = pd.DataFrame(dataset)
    # Save the dataset
    if for_gt_lesion_seg:
        dataset_path = os.path.join(output_folder, 'lesion_dataset_gt_lesion.csv')
    else:   
        dataset_path = os.path.join(output_folder, 'lesion_dataset_pred_lesion.csv')
    dataset.to_csv(dataset_path, index=False)
    logger.info(f"Dataset saved to {dataset_path}")
        
    return None


if __name__ == "__main__":  
    args = parse_args()
    input_msd_dataset = args.input_msd
    pred_folder = args.pred
    gt_mappings = args.gt_mapping
    output_folder = args.output_folder
    for_gt_lesion_seg = args.for_gt_lesion_seg

    # Build the dataset
    build_dataset(input_msd_dataset, pred_folder, gt_mappings, output_folder, for_gt_lesion_seg)