"""
This file creates the MSD-style JSON datalist.

Arguments:
    -pd, --path-data: Path to the data set directory
    -po, --path-out: Path to the output directory where dataset json is saved
    --canproco-exclude: Path to the file containing the list of subjects to exclude from CanProCo

Pierre-Louis Benveniste
"""
import os
import json
from tqdm import tqdm
import yaml
import argparse
from datetime import date
from pathlib import Path


def get_parser():
    """
    Get parser for script create_msd_data.py

    Input:
        None

    Returns:
        parser : argparse object
    """

    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for lesion-agnostic nnunet model training.')

    parser.add_argument('-pd', '--path-data', required=True, type=str, help='Path to the folder containing the datasets')
    parser.add_argument('-po', '--path-out', type=str, help='Path to the output directory where dataset json is saved')
    parser.add_argument('--canproco-exclude', type=str, help='Path to the file containing the list of subjects to exclude from CanProCo')
    return parser


def print_dataset_contrasts_distribution(derivatives, dataset_name):
    """
    This function takes a list of derivatives and prints the distribution of contrasts in the dataset.
    Input:
        derivatives : list : List of derivatives
    Returns:
        None
    """
    # Get the contrasts
    contrasts = []
    for derivative in derivatives:
        if 'basel-mp2rage' in str(derivative):
            contrast = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
            contrasts.append(contrast)
        else:
            contrast = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
            contrasts.append(contrast)        
        
    # Get the unique contrasts
    unique_contrasts = set(contrasts)
    # Print the distribution
    print(f"Distribution of contrasts in {dataset_name}:")
    for contrast in unique_contrasts:
        count = contrasts.count(contrast)
        print(f"{contrast}: {count}")
    return contrasts


def main():
    """
    This is the main function of the script.

    Input:
        None
    
    Returns:
        None
    """
    # Get the arguments
    parser = get_parser()
    args = parser.parse_args()
    root = args.path_data

    # Get all subjects
    path_basel_mp2rage = Path(os.path.join(root, "basel-mp2rage"))
    path_canproco = Path(os.path.join(root, "canproco"))
    path_nyu = Path(os.path.join(root, "ms-nyu"))

    derivatives_basel_mp2rage = list(path_basel_mp2rage.rglob('*_desc-rater3_label-lesion_seg.nii.gz'))
    derivatives_basel_mp2rage = [derivative for derivative in derivatives_basel_mp2rage if 'labels-ms-spinal' in str(derivative)]
    # We only want 100 subjects from Basel MP2RAGE, so we randomly sample 100 derivatives from the list
    if len(derivatives_basel_mp2rage) > 100:
        derivatives_basel_mp2rage = derivatives_basel_mp2rage[:100]
    
    ## We only want to keep Tor and Cal
    derivatives_canproco = list(path_canproco.rglob('*_lesion-manual.nii.gz'))
    derivatives_canproco = [derivative for derivative in derivatives_canproco if 'labels-ms-spinal' in str(derivative)]
    derivatives_canproco = [derivative for derivative in derivatives_canproco if ('sub-tor' in str(derivative) or 'sub-cal' in str(derivative))]
    
    derivatives_nyu = list(path_nyu.rglob('*_lesion-manual.nii.gz'))
    derivatives_nyu = [derivative for derivative in derivatives_nyu if 'labels-ms-spinal' in str(derivative)]
    # We only want 100 T2w ax
    derivatives_nyu_ax = [derivative for derivative in derivatives_nyu if 'acq-ax' in str(derivative)]
    if len(derivatives_nyu_ax) > 100:
        derivatives_nyu_ax = derivatives_nyu_ax[:100]
    derivatives_nyu = derivatives_nyu_ax

    # Path to the file containing the list of subjects to exclude from CanProCo
    if args.canproco_exclude is not None:
       with open(args.canproco_exclude, 'r') as file:
            canproco_exclude_list = yaml.load(file, Loader=yaml.FullLoader)
    # only keep the contrast psir and stir
    canproco_exclude_list = canproco_exclude_list['PSIR'] + canproco_exclude_list['STIR']

    # Remove the excluded subjects from the CanProCo dataset
    for derivative in derivatives_canproco:
        subject_id = derivative.name.replace('_PSIR_lesion-manual.nii.gz', '')
        subject_id = subject_id.replace('_STIR_lesion-manual.nii.gz', '')
        if subject_id in canproco_exclude_list:
            # remove the derivative from the list
            derivatives_canproco.remove(derivative)

    # Print the distribution of contrasts in the datasets
    contrasts_basel_mp2rage = print_dataset_contrasts_distribution(derivatives_basel_mp2rage, "Basel MP2RAGE")
    contrasts_canproco = print_dataset_contrasts_distribution(derivatives_canproco, "CanProCo")
    contrasts_nyu = print_dataset_contrasts_distribution(derivatives_nyu, "NYU")

    all_contrasts = contrasts_basel_mp2rage + contrasts_canproco + contrasts_nyu
    # Change MEGRE TO T2star in the list
    all_contrasts = [contrast.replace('MEGRE', 'T2star') for contrast in all_contrasts]
    # create a dictionnary which stores the counts for each contrast
    contrast_counts = {}
    for contrast in all_contrasts:
        if contrast in contrast_counts:
            contrast_counts[contrast] += 1
        else:
            contrast_counts[contrast] = 1

    all_derivatives_list = derivatives_basel_mp2rage + derivatives_canproco + derivatives_nyu

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = "soft-ms-lesion-downsampling"
    params["labels"] = {
        "0": "background",
        "1": "ms-lesion-seg"
        }
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = "soft-ms-lesion-downsampling"
    params["reference"] = "NeuroPoly"
    # Add the contrasts counts in the params
    params["contrasts"] = contrast_counts

    # iterate through the train/val/test splits and add those which have both image and label
    subjects_basel = []
    subjects_canproco = []
    subjects_nyu = []

    temp_list = []
    for subject_no, derivative in tqdm(enumerate(all_derivatives_list)):

        temp_data_basel = {}
        temp_data_canproco = {}
        temp_data_nyu = {}
        
        # Basel
        if 'basel-mp2rage' in str(derivative):
            temp_data_basel["label"] = str(derivative)
            temp_data_basel["image"] = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
            if os.path.exists(temp_data_basel["label"]) and os.path.exists(temp_data_basel["image"]):
                temp_data_basel["site"]='basel-mp2rage'
                temp_data_basel["contrast"] = str(derivative).replace('_desc-rater3_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                temp_list.append(temp_data_basel)
                # Get the subject
                subject = str(derivative).split('/')[-1].split('_')[0]
                subjects_basel.append(subject)
            else:
                print(f"Label or image not found for {derivative}")
        
        # Canproco
        elif 'canproco' in str(derivative):
            temp_data_canproco["label"] = str(derivative)
            temp_data_canproco["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
            if os.path.exists(temp_data_canproco["label"]) and os.path.exists(temp_data_canproco["image"]):
                temp_data_canproco["site"]='canproco'
                temp_data_canproco["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                temp_list.append(temp_data_canproco)
                # Get the subject
                subject = str(derivative).split('/')[-1].split('_')[0]
                subjects_canproco.append(subject)
            else:
                print(f"Label or image not found for {derivative}")

        # ms-nyu   
        elif 'ms-nyu' in str(derivative):
            temp_data_nyu["label"] = str(derivative)
            temp_data_nyu["image"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').replace('derivatives/labels-ms-spinal-cord-only/', '')
            if os.path.exists(temp_data_nyu["label"]) and os.path.exists(temp_data_nyu["image"]):
                temp_data_nyu["site"]='ms-nyu'
                temp_data_nyu["contrast"] = str(derivative).replace('_lesion-manual.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')
                temp_list.append(temp_data_nyu)
                # Get the subject
                subject = str(derivative).split('/')[-1].split('_')[0]
                subjects_nyu.append(subject)
            else:
                print(f"Label or image not found for {derivative}")

    params["images"] = temp_list
    params["numImages"] = len(params["images"])
    print(f"Number of images: {params['numImages']}")
    params["numSubjects"] = len(set(subjects_basel)) + len(set(subjects_canproco)) + len(set(subjects_nyu))
    print(f"Number of subjects: {params['numSubjects']}")

    final_json = json.dumps(params, indent=4, sort_keys=True)
    os.makedirs(args.path_out, exist_ok=True)
    json_path = os.path.join(args.path_out, f"dataset_{str(date.today())}.json")
    jsonFile = open(json_path, "w")
    jsonFile.write(final_json)
    jsonFile.close()
    print("Dataset JSON saved at: " + json_path)

    return json_path


if __name__ == "__main__":
    main()