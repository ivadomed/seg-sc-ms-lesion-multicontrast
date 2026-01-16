"""
This script plots the volumes of soft and binary segmentations for comparison.

Input:
    -i: path the folder containing soft and binary segmentation files
    -o: path to the output folder to save the plots

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.stats import wilcoxon


def parse_args():
    parser = argparse.ArgumentParser(description="Plot volumes of soft and binary segmentations.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to folder containing segmentation files")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output folder to save plots")
    return parser.parse_args()


def compute_lesion_volume(lesion_mask_path):
    # Load the lesion mask
    lesion_img = nib.load(lesion_mask_path)
    lesion_data = lesion_img.get_fdata()

    # Get voxel dimensions in mm
    voxel_dims = lesion_img.header.get_zooms()

    # Calculate the volume of a single voxel in cubic millimeters
    voxel_volume = np.prod(voxel_dims)

    # Compute volume in case of soft segmentation, i.e. volume of a voxel labeled 0.5 is counted as half a voxel
    seg_data = lesion_data[lesion_data > 1e-6]
    lesion_voxel_volume = np.sum(seg_data)

    # Calculate the total lesion volume
    lesion_volume = lesion_voxel_volume * voxel_volume
    return lesion_volume


def main():

    # Parse arguments
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List soft and binary segmentation files
    soft_files = list(Path(input_folder).rglob("*_lesion.nii.gz"))
    soft_files = [str(f) for f in soft_files]
    soft_files = [f for f in soft_files if "/soft/" in f]
    binary_files = list(Path(input_folder).rglob("*_lesion.nii.gz"))
    binary_files = [str(f) for f in binary_files]
    binary_files = [f for f in binary_files if "/binary/" in f]

    # Initialize dataframe of computed volumes
    volumes_df = pd.DataFrame(columns=["subject", "soft_volume", "binary_volume"])

    # For each file in soft_files, compute the total lesion volume
    for soft_file in tqdm(soft_files, desc="Computing lesion volumes"):
        # Compute lesion volume
        soft_volume = compute_lesion_volume(soft_file)
        soft_file_site = soft_file.split("/")[-4]
        soft_file_sub = soft_file.split("/")[-3]
        sub = soft_file_site + "_" + soft_file_sub
        # Update the dataframe
        new_line = pd.DataFrame({"subject": [sub], "soft_volume": [soft_volume]})
        volumes_df = pd.concat([volumes_df, new_line], ignore_index=True)

    # For each file in binary_files, compute the total lesion volume
    for binary_file in tqdm(binary_files, desc="Computing lesion volumes"):
        # Compute lesion volume
        binary_volume = compute_lesion_volume(binary_file)
        binary_file_site = binary_file.split("/")[-4]
        binary_file_sub = binary_file.split("/")[-3]
        sub = binary_file_site + "_" + binary_file_sub
        # Update the dataframe
        new_line = pd.DataFrame({"subject": [sub], "binary_volume": [binary_volume]})
        volumes_df = pd.concat([volumes_df, new_line], ignore_index=True)

    # We save the volumes dataframe to a csv file
    volumes_csv_path = os.path.join(output_folder, "lesion_volumes.csv")
    volumes_df.to_csv(volumes_csv_path, index=False)

    # For each subject, we compute the mean and std of soft and binary volumes
    final_volumes_df = pd.DataFrame(columns=["subject", "soft_volume_mean", "soft_volume_std", "binary_volume_mean", "binary_volume_std"])
    subjects = volumes_df['subject'].unique()
    for sub in subjects:
        sub_volumes = volumes_df[volumes_df['subject'] == sub]
        soft_volume_mean = sub_volumes['soft_volume'].mean()
        soft_volume_std = sub_volumes['soft_volume'].std()
        binary_volume_mean = sub_volumes['binary_volume'].mean()
        binary_volume_std = sub_volumes['binary_volume'].std()
        # Update the final volumes dataframe
        new_line = pd.DataFrame({"subject": [sub],
                                 "soft_volume_mean": [soft_volume_mean],
                                 "soft_volume_std": [soft_volume_std],
                                 "binary_volume_mean": [binary_volume_mean],
                                 "binary_volume_std": [binary_volume_std]})
        final_volumes_df = pd.concat([final_volumes_df, new_line], ignore_index=True)

    # We then print the avg difference in stds between soft and binary volumes
    diff_stds = final_volumes_df['soft_volume_std'] - final_volumes_df['binary_volume_std']
    avg_diff_stds = diff_stds.mean()
    print("VOLUMES:")
    print("Average volumes for soft segmentations:", final_volumes_df['soft_volume_mean'].mean())
    print("Average volumes for binary segmentations:", final_volumes_df['binary_volume_mean'].mean())
    print("STD:")
    print("Average std for soft volumes:", final_volumes_df['soft_volume_std'].mean())
    print("Average std for binary volumes:", final_volumes_df['binary_volume_std'].mean())
    print("DIFF STD:")
    print(f"Average difference in stds between soft and binary volumes: {avg_diff_stds}")

    # We check if the std lists are significantly different using a Wilcoxon test
    print("STAT TEST:")
    stat, p_value = wilcoxon(final_volumes_df['soft_volume_std'], final_volumes_df['binary_volume_std'])
    print("Wilcoxon test between soft and binary volume stds:")
    print(f"Statistic: {stat}, p-value: {p_value}")


if __name__ == "__main__":
    main()