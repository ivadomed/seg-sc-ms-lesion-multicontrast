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
    return lesion_volume, voxel_volume


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

    # Initialize dataframe of computed volumes
    volumes_df = pd.DataFrame(columns=["subject", "resamp_factor", "voxel_volume", "soft_volume", "binary_volume"])

    # For each file in soft_files, compute the total lesion volume
    for soft_file in tqdm(soft_files, desc="Computing lesion volumes"):
        # Compute lesion volume for soft segmentation
        soft_volume, voxel_volume = compute_lesion_volume(soft_file)
        resamp_factor = soft_file.split("/")[-1].replace("downsampled_", "").replace("_lesion.nii.gz", "")
        soft_file_site = soft_file.split("/")[-4]
        soft_file_sub = soft_file.split("/")[-3]
        sub = soft_file_site + "_" + soft_file_sub
        # Compute lesion volume for binary segmentation
        corresponding_binary_file = soft_file.replace("/soft/", "/binary/")
        binary_volume, _ = compute_lesion_volume(corresponding_binary_file)
        # Update the dataframe
        new_line = pd.DataFrame({"subject": [sub], "resamp_factor": [resamp_factor], "voxel_volume": [voxel_volume], "soft_volume": [soft_volume], "binary_volume": [binary_volume]})
        volumes_df = pd.concat([volumes_df, new_line], ignore_index=True)

    # We save the volumes dataframe to a csv file
    volumes_csv_path = os.path.join(output_folder, "lesion_volumes.csv")
    volumes_df.to_csv(volumes_csv_path, index=False)


if __name__ == "__main__":
    main()