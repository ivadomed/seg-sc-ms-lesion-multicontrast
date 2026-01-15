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
    soft_files = list(Path(input_folder).rglob("*_lesion_soft.nii.gz"))
    binary_files = list(Path(input_folder).rglob("*_lesion_binary.nii.gz"))

    # Initialize volume lists
    soft_volumes = []
    binary_volumes = []

    # For each file in soft_files, compute the total lesion volume
    for soft_file in tqdm(soft_files, desc="Computing lesion volumes"):
        # Compute lesion volume
        soft_volume = compute_lesion_volume(soft_file)
        soft_volumes.append(soft_volume)

    # For each file in binary_files, compute the total lesion volume
    for binary_file in tqdm(binary_files, desc="Computing lesion volumes"):
        # Compute lesion volume
        binary_volume = compute_lesion_volume(binary_file)
        binary_volumes.append(binary_volume)

    # Print the stds
    print(f"Soft segmentation volumes (std): {np.std(soft_volumes)}")
    print(f"Binary segmentation volumes (std): {np.std(binary_volumes)}")
    
    # we also perform a statistical test (wilcoxon) to compare the two distributions 
    stat, p = wilcoxon(soft_volumes, binary_volumes)
    print(f"Wilcoxon test statistic: {stat}, p-value: {p}")
    # print if p>0.1 : then both are estimating the same volume
    if p > 0.1:
        print("The two segmentation methods estimate the same volume (p > 0.1)")
    else:
        print("The two segmentation methods estimate different volumes (p <= 0.1)")


if __name__ == "__main__":
    main()