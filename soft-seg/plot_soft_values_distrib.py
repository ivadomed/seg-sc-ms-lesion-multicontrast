"""
This scripts explores the current distribution of soft values in soft segmentations.

Input:
    -i: path the folder containing soft segmentation files
    -o: path to the output folder to save the plots

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Plot distribution of soft values in soft segmentations.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to folder containing soft segmentation files")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output folder to save plots")
    return parser.parse_args()


def main_plot_soft_values_distrib(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List soft segmentation files
    soft_files = list(Path(input_folder).rglob("*_lesion.nii.gz"))
    soft_files = [str(f) for f in soft_files]
    soft_files = [f for f in soft_files if "/soft/" in f]

    all_soft_values = []
    for soft_file in tqdm(soft_files):
        # Load the lesion mask
        lesion_img = nib.load(soft_file)
        lesion_data = lesion_img.get_fdata()

        # Get all non-zero values (i.e. all values that are part of the lesion)
        seg_data = lesion_data[lesion_data > 1e-6]
        all_soft_values.extend(seg_data)

    # Plot distribution of soft values
    plt.figure(figsize=(10, 6))
    sns.histplot(all_soft_values, bins=50, kde=True)
    plt.title("Distribution of Soft Values in Soft Segmentations")
    plt.xlabel("Soft Value")
    plt.ylabel("Frequency")
    plt.xlim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(output_folder, "soft_values_distribution.png"))

    return None


if __name__ == "__main__":
    args = parse_args()
    main_plot_soft_values_distrib(args.input_folder, args.output_folder)