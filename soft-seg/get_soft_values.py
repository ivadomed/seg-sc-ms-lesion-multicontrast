"""
This script gets the soft values in the soft segmentation produced

Input:
    -i: Path to input segmentations (the folder contains both soft and binary segmentations)

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Threshold soft segmentations to create binary segmentations.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to input segmentations folder")
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = args.input_folder
    
    soft_seg_files = list(Path(input_folder).rglob("*_lesion.nii.gz"))
    soft_seg_files = [str(f) for f in soft_seg_files if "/soft/" in str(f)]

    soft_values = []

    for soft_file in tqdm(soft_seg_files):
        # Load soft segmentation
        soft_img = nib.load(soft_file)
        soft_data = soft_img.get_fdata()

        # Get the soft values (non-zero values)
        soft_values_file = soft_data[soft_data > 0]
        soft_values.extend(soft_values_file.flatten().tolist())

    print("Total number of soft values collected:", len(soft_values))
    print("Range of soft values:", min(soft_values), "to", max(soft_values))


if __name__ == "__main__":
    main()
