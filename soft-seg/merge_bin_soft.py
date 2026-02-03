"""
This script takes the binary segmentations and merge them with the soft segmentations.

Input:
    -i: Path to input segmentations (the folder contains both soft and binary segmentations)
    -o: Path to output folder to save merged segmentations

Author: Pierre-Louis Benveniste 
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge binary segmentations with soft segmentations.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to input segmentations folder")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output folder to save merged segmentations")
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Get all binary segmentation files
    binary_seg_files = list(Path(input_folder).rglob("*_lesion.nii.gz"))
    binary_seg_files = [str(f) for f in binary_seg_files if "/binary/" in str(f)]

    # For each binary segmentation, we copy them to the output folder, where the soft segmentation is located
    for binary_file in binary_seg_files:
        # Extract path where to save
        relative_path = binary_file.split("/")[-4:]
        output_path = os.path.join(output_folder, *relative_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Copy binary segmentation to the output folder
        os.system(f"cp {binary_file} {output_path}")


if __name__ == "__main__":
    main()
