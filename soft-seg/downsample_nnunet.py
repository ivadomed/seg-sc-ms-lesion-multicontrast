"""
This script takes an input msd file and downsamples each of its images to several lower resolutions and saves the images in the output folder

Input:
    -i: Path to input msd file
    -o: Path to output folder

Example usage:
    python downsample.py -i path/to/input/msd_file.json -o path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from tqdm import tqdm
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample images in an msd file to several lower resolutions.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the nnunet format image folder")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output folder")
    return parser.parse_args()


def apply_downsampling(input_path, output_path, factors):
    """
    This function downsamples the input image by the specified factors and saves them to the output folder.
    """
    assert os.system(f"sct_resample -i {input_path} -o {output_path} -f {factors[0]}x{factors[1]}x{factors[2]}") == 0, "Error in downsampling image"


def main_downsample(input_msd, output_folder):
    # Build output folder
    os.makedirs(output_folder, exist_ok=True)

    # List all images in the input folder
    images = list(Path(input_msd).rglob("*.nii.gz"))
    images = [str(img) for img in images]

    downsample_factors = [(0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.7, 0.7, 0.7), (0.6, 0.6, 0.6), (0.5, 0.5, 0.5),
                          (0.9, 0.9, 1.0), (0.8, 0.8, 1.0), (0.7, 0.7, 1.0), (0.6, 0.6, 1.0), (0.5, 0.5, 1.0),
                          (0.9, 1.0, 0.9), (0.8, 1.0, 0.8), (0.7, 1.0, 0.7), (0.6, 1.0, 0.6), (0.5, 1.0, 0.5),
                          (1.0, 0.9, 0.9), (1.0, 0.8, 0.8), (1.0, 0.7, 0.7), (1.0, 0.6, 0.6), (1.0, 0.5, 0.5),
    ]

    for down_factor in downsample_factors:
        # Build the output folder
        output_folder_factor = os.path.join(output_folder, f"downsampled_{down_factor[0]}x{down_factor[1]}x{down_factor[2]}")
        os.makedirs(output_folder_factor, exist_ok=True)
        # Downsample each image and save it to the output folder
        for image in tqdm(images, desc=f"Downsampling images with factor {down_factor}"):
            downsampled_image_path = os.path.join(output_folder_factor, image.split('/')[-1])
            apply_downsampling(image, downsampled_image_path, down_factor)
        
    return None


if __name__ == "__main__":
    args = parse_args()
    input_msd = args.input
    output_folder = args.output

    main_downsample(input_msd, output_folder)