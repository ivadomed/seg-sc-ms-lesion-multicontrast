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


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample images in an msd file to several lower resolutions.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input msd file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output folder")
    return parser.parse_args()


def apply_downsampling(input_path, output_path, factors):
    """
    This function downsamples the input image by the specified factors and saves them to the output folder.
    """
    assert os.system(f"sct_resample -i {input_path} -o {output_path} -f {factors[0]}x{factors[1]}x{factors[2]}") == 0, "Error in downsampling image"
    

def downsample_image(input_image, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    downsample_factors = [(0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.7, 0.7, 0.7), (0.6, 0.6, 0.6), (0.5, 0.5, 0.5),
                          (0.9, 0.9, 1.0), (0.8, 0.8, 1.0), (0.7, 0.7, 1.0), (0.6, 0.6, 1.0), (0.5, 0.5, 1.0),
                          (0.9, 1.0, 0.9), (0.8, 1.0, 0.8), (0.7, 1.0, 0.7), (0.6, 1.0, 0.6), (0.5, 1.0, 0.5),
                          (1.0, 0.9, 0.9), (1.0, 0.8, 0.8), (1.0, 0.7, 0.7), (1.0, 0.6, 0.6), (1.0, 0.5, 0.5),
    ]
    
    image_name = input_image.split('/')[-1].replace('.nii.gz', '')
    for factor in downsample_factors:
        output_image = os.path.join(output_folder, f"{image_name}_downsampled_{factor[0]}x{factor[1]}x{factor[2]}.nii.gz")
        apply_downsampling(input_image, output_image, factor)
        print(f"Saved downsampled image at: {output_image}")

def main_downsample(input_msd, output_folder):
    # Build output folder
    os.makedirs(output_folder, exist_ok=True)

    # Open the msd file
    with open(input_msd, 'r') as f:
        msd_data = json.load(f)
    images = msd_data['images']

    # Iterate through each image in the msd file
    for image in tqdm(images, desc="Downsampling images"):
        input_msd_file = image['image']
        image_site = image['site']
        # Build the output folder for this image
        output_folder_image = os.path.join(output_folder, image_site, input_msd_file.split('/')[-1].replace('.nii.gz', ''))
        os.makedirs(output_folder_image, exist_ok=True)
        # Downsample the image
        downsample_image(input_msd_file, output_folder_image)
        
    return None


if __name__ == "__main__":
    args = parse_args()
    input_msd = args.input
    output_folder = args.output

    main_downsample(input_msd, output_folder)