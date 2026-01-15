"""
This script takes an input image and downsamples to several lower resolutions and saves the images in the output folder

Input:
    -i: Path to input image
    -o: Path to output folder

Example usage:
    python downsample.py -i path/to/input/image.png -o path/to/output/folder

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Downsample an image to several lower resolutions.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output folder")
    return parser.parse_args()


def downsample_image(input_path, output_path, factors):
    """
    This function downsamples the input image by the specified factors and saves them to the output folder.
    """
    assert os.system(f"sct_resample -i {input_path} -o {output_path} -f {factors[0]}x{factors[1]}x{factors[2]}") == 0, "Error in downsampling image"
    

def main():
    args = parse_args()
    input_image = args.input
    output_folder = args.output
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    downsample_factors = [(0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.7, 0.7, 0.7), (0.6, 0.6, 0.6), (0.5, 0.5, 0.5), (0.4, 0.4, 0.4),
                          (0.9, 0.9, 1.0), (0.8, 0.8, 1.0), (0.7, 0.7, 1.0), (0.6, 0.6, 1.0), (0.5, 0.5, 1.0), (0.4, 0.4, 1.0),
                          (0.9, 1.0, 0.9), (0.8, 1.0, 0.8), (0.7, 1.0, 0.7), (0.6, 1.0, 0.6), (0.5, 1.0, 0.5), (0.4, 1.0, 0.4),
                          (1.0, 0.9, 0.9), (1.0, 0.8, 0.8), (1.0, 0.7, 0.7), (1.0, 0.6, 0.6), (1.0, 0.5, 0.5), (1.0, 0.4, 0.4)
    ]
    
    for factor in tqdm(downsample_factors, desc="Downsampling images"):
        output_image = os.path.join(output_folder, f"downsampled_{factor[0]}x{factor[1]}x{factor[2]}.nii.gz")
        downsample_image(input_image, output_image, factor)
        print(f"Saved downsampled image at: {output_image}")


if __name__ == "__main__":
    main()