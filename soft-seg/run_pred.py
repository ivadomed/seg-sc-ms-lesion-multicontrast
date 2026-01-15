"""
This script takes all files in a folder and runs lesion segmentation using SCT.
For each image, both a binary and a soft segmentation are generated.

Input:
    -i: Path to input folder containing images
    -o: Path to output folder to save predictions

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from tqdm import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run lesion segmentation on all images in a folder.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output folder to save predictions")
    return parser.parse_args()


def run_segmentation(input_image, output_image, soft_seg=False):
    """
    This function runs lesion segmentation on the input image and saves the prediction to the output image.
    """
    soft_ms_lesion_flag = "-soft-ms-lesion" if soft_seg else ""
    assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {input_image} -o {output_image} {soft_ms_lesion_flag}") == 0, "Error in lesion segmentation"


def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    input_images = list(Path(input_folder).rglob("*.nii.gz"))
    input_images = [str(img) for img in input_images]

    for img in tqdm(input_images, desc="Running lesion segmentation"):
        
        # Define output paths
        site = img.split("/")[-3]
        sub_name = img.split("/")[-2]
        img_name = img.split("/")[-1].replace(".nii.gz", "")
        output_folder_site = os.path.join(output_folder, site, sub_name)
        output_binary_path = os.path.join(output_folder_site, "binary", f"{img_name}_lesion.nii.gz")
        output_soft_path = os.path.join(output_folder_site, "soft", f"{img_name}_lesion.nii.gz")
        
        # Run binary segmentation
        run_segmentation(img, output_binary_path, soft_seg=False)
        
        # Run soft segmentation
        run_segmentation(img, output_soft_path, soft_seg=True)


if __name__ == "__main__":
    main()