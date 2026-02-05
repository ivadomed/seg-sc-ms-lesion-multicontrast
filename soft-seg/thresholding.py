"""
This code takes existing soft segmentation and performs thresholding to create binary segmentations at various thresholds.

Input:
    -i: Path to input segmentations (the folder contains both soft and binary segmentations)
    -o: Path to output folder to save thresholded segmentations
    --high-threshs: flag to indicate to use high thresholds


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
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output folder to save thresholded segmentations")
    parser.add_argument("--high-thresh", action="store_true", help="Use high thresholds if set")
    return parser.parse_args()


def main_thresholding(input_folder, output_folder, high_thresh=False):
    # Build output folder
    os.makedirs(output_folder, exist_ok=True)

    thresholds = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    if high_thresh:
        thresholds = [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2]

    # For each threshold, create a new folder and threshold the soft segmentations
    for thresh in tqdm(thresholds, desc="Threshold levels"):
        thresh_folder = os.path.join(output_folder, f"preds_thresh_{thresh}")
        os.makedirs(thresh_folder, exist_ok=True)

        # We first start by copying the input folder to the destination output folder
        os.system(f"cp -r {input_folder} {thresh_folder}")

        # Now we list all soft segmentation files
        soft_seg_files = list(Path(thresh_folder).rglob("*_lesion.nii.gz"))
        soft_seg_files = [str(f) for f in soft_seg_files if "/soft/" in str(f)]

        # Threshold each soft segmentation
        for soft_file in tqdm(soft_seg_files, desc=f"Thresholding at {thresh}"):
            # Load soft segmentation
            soft_img = nib.load(soft_file)
            soft_data = soft_img.get_fdata()

            # Apply threshold to produce soft segmentation where voxels below threshold are set to 0
            soft_data_thresh = soft_data.copy()
            soft_data_thresh[soft_data_thresh < thresh] = 0

            # Save the thresholded soft segmentation back
            soft_img_thresh = nib.Nifti1Image(soft_data_thresh, affine=soft_img.affine, header=soft_img.header)
            nib.save(soft_img_thresh, soft_file)


if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    high_thresh = args.high_thresh
    
    main_thresholding(input_folder, output_folder, high_thresh)