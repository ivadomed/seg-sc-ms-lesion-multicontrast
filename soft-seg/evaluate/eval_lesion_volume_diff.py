"""
This script computes the volume differrence between lesion segmentations of the same sessions.

Input:
    -msd: Path to the msd dataset
    -bin: Folder containing the binary lesion masks
    -soft: Folder containing the soft lesion segmentations
    -output: Path to the output folder

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
import json
import nibabel as nib
import numpy as np
from loguru import logger
from tqdm import tqdm
import numpy as np
from scipy.stats import mannwhitneyu


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate lesion volume difference between binary and soft segmentations.")
    parser.add_argument("-msd", type=str, required=True, help="Path to the msd dataset")
    parser.add_argument("-bin", type=str, required=True, help="Folder containing the binary lesion masks")
    parser.add_argument("-soft", type=str, required=True, help="Folder containing the soft lesion segmentations")
    parser.add_argument("-output", type=str, required=True, help="Path to the output folder")
    return parser.parse_args()


def get_volume(segmentation_path, resolution, threshold=0.5):
    # Load the segmentation
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()
    # Compute volume
    voxel_volume = resolution[0] * resolution[1] * resolution[2]
    # Thresholding the segmentation:
    seg_data = seg_data[seg_data >= threshold]
    # Compute volume in case of soft segmentation, i.e. volume of a voxel labeled 0.5 is counted as half a voxel
    lesion_volume = np.sum(seg_data) * voxel_volume
    return lesion_volume


def main():
    # Parsing arguments
    args = parse_args()
    msd_path = args.msd
    bin_folder = args.bin
    soft_folder = args.soft
    output_folder = args.output

    # Build output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    logger.add(os.path.join(output_folder, 'eval_lesion_volume_diff.log'))

    # Open the msd dataset to get the list of images
    # Load the msd json file
    with open(msd_path, 'r') as f:
        msd_json = json.load(f)
    images = msd_json['data']

    bin_avgs = []
    bin_stds = []
    soft_avgs = []
    soft_stds = []
    number_subjects = 0
    number_sessions = 0
    for sub in tqdm(images):
        number_subjects += 1
        # iterate over sessions
        for session in images[sub]:
            number_sessions += 1
            # Iterate over the images
            session_bin_volumes = []
            session_soft_volumes = []
            for img in images[sub][session]['images']:
                # Build paths to binary and soft lesion segmentations
                bin_path = os.path.join(bin_folder, sub, session, 'anat', img.split('/')[-1].replace('.nii.gz', '_lesion-seg-bin.nii.gz'))
                soft_path = os.path.join(soft_folder, sub, session, 'anat', img.split('/')[-1].replace('.nii.gz', '_lesion-seg-soft.nii.gz'))

                # Get resolution
                resolution = nib.load(bin_path).header.get_zooms()

                # Get volumes
                volume_bin = get_volume(bin_path, resolution)
                volume_soft = get_volume(soft_path, resolution, threshold=1e-5)
                session_bin_volumes.append(volume_bin)
                session_soft_volumes.append(volume_soft)
            # Add the average and std of the session
            bin_avgs.append(np.mean(session_bin_volumes))
            bin_stds.append(np.std(session_bin_volumes))
            soft_avgs.append(np.mean(session_soft_volumes))
            soft_stds.append(np.std(session_soft_volumes))

    # Print the number of subjects and sessions
    logger.info(f"Number of subjects: {number_subjects}")
    logger.info(f"Number of sessions: {number_sessions}")
    
    # Print the avg of avg volumes per session
    logger.info(f"Average binary lesion volume: {np.mean(bin_avgs)} ± {np.mean(bin_stds)} mm³")
    logger.info(f"Average soft lesion volume: {np.mean(soft_avgs)} ± {np.mean(soft_stds)} mm³")

    # Make statistic test to see if bin_stds and soft_stds are significantly different
    stat, p_value = mannwhitneyu(bin_stds, soft_stds)
    logger.info(f"Mann-Whitney U test between binary and soft lesion volume stds: stat={stat}, p-value={p_value}")
    

if __name__ == "__main__":
    main()