"""
This code compares lesion segmentations from two timepoints for a single subject using registration to match lesions.
The lesion matching is based on Hungarian algorithm applied to the center of mass (CoM) of lesions.
Registration direction: M0 → M12 (i.e., M0 lesion seg is warped into M12 space for matching).

Input:
    -i1 : path to the input image at timepoint 1 (M0)
    -i2 : path to the input image at timepoint 2 (M12)
    -pred: path to the folder containing predicted files (SC, lesion ...)
    -o : path to the output folder where comparison results will be stored

Output:
    None

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
from loguru import logger
from datetime import date
import nibabel as nib
from scipy import ndimage
import numpy as np
from scipy.optimize import linear_sum_assignment
import json
import sys

file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import compute_lesion_CoM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1 (M0)')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2 (M12)')
    parser.add_argument('-pred', '--pred_folder', type=str, required=False, help='Path to the folder containing predicted files (SC, lesion ...)')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    return parser.parse_args()


def compute_lesion_mapping(lesion_1_CoM, lesion_2_CoM):
    """
    Compute optimal lesion assignment between two sets of CoMs using the Hungarian algorithm.

    Inputs:
        lesion_1_CoM : dict mapping lesion indices to CoM coordinates (source)
        lesion_2_CoM : dict mapping lesion indices to CoM coordinates (target)

    Outputs:
        lesion_mapping : dict mapping source lesion ID to target lesion ID
    """
    cost_matrix = np.zeros((len(lesion_1_CoM), len(lesion_2_CoM)))
    for i, (lesion1_id, com1) in enumerate(lesion_1_CoM.items()):
        for j, (lesion2_id, com2) in enumerate(lesion_2_CoM.items()):
            cost_matrix[i, j] = np.linalg.norm(np.array(com1) - np.array(com2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    lesion_mapping = {}
    for i, j in zip(row_ind, col_ind):
        lesion1_id = int(list(lesion_1_CoM.keys())[i])
        lesion2_id = int(list(lesion_2_CoM.keys())[j])
        lesion_mapping[lesion1_id] = lesion2_id

    return lesion_mapping


def map_lesions_registered_with_CoM_m0_to_m12(input_image1, input_image2, pred_folder, output_folder):
    """
    Perform lesion mapping from M0 to M12 by registering M0 into M12 space first.

    The inverse warp field (M0→M12) is used to bring M0 lesion labels into M12
    space for CoM-based matching. The matched labels are then warped back to M0
    space to resolve any label relabeling introduced by the double warp, and the
    final mapping M0→M12 is assembled via chaining.

    Inputs:
        input_image1 : path to the M0 image
        input_image2 : path to the M12 image
        pred_folder  : folder containing pre-computed segmentations and warp fields
        output_folder: folder where results and logs are written

    Outputs:
        full_mapping_1_to_2 : dict mapping M0 lesion IDs to lists of M12 lesion IDs
    """
    os.makedirs(output_folder, exist_ok=True)
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "qc"), exist_ok=True)

    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))

    image_1_name = Path(input_image1).name  # M0
    image_2_name = Path(input_image2).name  # M12

    # Pre-existing files
    labeled_lesion_seg_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
    labeled_lesion_seg_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
    # warp_to_M0: M12→M0 direction (used to warp back from M12 space to M0 space)
    warping_field_img2_to_1 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_warp_to_' + image_1_name))
    # inv_warp_to_M0: M0→M12 direction (used to warp M0 images into M12 space)
    inv_warping_field_img2_to_1 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_inv_warp_to_' + image_1_name))

    # Generated files (written to temp)
    labeled_lesion_seg_1_reg = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion-seg-reg-labeled.nii.gz'))
    reg_back_labeled_lesion_seg_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_lesion-seg-registered-back-labeled.nii.gz'))

    # Step 1: warp M0 labeled lesion seg into M12 space
    assert os.system(
        f"sct_apply_transfo -i {labeled_lesion_seg_1} -d {input_image2} "
        f"-w {inv_warping_field_img2_to_1} -o {labeled_lesion_seg_1_reg} -x nn"
    ) == 0, "Failed to warp M0 lesion segmentation into M12 space"

    # Step 2: compute CoMs — M0 native, M12 native, M0-in-M12
    lesion_1_CoM = compute_lesion_CoM(labeled_lesion_seg_1)
    lesion_2_CoM = compute_lesion_CoM(labeled_lesion_seg_2)
    lesion_1_reg_CoM = compute_lesion_CoM(labeled_lesion_seg_1_reg)

    # Step 3: match M0-in-M12 lesions against M12 lesions (both in M12 space)
    lesion_mapping_reg1_to_2 = compute_lesion_mapping(lesion_1_reg_CoM, lesion_2_CoM)
    logger.info(f"Lesion mapping from M0-in-M12 to M12: {lesion_mapping_reg1_to_2}")

    # Step 4: warp M0-in-M12 labels back to M0 space to recover original M0 label IDs
    assert os.system(
        f"sct_apply_transfo -i {labeled_lesion_seg_1_reg} -d {input_image1} "
        f"-w {warping_field_img2_to_1} -o {reg_back_labeled_lesion_seg_1} -x nn"
    ) == 0, "Failed to warp M0-in-M12 lesion segmentation back to M0 space"

    # Step 5: match original M0 labels with round-tripped M0 labels
    lesion_1_reg_back_CoM = compute_lesion_CoM(reg_back_labeled_lesion_seg_1)
    lesion_mapping_1_to_regback1 = compute_lesion_mapping(lesion_1_CoM, lesion_1_reg_back_CoM)
    logger.info(f"Lesion mapping from M0 to round-tripped M0: {lesion_mapping_1_to_regback1}")

    # Step 6: chain M0 → M0-in-M12 (round-trip labels) → M12
    full_mapping_1_to_2 = {}
    for lesion1_id, regback1_id in lesion_mapping_1_to_regback1.items():
        if regback1_id in lesion_mapping_reg1_to_2:
            lesion2_id = lesion_mapping_reg1_to_2[regback1_id]
            full_mapping_1_to_2[lesion1_id] = [lesion2_id]
    logger.info(f"Full lesion mapping M0→M12: {full_mapping_1_to_2}")

    return full_mapping_1_to_2


if __name__ == "__main__":
    args = parse_args()
    map_lesions_registered_with_CoM_m0_to_m12(
        args.input_image1, args.input_image2, args.pred_folder, args.output_folder
    )
