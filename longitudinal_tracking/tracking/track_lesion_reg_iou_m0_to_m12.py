"""
This code compares lesion segmentations from two timepoints for a single subject using registration to match lesions.
The lesion matching is performed based on the IoU (Intersection over Union) between lesions.
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
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1 (M0)')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2 (M12)')
    parser.add_argument('-pred', '--pred_folder', type=str, required=False, help='Path to the folder containing predicted files (SC, lesion ...)')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    return parser.parse_args()


def compute_IoU_matrix(data_lesion_1, data_lesion_2):
    """
    Compute the IoU matrix between all pairs of lesions in two labeled segmentations.

    Inputs:
        data_lesion_1 : numpy array of the first labeled lesion segmentation
        data_lesion_2 : numpy array of the second labeled lesion segmentation

    Outputs:
        IoU_matrix : array of shape (n_lesions_1, n_lesions_2)
    """
    n_lesions_1 = len(np.unique(data_lesion_1)) - 1
    n_lesions_2 = len(np.unique(data_lesion_2)) - 1
    IoU_matrix = np.zeros((n_lesions_1, n_lesions_2))
    for i in range(1, n_lesions_1 + 1):
        mask_1 = (data_lesion_1 == i)
        for j in range(1, n_lesions_2 + 1):
            mask_2 = (data_lesion_2 == j)
            intersection = np.logical_and(mask_1, mask_2).sum()
            union = np.logical_or(mask_1, mask_2).sum()
            IoU_matrix[i - 1, j - 1] = intersection / union if union > 0 else 0
    return IoU_matrix


def compute_lesion_mapping(IoU_matrix, IoU_threshold):
    """
    Map lesions based on an IoU threshold.

    Inputs:
        IoU_matrix    : array of shape (n_lesions_1, n_lesions_2)
        IoU_threshold : float, minimum IoU to consider a match

    Outputs:
        lesion_mapping : dict mapping row lesion ID (1-indexed) to list of matched column lesion IDs
    """
    n_lesions_1, n_lesions_2 = IoU_matrix.shape
    lesion_mapping = {}
    for i in range(n_lesions_1):
        lesion_mapping[i + 1] = []
        for j in range(n_lesions_2):
            if IoU_matrix[i, j] >= IoU_threshold:
                lesion_mapping[i + 1].append(j + 1)
    return lesion_mapping


def map_lesions_registered_with_IoU_m0_to_m12(input_image1, input_image2, pred_folder, output_folder, IoU_threshold=1e-5):
    """
    Perform lesion mapping from M0 to M12 by registering M0 into M12 space first.

    The inverse warp field (M0→M12) is used to bring M0 lesion labels into M12
    space for IoU-based matching. The matched labels are then warped back to M0
    space to resolve any label relabeling introduced by the double warp, and the
    final mapping M0→M12 is assembled via chaining.

    Inputs:
        input_image1  : path to the M0 image
        input_image2  : path to the M12 image
        pred_folder   : folder containing pre-computed segmentations and warp fields
        output_folder : folder where results and logs are written
        IoU_threshold : minimum IoU to count a lesion pair as matched

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
    assert os.system(f"sct_apply_transfo -i {labeled_lesion_seg_1} -d {input_image2} -w {inv_warping_field_img2_to_1} -o {labeled_lesion_seg_1_reg} -x nn") == 0, "Failed to warp M0 lesion segmentation into M12 space"

    # Step 2: match M0-in-M12 lesions against M12 lesions using IoU
    data_lesion_1_reg = nib.load(labeled_lesion_seg_1_reg).get_fdata()
    data_lesion_2 = nib.load(labeled_lesion_seg_2).get_fdata()
    IoU_matrix = compute_IoU_matrix(data_lesion_1_reg, data_lesion_2)
    logger.info(f"IoU matrix M0-in-M12 vs M12:\n{IoU_matrix}")
    lesion_mapping_reg1_to_2 = compute_lesion_mapping(IoU_matrix, IoU_threshold)
    logger.info(f"Lesion mapping M0-in-M12 → M12 (threshold={IoU_threshold}): {lesion_mapping_reg1_to_2}")

    # Step 3: warp M0-in-M12 labels back to M0 space to recover original M0 label IDs
    assert os.system(f"sct_apply_transfo -i {labeled_lesion_seg_1_reg} -d {input_image1} -w {warping_field_img2_to_1} -o {reg_back_labeled_lesion_seg_1} -x nn") == 0, "Failed to warp M0-in-M12 lesion segmentation back to M0 space"

    # Step 4: match original M0 labels with round-tripped M0 labels using IoU
    data_lesion_1 = nib.load(labeled_lesion_seg_1).get_fdata()
    data_lesion_1_reg_back = nib.load(reg_back_labeled_lesion_seg_1).get_fdata()
    IoU_matrix_reg_back = compute_IoU_matrix(data_lesion_1_reg_back, data_lesion_1)
    logger.info(f"IoU matrix M0-in-M12-back vs M0:\n{IoU_matrix_reg_back}")
    # rows = round-tripped M0-in-M12 labels, cols = original M0 labels
    lesion_mapping_regback1_to_1 = compute_lesion_mapping(IoU_matrix_reg_back, IoU_threshold)
    logger.info(f"Lesion mapping round-tripped M0 → M0: {lesion_mapping_regback1_to_1}")

    # Step 5: chain round-trip labels → M0 → M12
    # Invert lesion_mapping_regback1_to_1 to get M0 → round-trip-label
    lesion_mapping_1_to_regback1 = {}
    for regback1_id, m0_ids in lesion_mapping_regback1_to_1.items():
        for m0_id in m0_ids:
            lesion_mapping_1_to_regback1.setdefault(m0_id, []).append(regback1_id)

    full_mapping_1_to_2 = {}
    for lesion1_id, regback1_ids in lesion_mapping_1_to_regback1.items():
        full_mapped = []
        for regback1_id in regback1_ids:
            if regback1_id in lesion_mapping_reg1_to_2:
                full_mapped.extend(lesion_mapping_reg1_to_2[regback1_id])
        full_mapping_1_to_2[lesion1_id] = full_mapped
    logger.info(f"Full lesion mapping M0→M12 (IoU threshold={IoU_threshold}): {full_mapping_1_to_2}")

    return full_mapping_1_to_2


if __name__ == "__main__":
    args = parse_args()
    map_lesions_registered_with_IoU_m0_to_m12(
        args.input_image1, args.input_image2, args.pred_folder, args.output_folder
    )
