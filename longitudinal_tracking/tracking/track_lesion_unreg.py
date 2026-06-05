"""
This code compares lesion segmentations from two timepoints for a single subject.

Input:
    -i1 : path to the input image at timepoint 1
    -i2 : path to the input image at timepoint 2
    -pred: path tp the folder containing predicted files (SC, lesion ...)
    -o : path to the output folder where comparison results will be stored
    -w_z_over_disk : weight for the z-axis distance relative to the disk plane when computing lesion matching (default: 25.0)

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
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
import json
import sys
# Import the functions from utils in parent folder
file_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(file_path, ".."))
sys.path.insert(0, root_path)
from utils import segment_sc, segment_lesions, get_centerline, get_levels, label_lesion_seg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i1', '--input_image1', type=str, required=True, help='Path to the input image at timepoint 1')
    parser.add_argument('-i2', '--input_image2', type=str, required=True, help='Path to the input image at timepoint 2')
    parser.add_argument('-pred', '--pred_folder', type=str, required=False, help='Path to the folder containing predicted files (SC, lesion ...)')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Path to the output folder where comparison results will be stored')
    parser.add_argument('--w_z_over_disk', type=float, default=25.0, help='Weight for the z-axis distance relative to the disk plane when computing lesion matching')
    return parser.parse_args()


def interpolate_between_levels(centerline_data, labeled_centerline_data, level_coords_1, level_coords_2, SI_axis, superior_level, resolution):
    """
    Interpolate the labeled values between two vertebral levels along the centerline.
    """
    # We get all points between these two levels (i.e. between SI values of level 1 and level 2)
    between_coords = []
    for coord in np.argwhere(centerline_data == 1):
        SI_value = coord[SI_axis]
        if SI_value <= level_coords_1[SI_axis] and SI_value >= level_coords_2[SI_axis]:
            between_coords.append(coord)
    # Remove level coords from between coords
    between_coords = [coord for coord in between_coords if not np.array_equal(level_coords_1, coord) and not np.array_equal(level_coords_2, coord)]
    # We interpolate the values for these points
    ## If dist to level 1 is d1 and dist to level 2 is d2, then the value is (d2/(d1+d2))*level1 + (d1/(d1+d2))*level2
    for coord in between_coords:
        dist_1 = np.min(np.linalg.norm((level_coords_1 - coord) * resolution))
        dist_2 = np.min(np.linalg.norm((level_coords_2 - coord) * resolution))
        total_dist = dist_1 + dist_2
        weight = dist_1 / total_dist
        labeled_value = weight + superior_level
        labeled_centerline_data[tuple(coord)] = labeled_value
    return labeled_centerline_data


def label_centerline(centerline, levels, output_labeled_centerline):
    """
    This function labels the centerline so that it takes continuous values following the disc levels.
    For example, the closest point to C1 is labeled 1, the closest point to C2 is labeled 2, and so on.
    For intermediate points, the value is interpolated.

    Inputs:
        centerline : path to the centerline
        levels : path to the vertebral levels
        output_labeled_centerline : path to save the labeled centerline

    Outputs:
        None
    """
    # Load centerline and levels data
    centerline_data = nib.load(centerline).get_fdata()
    levels_data = nib.load(levels).get_fdata()

    # We get the SI direction of the image
    orientation = nib.aff2axcodes(nib.load(centerline).affine)
    SI_axis = orientation.index('I') if 'I' in orientation else orientation.index('S')

    # Get the resolution
    resolution = nib.load(centerline).header.get_zooms()

    # Create a copy of the centerline data to store the labeled values
    labeled_centerline_data = np.zeros_like(centerline_data)

    # For each level, assign the corresponding label to the closest centerline point
    level_labels = np.unique(levels_data)
    level_labels = level_labels[level_labels != 0]  # Exclude background
    for level in level_labels:
        # Get the coordinates of the current level
        level_coords = np.argwhere(levels_data == level)
        if level_coords.size == 0:
            continue
        # For each coordinate in the level, find the closest centerline point
        for coord in level_coords:
            distances = np.linalg.norm((np.argwhere(centerline_data == 1) - coord) * resolution, axis=1)
            closest_idx = np.argmin(distances)
            closest_point = np.argwhere(centerline_data == 1)[closest_idx]
            # Assign the level label to the closest centerline point
            labeled_centerline_data[tuple(closest_point)] = level
    
    # Interpolate values along the centerline
    for level, i in enumerate(level_labels):
        # We first take car of the centerline points that are above the first level
        if i == 0:
            superior_level = level + 1e-3 # A small value because we don't want it to be 0 like the background
            level_coords_2 = np.argwhere(labeled_centerline_data == level + 1)[0]
            # We consider that the upper level is at the top of the centerline (max IS value)
            level_coords_1 = np.argwhere(centerline_data == 1)[np.argmax(np.argwhere(centerline_data == 1)[:, SI_axis])]
            # And we assign a value for the superior level
            labeled_centerline_data[tuple(level_coords_1)] = superior_level
        # Then we take care of the centerline points that are below the last level
        elif i == len(level_labels):
            superior_level = level
            level_coords_1 = np.argwhere(labeled_centerline_data == level)[0]
            # We consider that the lower level is at the bottom of the centerline (min IS value)
            level_coords_2 = np.argwhere(centerline_data == 1)[np.argmin(np.argwhere(centerline_data == 1)[:, SI_axis])]
        # in this case, the centerline points are between two levels
        else:
            # We get the coordinates of the two levels
            level_coords_1 = np.argwhere(labeled_centerline_data == level)[0]
            level_coords_2 = np.argwhere(labeled_centerline_data == level + 1)[0]
            superior_level = level
        # Then we interpolate the values between these two levels
        labeled_centerline_data = interpolate_between_levels(centerline_data, labeled_centerline_data, level_coords_1, level_coords_2, SI_axis, superior_level, resolution)
    
    # Save the labeled centerline
    labeled_centerline_img = nib.Nifti1Image(labeled_centerline_data, nib.load(centerline).affine)
    nib.save(labeled_centerline_img, output_labeled_centerline)

    # Release memory
    del centerline_data, levels_data, labeled_centerline_data

    return None


def analyze_lesions(labeled_lesion_seg):
    """
    This function analyzes the lesions given the labeled lesion segmentation.
    It outputs a dictionary with lesion IDs as keys and their CoM coordinates as values.

    Inputs:
        labeled_lesion_seg : path to the labeled lesion segmentation
    
    Outputs:
        analysis_results : results of the lesion analysis
    """
    # Initialize the analysis results
    analysis_results = {}

    # We load the segmentation and look at each lesion (connected component)
    lesion_data = nib.load(labeled_lesion_seg).get_fdata()
    labels = np.unique(lesion_data)
    labels = labels[labels != 0]  # Exclude background
    lesion_center_of_mass = ndimage.center_of_mass(lesion_data, lesion_data, labels)
    # We also compute the volume of each lesion
    resolution = nib.load(labeled_lesion_seg).header.get_zooms()
    # We find to which plane each axis corresponds to
    orientation = nib.aff2axcodes(nib.load(labeled_lesion_seg).affine)
    for i, axis in enumerate(orientation):
        if axis in ['R', 'L']:
            RL_axis = i
        elif axis in ['A', 'P']:
            AP_axis = i
        elif axis in ['S', 'I']:
            SI_axis = i

    for label, CoM in zip(labels, lesion_center_of_mass):
        analysis_results[f'{int(label)}'] = {
            'center_of_mass': CoM}
        lesion_volume = np.sum(lesion_data == int(label)) * np.prod(resolution)
        analysis_results[f'{int(label)}']['volume_mm3'] = lesion_volume
        # We also compute lesion maximum diameter in each plane
        lesion_mask = (lesion_data == int(label))
        coords = np.argwhere(lesion_mask)
        # Calculate min and max coordinates along each axis
        min_bounds = coords.min(axis=0)
        max_bounds = coords.max(axis=0)
        # Calculate extent in voxels (max - min + 1)
        # We add 1 because if a lesion is at index 10, max=10, min=10, but length is 1 voxel
        extent_voxels = max_bounds - min_bounds + 1
        # Convert extent to millimeters
        extent_mm = extent_voxels * np.array(resolution)
        # Store dimensions based on anatomical axes determined earlier
        analysis_results[f'{int(label)}']['diameter_RL_mm'] = extent_mm[RL_axis]
        analysis_results[f'{int(label)}']['diameter_AP_mm'] = extent_mm[AP_axis]
        analysis_results[f'{int(label)}']['diameter_SI_mm'] = extent_mm[SI_axis]

    # Release memory
    del lesion_data

    return analysis_results


def compute_theta_angle(CoM, z_value, CoM_closest_centerline_point, lbl_centerline, levels, resolution):
    """
    This function computes the theta angle of a lesion center of mass relative to the spinal cord anatomy.
    The angle is computed as the angle from the centerline-centerspine plane: 0 degree is anterior, 90 right, 180 posterior, 270 left.
    The A-P direction is defined as the average direction of the 3 direction at 3 levels: above, at, and below the lesion z value.
    
    Inputs:
        CoM : center of mass coordinates
        z_value : centerline z value of the lesion
        lbl_centerline : labeled centerline data
        levels : vertebral levels
        resolution : image resolution

    Outputs:
        theta : angle from centerline-centerspine plane
    """
    # We first need to define the anterior-posterior direction as the centerline-centerspine plane
    ## We find the axis which first better the centerline and the centerspine points.
    lbl_centerline_data = nib.load(lbl_centerline).get_fdata()
    levels_data = nib.load(levels).get_fdata()
    centerline_coords = np.argwhere(lbl_centerline_data > 0)
    ## We find the 3 closest levels to the lesion z value: one above, one at, and one below
    at_level = int(round(z_value))
    above_level = at_level - 1 if at_level - 1 >= np.min(levels_data[levels_data > 0]) else at_level
    below_level = at_level + 1 if at_level + 1 <= np.max(levels_data[levels_data > 0]) else at_level
    ## For each level of the centerspine, we get the vector from centerline to centerspine
    vectors = []
    for level in [above_level, at_level, below_level]:
        # Get the coordinates of the current level
        level_coord = np.argwhere(levels_data == level)
        # We find the closest point in the centerline
        distances_centerline = np.linalg.norm((centerline_coords - level_coord) * resolution, axis=1)
        closest_centerline_idx = np.argmin(distances_centerline)
        closest_centerline_point = centerline_coords[closest_centerline_idx]
        # Compute the vector from centerline to centerspine
        vector = (level_coord - closest_centerline_point ) * resolution
        vectors.append(vector)
    ## We average the vectors to get a mean anterior-posterior direction
    mean_vector = np.mean(vectors, axis=0).flatten()

    # Now we can compute the theta angle of the lesion CoM
    direction_lesion = (np.array(CoM) - CoM_closest_centerline_point) * resolution
    ## Compute angle in degrees between mean_vector and direction_lesion
    theta = np.arctan2(np.linalg.norm(np.cross(mean_vector, direction_lesion)), np.dot(mean_vector, direction_lesion))
    theta = np.degrees(theta)

    # Release memory
    del lbl_centerline_data, levels_data, centerline_coords, vectors, mean_vector,

    return theta


def compute_lesion_location(lesion_analysis, lesion_seg, sc_seg, lbl_centerline, levels):
    """
    This function computes the lesion location relative to spinal cord anatomy.
    The center of mass coordinates in this new space used the following parameters:
        -z: along the spinal cord centerline: if z = 1.5 -> lesion is located exactly between level C1 and C2
        -r: radial distance from the centerline (in mm)
        -theta: angle from centerline-centerspine plane: 0 degree is anterior, 90 right, 180 posterior, 270 left

    Inputs:
        lesion_analysis : results of the lesion analysis
        lesion_seg : path to the lesion segmentation
        sc_seg : path to the spinal cord segmentation
        lbl_centerline : path to the labeled centerline
        levels : path to the vertebral levels

    Outputs:
        updated_lesion_analysis : updated results of the lesion analysis with location information
    """
    # Load the data necessary for the computation
    lesion_data = nib.load(lesion_seg).get_fdata()
    lbl_centerline_data = nib.load(lbl_centerline).get_fdata()

    # Get the resolution
    resolution = nib.load(lesion_seg).header.get_zooms()

    # iterate over each lesion:
    for lesion_id, lesion_data in lesion_analysis.items():
        CoM = tuple(map(float, lesion_data['center_of_mass']))

        # Compute the z value: the closest point on the centerline from the center of mass
        ## Get all centerline coordinates
        centerline_coords = np.argwhere(lbl_centerline_data > 0)
        ## Compute distances from CoM to all centerline voxels
        distances = np.linalg.norm((centerline_coords - CoM) * resolution, axis=1)
        ## Find the closest centerline voxel
        closest_idx = np.argmin(distances)
        closest_centerline_point = centerline_coords[closest_idx]
        z_value = lbl_centerline_data[tuple(closest_centerline_point)]
        lesion_analysis[lesion_id]['centerline_z'] = z_value

        # Compute the distance from the centerline (r) and angle (theta)
        r_value = distances[closest_idx] # in mm
        lesion_analysis[lesion_id]['radius_mm'] = r_value

        # To compute theta, we need to define the anterior-posterior direction
        theta = compute_theta_angle(CoM, z_value, closest_centerline_point, lbl_centerline, levels, resolution)
        lesion_analysis[lesion_id]['theta'] = theta

    # Release memory
    del lesion_data, lbl_centerline_data

    return lesion_analysis


def lesion_matching(lesion_analysis_1, lesion_analysis_2, w_z_over_disk=25.0):
    """
    This function performs lesion matching between two timepoints based on lesion location.

    Inputs:
        lesion_analysis_1 : results of the lesion analysis at timepoint 1
        lesion_analysis_2 : results of the lesion analysis at timepoint 2
        w_z_over_disk : weight for the z-axis distance relative to the disk plane

    Outputs:
        matched_lesions : dictionary containing matched lesions between the two timepoints
    """
    # We build the Hungarian matrix with distances between lesions based on location (z, r, theta)
    hungarian_matrix = np.zeros((len(lesion_analysis_1), len(lesion_analysis_2)))

    # We define weights for the different components of the distance
    w_z = w_z_over_disk
    w_disk = 1.0

    for i, lesion_1 in enumerate(lesion_analysis_1.values()):
        for j, lesion_2 in enumerate(lesion_analysis_2.values()):
            # Compute distance between lesion_1 and lesion_2 based on (z, r, theta)
            distance_disk = np.sqrt(lesion_1['radius_mm']**2 + lesion_2['radius_mm']**2 
                                    - 2 * lesion_1['radius_mm'] * lesion_2['radius_mm'] * np.cos(np.radians(lesion_1['theta'] - lesion_2['theta'])))
            z_dist = lesion_1['centerline_z'] - lesion_2['centerline_z']
            hungarian_matrix[i, j] = np.sqrt(w_z * z_dist**2 + w_disk * distance_disk**2)

    # We perform the Hungarian algorithm to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(hungarian_matrix)

    # We compute the lesion mapping based dictionary
    lesion_mapping = {}
    for i, j in zip(row_ind, col_ind):
        lesion_id_1 = list(lesion_analysis_1.keys())[i]
        lesion_id_2 = list(lesion_analysis_2.keys())[j]
        lesion_mapping[lesion_id_1] = [int(i) for i in lesion_id_2]

    return lesion_mapping


def map_lesions_unregistered(input_image1, input_image2, pred_folder, output_folder, GT_lesion=False, w_z_over_disk=25.0):
    """
    This function performs lesion mapping between two timepoints.

    Inputs:
        input_image1 : path to the input image at timepoint 1
        input_image2 : path to the input image at timepoint 2
        output_folder : path to the output folder where comparison results will be stored

    Outputs:
        None
    """
    # Build output directory
    os.makedirs(output_folder, exist_ok=True)
    # Build temporary directory
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    # Build QC folder
    qc_folder = os.path.join(output_folder, "qc")
    os.makedirs(qc_folder, exist_ok=True)

    # Build logger
    logger.add(os.path.join(output_folder, f'logger_{str(date.today())}.log'))

    # Initialize file names for lesions, sc and disc levels at both timepoints
    image_1_name = Path(input_image1).name
    image_2_name = Path(input_image2).name
    # Initialize the names
    sc_seg_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_sc-seg.nii.gz'))
    sc_seg_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_sc-seg.nii.gz'))
    centerline_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_centerline.nii.gz'))
    centerline_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_centerline.nii.gz'))
    lesion_seg_1 =  os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    lesion_seg_2 =  os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg.nii.gz'))
    levels_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_levels.nii.gz'))
    levels_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_levels.nii.gz'))
    registered_image2_to_1 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_registered_to_' + image_1_name))
    warping_field_img2_to_1 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_warp_to_' + image_1_name))
    inv_warping_field_img2_to_1 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_inv_warp_to_' + image_1_name))
    lesion_seg_2_reg = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg-reg.nii.gz'))
    labeled_lesion_seg_1 = os.path.join(pred_folder, image_1_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
    labeled_lesion_seg_2 = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
    labeled_lesion_seg_2_reg = os.path.join(pred_folder, image_2_name.replace('.nii.gz', '_lesion-seg-reg-labeled.nii.gz'))


    # For each timepoint, we analyze lesions
    lesion_analysis_1 = analyze_lesions(labeled_lesion_seg_1)
    lesion_analysis_2 = analyze_lesions(labeled_lesion_seg_2)

    # We label the centerline so that it takes continous various following the disc levels
    labeled_centerline_1 = os.path.join(temp_folder, image_1_name.replace('.nii.gz', '_labeled-centerline.nii.gz'))
    labeled_centerline_2 = os.path.join(temp_folder, image_2_name.replace('.nii.gz', '_labeled-centerline.nii.gz'))
    label_centerline(centerline_1, levels_1, labeled_centerline_1)
    label_centerline(centerline_2, levels_2, labeled_centerline_2)

    # We compute lesion location relative to spinal cord anatomy
    lesion_analysis_1 = compute_lesion_location(lesion_analysis_1, lesion_seg_1, sc_seg_1, labeled_centerline_1, levels_1)
    lesion_analysis_2 = compute_lesion_location(lesion_analysis_2, lesion_seg_2, sc_seg_2, labeled_centerline_2, levels_2)

    # We perform lesion matching between timepoint 1 and timepoint 2 based on location
    lesion_mapping = lesion_matching(lesion_analysis_1, lesion_analysis_2, w_z_over_disk=w_z_over_disk)
    logger.info(f"Lesion matching: {lesion_mapping}")

    # # Remove the temporary folder
    # assert os.system(f'rm -rf {temp_folder}') == 0, "Failed to remove temporary folder"

    return lesion_mapping


def main():
    args = parse_args()
    input_image1 = args.input_image1
    input_image2 = args.input_image2
    pred_folder = args.pred_folder
    output_folder = args.output_folder
    w_z_over_disk = args.w_z_over_disk

    lesion_mapping = map_lesions_unregistered(input_image1, input_image2, pred_folder, output_folder, w_z_over_disk=w_z_over_disk)

    return None


if __name__ == "__main__":
    main()