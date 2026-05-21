"""
This file runs predicted lesion segmentation on all PSIR and STIR images in the canproco set when the sub has 2 timepoint.
It makes sure to segment in order the tiempoint M0 then the timepoint M12.

Input:
    --d: path to the canproco dataset
    --exclude: yml file containing the list of subjects to exclude from the analysis (because they are already in the project)
    --o: path to the output directory where the predicted segmentation will be saved

Author: Pierre-Louis Benveniste
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Run predicted lesion segmentation on canproco set")
    parser.add_argument("--d", type=str, required=True, help="Path to the canproco dataset")
    parser.add_argument("--exclude", type=str, required=True, help="YML file containing the list of subjects to exclude from the analysis (because they are already in the project)")
    parser.add_argument("--o", type=str, required=True, help="Path to the output directory where the predicted segmentation will be saved")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.d)
    output_path = Path(args.o)

    # Build output path 
    os.makedirs(output_path, exist_ok=True)
    qc_folder = os.path.join(output_path, "QC")
    os.makedirs(qc_folder, exist_ok=True)

    # Load the list of subjects to exclude if provided
    with open(args.exclude, 'r') as f:
        exclude_dict = yaml.safe_load(f)
    exclude_list = exclude_dict["include"]

    # Change the list from images to only subjects
    exclude_subjects = set()
    for img in exclude_list:
        sub = img.split('_')[0]  # Assuming the subject ID is the first part of the filename
        exclude_subjects.add(sub)

    # List all PSIR and STIR images in the dataset
    psir_images = list(dataset_path.rglob("*PSIR.nii.gz"))
    stir_images = list(dataset_path.rglob("*STIR.nii.gz"))
    images = psir_images + stir_images
    images = [str(img) for img in images]
    images = [img for img in images if 'SHA256' not in img]
    # Exclude images from subjects in the exclude list
    images = [img for img in images if img.split('/')[-1].split('_')[0] not in exclude_subjects]
    images = sorted(images)  # Sort images to ensure consistent order
    print(f"Number of PSIR and STIR images in the dataset after exclusion: {len(images)}")

    # Group images by subject and timepoint
    images_by_subject = {}
    for img in images:
        sub = img.split('/')[-1].split('_')[0]  # Assuming the subject ID is the first part of the filename
        timepoint = img.split('/')[-1].split('_')[1]  # Assuming the timepoint is the second part of the filename
        if sub not in images_by_subject:
            images_by_subject[sub] = {}
        images_by_subject[sub][timepoint] = img
    
    # Run predicted lesion segmentation on each subject's images in order of timepoint
    for sub, timepoints in tqdm(images_by_subject.items(), desc="Processing subjects"):
        if 'ses-M0' in timepoints and 'ses-M12' in timepoints:
            if "sub-cal123" in sub:
                print(f"Skipping subject {sub} because problematic")
                continue
            # Build sub output folder
            sub_output_folder = os.path.join(output_path, sub)
            os.makedirs(sub_output_folder, exist_ok=True)

            # Segment the spinal cord in the M0 timepoint image
            sc_seg_path = os.path.join(sub_output_folder, f"{sub}_ses-M0_sc-seg.nii.gz")
            if not os.path.exists(sc_seg_path):
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {timepoints['ses-M0']} -o {sc_seg_path}") == 0, f"Error segmenting spinal cord for subject {sub} at timepoint M0"
            # Segment the lesion in the M0 timepoint image using the predicted lesion segmentation model
            lesion_seg_path = os.path.join(sub_output_folder, f"{sub}_ses-M0_lesion-seg.nii.gz")
            if not os.path.exists(lesion_seg_path):
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {timepoints['ses-M0']} -o {lesion_seg_path} -test-time-aug -qc {qc_folder} -qc-seg {sc_seg_path} -qc-plane Sagittal") == 0, f"Error segmenting lesion for subject {sub} at timepoint M0"

            # Segment the spinal cord in the M12 timepoint image
            sc_seg_path = os.path.join(sub_output_folder, f"{sub}_ses-M12_sc-seg.nii.gz")
            if not os.path.exists(sc_seg_path):
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {timepoints['ses-M12']} -o {sc_seg_path}") == 0, f"Error segmenting spinal cord for subject {sub} at timepoint M12"
            # Segment the lesion in the M12 timepoint image using the predicted lesion segmentation model
            lesion_seg_path = os.path.join(sub_output_folder, f"{sub}_ses-M12_lesion-seg.nii.gz")
            if not os.path.exists(lesion_seg_path):
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {timepoints['ses-M12']} -o {lesion_seg_path} -test-time-aug -qc {qc_folder} -qc-seg {sc_seg_path} -qc-plane Sagittal") == 0, f"Error segmenting lesion for subject {sub} at timepoint M12"

    print("All subjects processed successfully.")


if __name__ == "__main__":
    main()