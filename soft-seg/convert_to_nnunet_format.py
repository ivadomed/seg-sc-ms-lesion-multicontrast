"""
This script converts the files from the msd dataset to the nnunet format.

Input:
    -i: Path to input msd file
    -o: Path to output nnunet formatted folder

Author: Pierre-Louis Benveniste
"""
import os
import shutil
import argparse
import json
from pathlib import Path
import tqdm
import nibabel as nib
import numpy as np
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MSD dataset to nnU-Net format')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the MSD dataset json file')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    input_msd = args.input
    output_dir = args.output

    # Load the MSD dataset
    with open(input_msd, 'r') as f:
        msd_data = json.load(f)

    # Define the output path
    path_out = Path(os.path.join(output_dir, 'Dataset101_downsamplingExp'))
    
    # Define paths for train and test folders 
    path_out_images = Path(os.path.join(path_out, 'images'))
    path_out_labels = Path(os.path.join(path_out, 'labels'))

    # Load both train and validation set into the train images as nnunet uses cross-fold-validation
    images, labels = [], []

    # Make the directories
    path_out.mkdir(parents=True, exist_ok=True)
    path_out_images.mkdir(parents=True, exist_ok=True)
    path_out_labels.mkdir(parents=True, exist_ok=True)

    # Initialise the conversion dict
    conversion_dict = {}

    # Initialise the number of scans in train and in test folder
    scan_cnt = 0
    
    # Load the json file
    data = msd_data["images"]

    # Iterate over all  training images
    for img_dict in tqdm.tqdm(data):

        scan_cnt += 1

        image_file_nnunet = os.path.join(path_out_images,f'downsamplingExp_{scan_cnt:03d}_0000.nii.gz')
        label_file_nnunet = os.path.join(path_out_labels,f'downsamplingExp_{scan_cnt:03d}.nii.gz')
        
        images.append(str(image_file_nnunet))
        labels.append(str(label_file_nnunet))

        # Instead of copying we will reorient the image to RPI
        assert os.system(f"sct_image -i {img_dict['image']} -setorient RPI -o {image_file_nnunet}") ==0

        # Binarize the label and save it to the adequate path
        label = nib.load(img_dict['label']).get_fdata()
        label[label > 0] = 1
        label = nib.Nifti1Image(label, nib.load(img_dict['label']).affine)
        nib.save(label, label_file_nnunet)
        # Then we reorient the label to RPI
        assert os.system(f"sct_image -i {label_file_nnunet} -setorient RPI -o {label_file_nnunet}") ==0

        # Update the conversion dict
        conversion_dict[str(os.path.abspath(img_dict['image']))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(img_dict['label']))] = label_file_nnunet

    #----------------- CREATION OF THE DICTIONNARY-----------------------------------
    # create dataset_description.json
    json_object = json.dumps(conversion_dict, indent=4)
    # write to dataset description
    conversion_dict_name = f"conversion_dict.json"
    with open(os.path.join(path_out, conversion_dict_name), "w") as outfile:
        outfile.write(json_object)

    return None


if __name__ == '__main__':
    main()







    

    

