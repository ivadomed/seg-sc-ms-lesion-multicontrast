"""
This script is used to convert the WMn data to the nnUNet format and add them to the an existing dataset. 
This way we avoid having to re-convert all 4000 images and only haev to do it with WMn images.

Input:
    -nnnunet-data: Path to existing nnUNet dataset where the data will be copied (e.g. nnUNet_raw/TaskXXX_MyDataset)
    -wmn-data: Path to WMn data
    -with-hc: Whether to include healthy controls in the dataset (default: False)

Author: Pierre-Louis Benveniste
"""
import os
import shutil
import argparse
from pathlib import Path
import nibabel as nib
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert WMn data to nnUNet format and add to existing dataset")
    parser.add_argument("--nnunet-data", type=str, required=True, help="Path to existing nnUNet dataset (e.g. nnUNet_raw/TaskXXX_MyDataset)")
    parser.add_argument("--wmn-data", type=str, required=True, help="Path to WMn data")
    parser.add_argument("--with-hc", action="store_true", help="Whether to include healthy controls in the dataset (default: False)")
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    nnunet_data = args.nnunet_data
    wmn_data = args.wmn_data

    # If the output directory does not exist, create it
    os.makedirs(nnunet_data, exist_ok=True)

    # Get all files from WMn data
    list_derivatives = list(Path(wmn_data).rglob("*label-lesion_seg.nii.gz"))
    list_derivatives = [str(path) for path in list_derivatives]

    # Get all WMn images
    list_images = list(Path(wmn_data).rglob("*WMn.nii.gz"))
    list_images = [str(path) for path in list_images]

    image_pairs = {}
    # For each image we keep only the derivative with the highest rater number (e.g. desc-rater4_label-lesion_seg.nii.gz should be kept over desc-rater1_label-lesion_seg.nii.gz)
    for image in tqdm(list_images):
        image_name = image.split('/')[-1].replace(".nii.gz", "")
        corresponding_derivative = [derivative for derivative in list_derivatives if image_name in derivative]
        # Only the derivative with the highest rater number should be kept (e.g. desc-rater4_label-lesion_seg.nii.gz should be kept over desc-rater1_label-lesion_seg.nii.gz)
        derivatives_to_remove = sorted(corresponding_derivative, key=lambda x: int(x.split('rater')[1].split('_')[0]), reverse=True)
        image_pairs[image] = derivatives_to_remove[0]

    # Remove variable list_derivatives
    del list_derivatives
    del list_images

    print(f"Found {len(image_pairs)} WMn images in the WMn dataset")

    # remove images where sub-XXX greater or equal to 099
    if not args.with_hc:
        image_pairs = {image: derivative for image, derivative in image_pairs.items() if int(image.split('/')[-1].split('_')[0].split('sub-')[1]) < 99}
        
    print(f"Found {len(image_pairs)} WMn images to convert and add to nnUNet dataset")
    
    # Initialize values of each data split
    count_Tr = 3926
    count_Ts = 433

    list_subjects = []
    for sub in image_pairs.keys():
        subject_id = sub.split('/')[-1].split('_')[0]
        list_subjects.append(subject_id)

    # Split the subjects into train and test sets 85/15
    list_subjects_train = list_subjects[:int(0.85 * len(list_subjects))]
    list_subjects_test = list_subjects[int(0.85 * len(list_subjects)):]

    # Load the conversion dict in the existing nnUNet dataset if it exists, otherwise create an empty dict
    conversion_dict_path = os.path.join(nnunet_data, "conversion_dict.json")
    # Load the conversion dict
    with open(conversion_dict_path, "r") as f:
        conversion_dict = json.load(f)

    # Load the dataset.json file in the existing nnUNet dataset
    dataset_json_path = os.path.join(nnunet_data, "dataset.json")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)

    # Build output folders
    path_out_imagesTr = os.path.join(nnunet_data, 'imagesTr')
    path_out_labelsTr = os.path.join(nnunet_data, 'labelsTr')
    path_out_imagesTs = os.path.join(nnunet_data, 'imagesTs')
    path_out_labelsTs = os.path.join(nnunet_data, 'labelsTs')
    os.makedirs(path_out_imagesTr, exist_ok=True)
    os.makedirs(path_out_labelsTr, exist_ok=True)
    os.makedirs(path_out_imagesTs, exist_ok=True)
    os.makedirs(path_out_labelsTs, exist_ok=True)

    # Iterate over the derivatives and move them to the nnunet dataset
    for image_file in tqdm(image_pairs):
        subject_id = image_file.split('/')[-1].split('_')[0]
        print(f"Processing subject {subject_id} with path {image_file}")
        # Find corresponding seg file
        seg_file = image_pairs[image_file]
    
        if subject_id in list_subjects_train:
            count_Tr +=1
            image_file_nnunet = os.path.join(path_out_imagesTr,f'Dataset903_msLesionAgnostic_{count_Tr:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTr,f'Dataset903_msLesionAgnostic_{count_Tr:03d}.nii.gz')

            # Reorient the image
            assert os.system(f"sct_image -i {image_file} -setorient RPI -o {image_file_nnunet}") ==0

            # Binarize the label and save it to the adequate path
            label = nib.load(seg_file).get_fdata()
            label[label > 0] = 1
            label = nib.Nifti1Image(label, nib.load(seg_file).affine)
            nib.save(label, label_file_nnunet)
            # Then we reorient the label to RPI
            assert os.system(f"sct_image -i {label_file_nnunet} -setorient RPI -o {label_file_nnunet}") ==0

            # For each label fils, we reorient them to the same orientation as the image using sct_register_multimodal -identity 1
            assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") ==0
            # Remove the other useless files
            assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") ==0
            other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
            assert os.system(f"rm {other_file_to_remove}") ==0

            # Then we binarize the label
            assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") ==0

            # Add the file to the dataset.json file
            dataset_json['training'].append({
                "image": image_file_nnunet,
                "label": label_file_nnunet
            })

        else:
            count_Ts +=1
            image_file_nnunet = os.path.join(path_out_imagesTs,f'Dataset903_msLesionAgnostic_{count_Ts:03d}_0000.nii.gz')
            label_file_nnunet = os.path.join(path_out_labelsTs,f'Dataset903_msLesionAgnostic_{count_Ts:03d}.nii.gz')

            # Reorient the image
            assert os.system(f"sct_image -i {image_file} -setorient RPI -o {image_file_nnunet}") ==0

            # Binarize the label and save it to the adequate path
            label = nib.load(seg_file).get_fdata()
            label[label > 0] = 1
            label = nib.Nifti1Image(label, nib.load(seg_file).affine)
            nib.save(label, label_file_nnunet)
            # Then we reorient the label to RPI
            assert os.system(f"sct_image -i {label_file_nnunet} -setorient RPI -o {label_file_nnunet}") ==0

            # For each label fils, we reorient them to the same orientation as the image using sct_register_multimodal -identity 1
            assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") ==0
            # Remove the other useless files
            assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") ==0
            other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
            assert os.system(f"rm {other_file_to_remove}") ==0

            # Then we binarize the label
            assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") ==0

             # Add the file to the dataset.json file
            dataset_json['test'].append({
                "image": image_file_nnunet,
                "label": label_file_nnunet
            })

        # Update the conversion dict
        conversion_dict[image_file] = image_file_nnunet
        conversion_dict[seg_file] = label_file_nnunet
    
    # Update the dataset.json files count values
    dataset_json['numTraining'] = count_Tr
    dataset_json['numTest'] = count_Ts

    # Save the updated dataset.json file
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    # Save the updated conversion dict
    with open(conversion_dict_path, "w") as f:
        json.dump(conversion_dict, f, indent=4)

    return None

    
if __name__ == "__main__":
    main()