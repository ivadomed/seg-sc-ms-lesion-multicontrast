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

    print(f"Found {len(list_derivatives)} WMn images in the WMn dataset")

    # remove images where sub-XXX greater or equal to 099
    if not args.with_hc:
        list_derivatives = [path for path in list_derivatives if int(path.split('/')[-1].split('_')[0].replace("sub-", "")) < 99]

    print(f"Found {len(list_derivatives)} WMn images to convert and add to nnUNet dataset")
    # Initialize values of each data split
    count_Tr = 3926
    count_Ts = 433

    list_subjects = []
    for path in list_derivatives:
        subject_id = path.split('/')[-1].split('_')[0]
        list_subjects.append(subject_id)

    # Split the subjects into train and test sets 85/15
    list_subjects_train = list_subjects[:int(0.85 * len(list_subjects))]
    list_subjects_test = list_subjects[int(0.85 * len(list_subjects)):]

    # Load the conversion dict in the existing nnUNet dataset if it exists, otherwise create an empty dict
    conversion_dict_path = os.path.join(nnunet_data, "conversion_dict.json")

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
    for seg_file in list_derivatives:
        subject_id = seg_file.split('/')[-1].split('_')[0]
        print(f"Processing subject {subject_id} with path {seg_file}")
        # Find corresponding image file
        image_file = seg_file.replace("_label-lesion_seg.nii.gz", ".nii.gz").replace("derivatives/labels/", "")
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

        # Update the conversion dict
        conversion_dict[str(os.path.abspath(img_dict['image']))] = image_file_nnunet
        conversion_dict[str(os.path.abspath(img_dict['label']))] = label_file_nnunet

        # For each label fils, we reorient them to the same orientation as the image using sct_register_multimodal -identity 1
        assert os.system(f"sct_register_multimodal -i {str(label_file_nnunet)} -d {str(image_file_nnunet)} -identity 1 -o {str(label_file_nnunet)} -owarp file_to_delete.nii.gz -owarpinv file_to_delete_2.nii.gz ") ==0
        # Remove the other useless files
        assert os.system("rm file_to_delete.nii.gz file_to_delete_2.nii.gz") ==0
        other_file_to_remove = str(label_file_nnunet).replace('.nii.gz', '_inv.nii.gz')
        assert os.system(f"rm {other_file_to_remove}") ==0

        # Then we binarize the label
        assert os.system(f"sct_maths -i {str(label_file_nnunet)} -bin 0.5 -o {str(label_file_nnunet)}") ==0

        break

    # TODO: do the same for the test set and test the above (issue with the conversion dict)

            
        # elif subject_id in list_subjects_test:

    # 



    
if __name__ == "__main__":
    main()