"""
This script performs temperature scaling on the soft segmentations to calibrate the predicted probabilities.

Input:
    --msd: Path to the MSD JSON file describing the dataset
    --model-path: Path to the nnUNet model to use for prediction
    -o : Folder to save the calibrated predictions
    --smaller-changes: If specified, only apply small downsampling changes

Author: Pierre-Louis Benveniste
"""
import argparse
import importlib
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json

os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

from tqdm import tqdm
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor   # noqa: E402
import torch
import importlib
from image import Image, get_orientation
import numpy as np
import torch.nn.functional as F
import pandas as pd
import nilearn.image


def parse_args():
    parser = argparse.ArgumentParser(description="Perform temperature scaling on soft segmentations.")
    parser.add_argument("--msd", required=True, type=str, help="Path to the MSD JSON file describing the dataset")
    parser.add_argument("--model-path", required=True, type=str, help="Path to the nnUNet model to use for prediction")
    parser.add_argument("-o", required=True, type=str, help="Folder to save the calibrated predictions")
    parser.add_argument("--smaller-changes", action="store_true", help="If specified, only apply small downsampling changes")
    return parser.parse_args()


def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def load_trainer_class_if_available(path_model):
    """
    This functions load a custom nnUNet trainer class if available in the model folder.
    """
    trainer_file = os.path.join(path_model, "trainer_class.py")
    if os.path.exists(trainer_file):
        spec = importlib.util.spec_from_file_location("trainer_class", trainer_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "get_trainer_class"):
            return module.get_trainer_class()
    return None


def initialize_predictor(path_model, five_folds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We load the model
    folds = [2]
    if five_folds:
        folds = [0, 1, 2, 3, 4]
    predictor = nnUNetPredictor(
        tile_step_size=0.5,  # changing it from 0.5 to 0.9 makes inference faster
        use_gaussian=True,  # applies gaussian noise and gaussian blur
        use_mirroring=False,  # test time augmentation by mirroring on all axes
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    trainer_class = load_trainer_class_if_available(path_model)
    predictor.initialize_from_trained_model_folder(
        join(path_model),
        use_folds=folds,
        checkpoint_name='checkpoint_best.pth',
        trainer_class=trainer_class
    )
    return predictor


def resample_img(image, resamp_factor, interpolation='linear'):
    """
    Docstring for resample_nib
    
    :param image: input image in the Image class format
    :param resamp_factor: tuple of 3 floats, the downsampling factor for each dimension (x, y, z)
    :param interpolation: interpolation method to use for resampling, either 'linear' or 'nearest'
    """
    # We first change the resamp factor: it should be 1/value
    resamp_factor = [1/f for f in resamp_factor]

    start_orientation = get_orientation(image)
    # Convert image from Image class to nibabel format
    img = nib.Nifti1Image(image.data, image.hdr.get_best_affine(), image.hdr)

    # Convert the resampling factor to a list of 3 floats
    resamp_factor = list(resamp_factor)

    # Resample the image using nilearn
    resamp_img = nilearn.image.resample_img(img, target_affine=np.diag(resamp_factor), interpolation=interpolation)

    #Convert bacl to Image class
    resamp_img = Image(np.asanyarray(resamp_img.dataobj), hdr=resamp_img.header, orientation=image.orientation, dim=resamp_img.header.get_data_shape())
    resamp_img.change_orientation(start_orientation)

    return resamp_img


def main_temperature_scaling(input_msd, path_model, output_folder, smaller_changes=False):

    # Build the output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the msd file
    with open(input_msd, 'r') as f:
        msd_data = json.load(f)
    images = msd_data['images']

    images = [img["image"] for img in images]
    labels = [img["label"] for img in msd_data['images']]

    # Shuffle the images
    np.random.seed(43)
    np.random.shuffle(images)
    # Shuffle the labels in the same way
    np.random.seed(43)
    np.random.shuffle(labels)

    # Select images for temperature scaling
    calib_images = images[:200]
    calib_labels = labels[:200]
    eval_images = images[200:]
    eval_labels = labels[200:]

    # save the list of images used for calibration and evaluation in the output folder
    with open(os.path.join(output_folder, "calib_images.txt"), "w") as f:
        f.write("Calibration images:\n")
        for img in calib_images:
            f.write(f"{img}\n")
        f.write("\nEvaluation images:\n")
        for img in eval_images:
            f.write(f"{img}\n")

    # initialize the model
    predictor = initialize_predictor(path_model, five_folds=False)

    # Downsampling factors to evaluate
    downsampling_factors = [(0.9, 0.9, 0.9), (0.8, 0.8, 0.8), (0.7, 0.7, 0.7), (0.6, 0.6, 0.6), (0.5, 0.5, 0.5),
                             (0.9, 0.9, 1.0), (0.8, 0.8, 1.0), (0.7, 0.7, 1.0), (0.6, 0.6, 1.0), (0.5, 0.5, 1.0),
                             (0.9, 1.0, 0.9), (0.8, 1.0, 0.8), (0.7, 1.0, 0.7), (0.6, 1.0, 0.6), (0.5, 1.0, 0.5),
                             (1.0, 0.9, 0.9), (1.0, 0.8, 0.8), (1.0, 0.7, 0.7), (1.0, 0.6, 0.6), (1.0, 0.5, 0.5),
                             ]
    if smaller_changes:
        downsampling_factors = [(1.0, 1.0, 1.0), 
                                (0.975, 0.975, 0.975), (0.95, 0.95, 0.95), (0.925, 0.925, 0.925), (0.9, 0.9, 0.9),
                                (0.975, 0.975, 1.000), (0.95, 0.95, 1.00), (0.925, 0.925, 1.000), (0.9, 0.9, 1.0),
                                (0.975, 1.000, 0.975), (0.95, 1.00, 0.95), (0.925, 1.000, 0.925), (0.9, 1.0, 0.9),
                                (1.000, 0.975, 0.975), (1.00, 0.95, 0.95), (1.000, 0.925, 0.925), (1.0, 0.9, 0.9),
                                ]
    
    # Iterate to the the following once on the calibration set and once on the evaluation set, and save the results in a dataframe
    data = [zip(calib_images, calib_labels), zip(eval_images, eval_labels)]

    for i, dataset in enumerate(data):
        if i == 0:
            print("Processing calibration set...")
        else:
            print("Processing evaluation set...")
        imgs, labels = zip(*dataset)
        results_df = pd.DataFrame(columns=["subject", "resamp_factor", "dice_at_0.5"])

        for img, label in tqdm(zip(imgs, labels), desc="Running lesion segmentation"):
            print(f"Processing image: {img}")
            subject = img.split("/")[-1].replace(".nii.gz","")

            orig_orientation = get_orientation(Image(img))
            model_orientation = "RPI"

            # Reorient the image to model orientation if not already
            img_in = Image(img)
            label_in = Image(label)
            if orig_orientation != model_orientation:
                img_in.change_orientation(model_orientation)
                label_in.change_orientation(model_orientation)

            for down_factor in downsampling_factors:
                print(f"Downsampling factor: {down_factor}")
                # We downsample the image: factor 2 means going from 1mm to 2mm spacing
                resamp_img = resample_img(img_in, down_factor, interpolation='linear')

                # we downsample the label as well, but with nearest neighbor interpolation to avoid creating new classes
                resamp_label = resample_img(label_in, down_factor, interpolation='nearest')

                # prepare the image for prediction
                data = resamp_img.data.transpose([2, 1, 0])
                data = np.expand_dims(data, axis=0).astype(np.float32)

                pred = predictor.predict_single_npy_array(
                    input_image=data,
                    # The spacings also have to be reversed to match nnUNet's conventions.
                    image_properties={'spacing': resamp_img.dim[6:3:-1]},
                    # Save the probability maps if specified
                    save_or_return_probabilities=True,
                    # If using a model ensemble, return the logits per fold so we can average them ourselves
                    return_logits_per_fold=False, 
                    return_logits = True
                )
                _, prob_maps = pred
                pred_soft = np.where(prob_maps[1] > prob_maps[0], prob_maps[1], 0)
                pred_bin = np.where(pred_soft > 0.5, 1, 0)
                pred_soft = pred_soft.transpose([2, 1, 0])
                pred_bin = pred_bin.transpose([2, 1, 0])

                # Compute lesion volume
                voxel_volume = np.prod(resamp_img.dim[6:3:-1])  # in mm^3
                lesion_total_volume_soft = pred_soft.sum() * voxel_volume
                lesion_total_volume_binary = pred_bin.sum() * voxel_volume

                # Compute ground truth lesion volume
                gt_lesion_volume = (resamp_label.data > 0).sum() * voxel_volume
                # We compute the Dice score between the binary predictions and the ground truth label
                dice = dice_score(pred_bin, resamp_label.data.astype(np.uint8))

                # Add everything to the results dataframe
                new_line = pd.DataFrame({"subject": [subject], "resamp_factor": [down_factor], "dice": [dice],
                                            "lesion_total_volume_soft": [lesion_total_volume_soft], "lesion_total_volume_binary": [lesion_total_volume_binary],
                                            "gt_lesion_volume": [gt_lesion_volume]})
                results_df = pd.concat([results_df, new_line], ignore_index=True)

            # Save the results dataframe to csv
            if i == 0:
                results_df.to_csv(os.path.join(output_folder, "results_train.csv"), index=False)
                print(f"Saved results to {os.path.join(output_folder, 'results_train.csv')}")
            else:
                results_df.to_csv(os.path.join(output_folder, "results_eval.csv"), index=False)
                print(f"Saved results to {os.path.join(output_folder, 'results_eval.csv')}")

        # We analyze the output dataframe:
        ## We compute the lesion volume std for soft and binary predictions
        lesion_volume_std_soft = results_df.groupby("subject")["lesion_total_volume_soft"].std()
        lesion_volume_std_binary = results_df.groupby("subject")["lesion_total_volume_binary"].std()
        ## We comput the coefficient of variation (std/mean) for soft and binary predictions
        lesion_volume_cv_soft = lesion_volume_std_soft / results_df.groupby("subject")["lesion_total_volume_soft"].mean()
        lesion_volume_cv_binary = lesion_volume_std_binary / results_df.groupby("subject")["lesion_total_volume_binary"].mean()
        ## Compute overall Dice score
        overall_dice = results_df["dice"].mean()
        # Write a text file summarizing the results
        text_to_write = f"Overall Dice score across all subjects and downsampling factors: {overall_dice}\n"
        text_to_write += f"Lesion volume std for soft predictions: {lesion_volume_std_soft.mean()}\n"
        text_to_write += f"Lesion volume std for binary predictions: {lesion_volume_std_binary.mean()}\n"
        text_to_write += f"Lesion volume coefficient of variation for soft predictions: {lesion_volume_cv_soft.mean()}\n"
        text_to_write += f"Lesion volume coefficient of variation for binary predictions: {lesion_volume_cv_binary.mean()}\n"
        if i == 0:
            out_txt_path = os.path.join(output_folder, "summary_results_train.txt")
        else:
            out_txt_path = os.path.join(output_folder, "summary_results_eval.txt")
        with open(out_txt_path, "w") as f:
            f.write(text_to_write)
        print(f"Saved summary results to {out_txt_path}")


if __name__ == "__main__":
    args = parse_args()
    main_temperature_scaling(args.msd, args.model_path, args.o, args.smaller_changes)