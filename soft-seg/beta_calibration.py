"""
Beta Calibration for Soft Segmentation

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
import pickle
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
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


def parse_args():
    parser = argparse.ArgumentParser(description="Perform beta calibration on soft segmentations.")
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


def main_beta_calibration(input_msd, path_model, output_folder, smaller_changes=False):

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

    # Initializ the calib values
    all_calib_probs = []
    all_calib_labels = []

    for img, label in tqdm(zip(calib_images, calib_labels), desc="Running lesion segmentation"):
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
        
        data = img_in.data.transpose([2, 1, 0])
        label_data = label_in.data.transpose([2, 1, 0])
        data = np.expand_dims(data, axis=0).astype(np.float32)
        label_data = np.expand_dims(label_data, axis=0).astype(np.float32)

        _, prob_maps = predictor.predict_single_npy_array(
            input_image=data,
            # The spacings also have to be reversed to match nnUNet's conventions.
            image_properties={'spacing': img_in.dim[6:3:-1]},
            # Save the probability maps if specified
            save_or_return_probabilities=True,
            # If using a model ensemble, return the logits per fold so we can average them ourselves
            return_logits_per_fold=False, 
            return_logits = False
        )
        probs = np.where(prob_maps[1] > prob_maps[0], prob_maps[1], 0)

        # Add the probs and labels to the calib values
        all_calib_probs.append(probs.flatten())
        all_calib_labels.append(label_data.flatten())
    
    # Concatenate all the calib values
    all_calib_probs = np.concatenate(all_calib_probs)
    all_calib_labels = np.concatenate(all_calib_labels)
    print(f"Number of calib values: {len(all_calib_probs)}")

    # 1. Transform scores into log-space features
    # We add a small epsilon to avoid log(0) or log(1) issues
    eps = np.finfo(float).eps
    all_calib_probs = np.clip(all_calib_probs, eps, 1 - eps)
    
    s_prime = np.log(all_calib_probs)                 # ln(s) 
    s_double_prime = -np.log(1 - all_calib_probs)      # -ln(1-s) 
    
    # 2. Reshape for bivariate logistic regression [cite: 322, 332]
    X = np.column_stack((s_prime, s_double_prime))
    
    # 3. Fit Logistic Regression [cite: 230, 291]
    # Note: If you want to strictly enforce monotonicity, 
    # you must ensure coefficients a and b are >= 0[cite: 249, 336].
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(X, all_calib_labels)

    # save the isotonic regression model using joblib
    with open(os.path.join(output_folder, 'beta_calib_model.pkl'),'wb') as f:
        pickle.dump(lr, f)
    
    a = lr.coef_[0][0]
    b = lr.coef_[0][1]
    c = lr.intercept_[0]

    print(f"Fitted logistic regression coefficients: a={a}, b={b}, c={c}")
    results_df = pd.DataFrame(columns=["subject", "resamp_factor", "dice_before_calib", "dice_after_calib",
                                       "soft_lesion_volume_after_calib", "soft_lesion_volume_before_calib",
                                       "bin_lesion_volume_after_calib", "bin_lesion_volume_before_calib"])
    
    # Initialize all eval calib values
    all_eval_calib_probs = []
    all_eval_probs = []
    all_eval_labels = []

    # Now we apply this to the images of the evaluation set, and save the calibrated probabilities
    for img, label in tqdm(zip(eval_images, eval_labels), desc="Running lesion segmentation on evaluation set"):
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
        
            data = resamp_img.data.transpose([2, 1, 0])
            label_data = resamp_label.data.transpose([2, 1, 0])
            data = np.expand_dims(data, axis=0).astype(np.float32)
            label_data = np.expand_dims(label_data, axis=0).astype(np.float32)

            _, prob_maps = predictor.predict_single_npy_array(
                input_image=data,
                # The spacings also have to be reversed to match nnUNet's conventions.
                image_properties={'spacing': resamp_img.dim[6:3:-1]},
                # Save the probability maps if specified
                save_or_return_probabilities=True,
                # If using a model ensemble, return the logits per fold so we can average them ourselves
                return_logits_per_fold=False, 
                return_logits = False
            )
            probs = np.where(prob_maps[1] > prob_maps[0], prob_maps[1], 0)


            # We also save the non-calibrated probabilities for evaluation purposes
            all_eval_probs.append(probs.flatten())
            all_eval_labels.append(label_data.flatten())

            # Compute Dice score and lesion volume before beta calibration
            binary_prediction_before = np.where(probs > 1e-3, 1, 0)
            dice_before = dice_score(binary_prediction_before, label_data)
            voxel_volume = np.prod(resamp_img.dim[6:3:-1])  # Get
            soft_lesion_volume_before = probs.sum() * voxel_volume
            bin_lesion_volume_before = binary_prediction_before.sum() * voxel_volume

            # # Perform beta calibration using the fitted logistic regression coefficients
            eps = np.finfo(float).eps
            probs = np.clip(probs, eps, 1 - eps)
            
            # Formula: 1 / (1 + 1 / (exp(c) * (s^a / (1-s)^b))) [cite: 248, 327]
            numerator = np.power(probs, a)
            denominator = np.power(1 - probs, b)
            
            lr_ratio = (np.exp(c) * (numerator / denominator))
            calibrated_probs = 1 / (1 + (1 / lr_ratio))

            # Add them to the eval calib values
            all_eval_calib_probs.append(calibrated_probs.flatten())

            # Compute Dice scores and lesion volume after beta calibration
            binary_prediction_after = np.where(calibrated_probs > 1e-3, 1, 0)
            dice_after = dice_score(binary_prediction_after, label_data)
            soft_lesion_volume_after = calibrated_probs.sum() * voxel_volume
            bin_lesion_volume_after = binary_prediction_after.sum() * voxel_volume

            # Add everything to the results dataframe
            new_line = pd.DataFrame({"subject": [subject], "resamp_factor": [down_factor], "dice_before_calib": [dice_before], "dice_after_calib": [dice_after],
                                    "soft_lesion_volume_after_calib": [soft_lesion_volume_after], "soft_lesion_volume_before_calib": [soft_lesion_volume_before],
                                        "bin_lesion_volume_after_calib": [bin_lesion_volume_after], "bin_lesion_volume_before_calib": [bin_lesion_volume_before]})
            results_df = pd.concat([results_df, new_line], ignore_index=True)

        # Save the results dataframe after each image to avoid losing everything in case of crash
        results_df.to_csv(os.path.join(output_folder, "results_evaluation.csv"), index=False)

    # # all_eval_probs = np.concatenate(all_eval_probs)
    # all_eval_calib_probs = np.concatenate(all_eval_calib_probs)
    # all_eval_labels = np.concatenate(all_eval_labels)
    
    # # Plot and save the calibration curves, before and after beta calibration
    # prob_true, prob_pred = calibration_curve(all_eval_labels, all_eval_probs, n_bins=20)
    # prob_true_calib, prob_pred_calib = calibration_curve(all_eval_labels, all_eval_calib_probs.flatten(), n_bins=20)
    # plt.figure(figsize=(10, 10))
    # plt.plot(prob_pred, prob_true, marker='o', label='Before beta calibration')
    # plt.plot(prob_pred_calib, prob_true_calib, marker='o', label='After beta calibration')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    # plt.xlabel('Mean Predicted Probability')
    # plt.ylabel('Fraction of Positives')
    # plt.title(f'Calibration Curve for {subject}')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_folder, f"calibration_curve.png"))
    # plt.close()

    # # Load the results dataframe
    # results_df = pd.read_csv(os.path.join(output_folder, "results_evaluation.csv"))

    # Now we evaluate the results of beta calibration:
    text_to_write = ""
    ## We compute the lesion volume std for soft and binary predictions before and after beta calibration
    lesion_volume_std_soft_before = results_df.groupby("subject")["soft_lesion_volume_before_calib"].std()
    lesion_volume_std_binary_before = results_df.groupby("subject")["bin_lesion_volume_before_calib"].std()
    lesion_volume_std_soft_after = results_df.groupby("subject")["soft_lesion_volume_after_calib"].std()
    lesion_volume_std_binary_after = results_df.groupby("subject")["bin_lesion_volume_after_calib"].std()
    ## We comput the coefficient of variation (std/mean) for soft and binary predictions
    lesion_volume_cv_soft_before = lesion_volume_std_soft_before / results_df.groupby("subject")["soft_lesion_volume_before_calib"].mean()
    lesion_volume_cv_soft_after = lesion_volume_std_soft_after / results_df.groupby("subject")["soft_lesion_volume_after_calib"].mean()
    lesion_volume_cv_binary_before = lesion_volume_std_binary_before / results_df.groupby("subject")["bin_lesion_volume_before_calib"].mean()
    lesion_volume_cv_binary_after = lesion_volume_std_binary_after / results_df.groupby("subject")["bin_lesion_volume_after_calib"].mean()
    # Dice score results
    dice_before_calib = results_df.groupby("subject")["dice_before_calib"].mean()
    dice_after_calib = results_df.groupby("subject")["dice_after_calib"].mean()
    text_to_write += "Summary of evaluation results:\n"
    # text_to_write += f"Coefficient used in beta calibration: a={a}, b={b}, c={c}\n"
    text_to_write += f"Lesion volume std for soft predictions before beta calibration: {lesion_volume_std_soft_before.mean()}\n"
    text_to_write += f"Lesion volume std for soft predictions after beta calibration: {lesion_volume_std_soft_after.mean()}\n"
    text_to_write += f"Lesion volume std for binary predictions before beta calibration: {lesion_volume_std_binary_before.mean()}\n"
    text_to_write += f"Lesion volume std for binary predictions after beta calibration: {lesion_volume_std_binary_after.mean()}\n"
    text_to_write += f"Lesion volume coefficient of variation for soft predictions before beta calibration: {lesion_volume_cv_soft_before.mean()}\n"
    text_to_write += f"Lesion volume coefficient of variation for soft predictions after beta calibration: {lesion_volume_cv_soft_after.mean()}\n"
    text_to_write += f"Lesion volume coefficient of variation for binary predictions before beta calibration: {lesion_volume_cv_binary_before.mean()}\n"
    text_to_write += f"Lesion volume coefficient of variation for binary predictions after beta calibration: {lesion_volume_cv_binary_after.mean()}\n"
    text_to_write += f"Dice score before beta calibration: {dice_before_calib.mean()}\n"
    text_to_write += f"Dice score after beta calibration: {dice_after_calib.mean()}\n"
    out_txt_path = os.path.join(output_folder, "summary_results_evaluation.txt")
    with open(out_txt_path, "w") as f:
            f.write(text_to_write)


if __name__ == "__main__":
    args = parse_args()
    main_beta_calibration(args.msd, args.model_path, args.o, args.smaller_changes)