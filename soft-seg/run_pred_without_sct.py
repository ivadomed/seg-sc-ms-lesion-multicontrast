"""
In this script, we run predictions without using SCT so that it is faster.

Input:
    -i: Path to input folder containing images
    --path-model: Path to the nnUNet model to use for prediction
    -o: Path to output folder to save predictions

Author: Pierre-Louis Benveniste
"""
import argparse
import os

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run lesion segmentation on all images in a folder.")
    parser.add_argument("-i", "--input_folder", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--path-model", type=str, required=True, help="Path to the nnUNet model to use for prediction")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Path to output folder to save predictions")
    return parser.parse_args()


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


def initialize_predictor(path_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We load the model
    folds = [2]
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


def main_run_pred(input_folder, output_folder, path_model):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    input_images = list(Path(input_folder).rglob("*.nii.gz"))
    input_images = [str(img) for img in input_images]

    predictor = initialize_predictor(path_model)

    # Now for each image, we run the prediction
    for img in tqdm(input_images, desc="Running lesion segmentation"):

        orig_orientation = get_orientation(Image(img))
        model_orientation = "RPI"

        # Reorient the image to model orientation if not already
        img_in = Image(img)
        if orig_orientation != model_orientation:
            img_in.change_orientation(model_orientation)

        # Run nnUNet prediction
        data = img_in.data.transpose([2, 1, 0])
        data = np.expand_dims(data, axis=0).astype(np.float32)
        pred = predictor.predict_single_npy_array(
            input_image=data,
            # The spacings also have to be reversed to match nnUNet's conventions.
            image_properties={'spacing': img_in.dim[6:3:-1]},
            # Save the probability maps if specified
            save_or_return_probabilities=True,
            # If using a model ensemble, return the logits per fold so we can average them ourselves
            return_logits_per_fold=False
        )
        _, prob_maps = pred
        pred_output_raw = prob_maps[1]
        pred_soft = np.where(prob_maps[1] > prob_maps[0], prob_maps[1], 0)
        pred_bin = np.where(pred_soft > 0.5, 1, 0)
        pred_output_raw = pred_output_raw.transpose([2, 1, 0])
        pred_soft = pred_soft.transpose([2, 1, 0])
        pred_bin = pred_bin.transpose([2, 1, 0])

        img_out_raw = img_in.copy()
        img_out_raw.data = pred_output_raw
        img_out_soft = img_in.copy()
        img_out_soft.data = pred_soft
        img_out_bin = img_in.copy()
        img_out_bin.data = pred_bin

        # Reorient the image back to original orientation
        if orig_orientation != model_orientation:
            img_out_raw.change_orientation(orig_orientation)
            img_out_soft.change_orientation(orig_orientation)
            img_out_bin.change_orientation(orig_orientation)

        # Define output paths
        site = img.split("/")[-3]
        sub_name = img.split("/")[-2]
        img_name = img.split("/")[-1].replace(".nii.gz", "")
        output_folder_site = os.path.join(output_folder, site, sub_name)
        output_raw_path = os.path.join(output_folder_site, "raw", f"{img_name}_lesion.nii.gz")
        os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
        output_binary_path = os.path.join(output_folder_site, "binary", f"{img_name}_lesion.nii.gz")
        os.makedirs(os.path.dirname(output_binary_path), exist_ok=True)
        output_soft_path = os.path.join(output_folder_site, "soft", f"{img_name}_lesion.nii.gz")
        os.makedirs(os.path.dirname(output_soft_path), exist_ok=True)

        img_out_raw.save(output_raw_path)
        img_out_soft.save(output_soft_path)
        img_out_bin.save(output_binary_path)
        print(f"Done with image {img}")

    return None
        

if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    path_model = args.path_model

    main_run_pred(input_folder, output_folder, path_model)