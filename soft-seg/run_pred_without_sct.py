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
from tqdm import tqdm
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor   # noqa: E402
import torch
import importlib


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


def main_run_pred(input_folder, output_folder, path_model):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    input_images = list(Path(input_folder).rglob("*.nii.gz"))
    input_images = [str(img) for img in input_images]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For each image, we want to run a soft and a binary segmentation.
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
        

if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    path_model = args.path_model

    main_run_pred(input_folder, output_folder, path_model)