"""
NOT USED FOR NOW
This script performs temperature scaling on the soft segmentations to calibrate the predicted probabilities.

Input:
    --msd: Path to the MSD JSON file describing the dataset
    --model-path: Path to the nnUNet model to use for prediction
    -o : Folder to save the calibrated predictions

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
import monai.transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Perform temperature scaling on soft segmentations.")
    parser.add_argument("--msd", required=True, type=str, help="Path to the MSD JSON file describing the dataset")
    parser.add_argument("--model-path", required=True, type=str, help="Path to the nnUNet model to use for prediction")
    parser.add_argument("-o", required=True, type=str, help="Folder to save the calibrated predictions")
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


def main_temperature_scaling(input_msd, path_model, output_folder):
    # Build the output folder
    os.makedirs(output_folder, exist_ok=True)
    
    with open(input_msd, 'r') as f:
        msd_data = json.load(f)
    
    images = [img["image"] for img in msd_data['images']]
    labels = [img["label"] for img in msd_data['images']]

    # Shuffle and split
    np.random.seed(43)
    combined = list(zip(images, labels))
    np.random.shuffle(combined)
    calib_data = combined[:200]

    # Initialize the model
    predictor = initialize_predictor(path_model, five_folds=False)

    # Define your transforms
    # We name them so we can track them in the CSV
    transforms_to_eval = {
        "GaussianNoise": monai.transforms.RandGaussianNoised(keys=["image"], prob=1.0, mean=0.0, std=0.05),
        "LowRes": monai.transforms.RandSimulateLowResolutiond(keys=["image", "label"], zoom_range=(0.90, 1.1), prob=1.0, downsample_mode=["trilinear", "nearest"], upsample_mode=["trilinear", "nearest"]),
        "Rotation": monai.transforms.RandRotated(keys=["image", "label"], range_x=(-0.08, 0.08), range_y=(-0.08, 0.08), range_z=(-0.08, 0.08), prob=1.0, mode=["bilinear", "nearest"])
    }

    temperatures = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]

    results_df = pd.DataFrame(columns=["subject", "transform_type", "temperature", "dice_at_0.5", "lesion_total_volume"])

    for img_path, label_path in tqdm(calib_data, desc="Processing"):
        subject = Path(img_path).name.replace(".nii.gz", "")
        
        # Load images (assuming your Image class handles loading)
        img_obj = Image(img_path)
        label_obj = Image(label_path)

        img_data = img_obj.data.transpose([2, 1, 0])
        img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

        label_data = label_obj.data.transpose([2, 1, 0])
        label_data = np.expand_dims(label_data, axis=0).astype(np.float32)

        # Prepare dictionary for MONAI
        # Note: MONAI expects [Channel, H, W, D]
        data_dict = {
            "image": img_data, 
            "label": label_data
        }
        
        for trans_name, transform in transforms_to_eval.items():
            # Apply the transform
            # Note: MONAI transforms usually require a seed for Rand transforms if you want reproducibility
            transformed_dict = transform(data_dict)

            # Extratc used zoom range for naming
            if trans_name == "LowRes":
                zoom_range = transform.zoom_range
                print(zoom_range)

            # Get logits from predictor
            _, logits = predictor.predict_single_npy_array(
                input_image=transformed_dict["image"],
                image_properties={'spacing': img_obj.dim[6:3:-1]}, # Using original spacing
                save_or_return_probabilities=True,
                return_logits=True
            )
            logits = torch.from_numpy(logits)

            for temp in temperatures:
                scaled_logits = logits / temp
                calibrated_probs = F.softmax(scaled_logits, dim=0)
                
                # Foreground probs (assume index 1)
                foreground_probs = calibrated_probs[1].cpu().numpy()
                
                # Metrics
                voxel_volume = np.prod(img_obj.dim[6:3:-1])
                binary_preds = (foreground_probs >= 0.5).astype(np.uint8)
                dice = dice_score(binary_preds, transformed_dict["label"].astype(np.uint8))
                
                lesion_vol = (foreground_probs >= 0.5).sum() * voxel_volume

                new_line = pd.DataFrame({
                    "subject": [subject], 
                    "transform_type": [trans_name], 
                    "temperature": [temp], 
                    "dice_at_0.5": [dice],
                    "lesion_total_volume": [lesion_vol]
                })
                results_df = pd.concat([results_df, new_line], ignore_index=True)

        results_df.to_csv(os.path.join(output_folder, "temperature_scaling_results.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    main_temperature_scaling(args.msd, args.model_path, args.o)