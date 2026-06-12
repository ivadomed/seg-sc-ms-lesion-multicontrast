"""
This script evaluates the variability of predicted lesion segmentation volume with very suble changes in the input image.
It takes as input a folder with nifti files and outputs the predicted lesion segmentation files for each DA strat.
It also computes the volume of the predicted lesion segmentation for each DA strat and saves it in a csv file.

Input:
    -i: input folder containing nifti files
    -o: output folder to save the predicted lesion segmentation files and the csv file with the volumes
    -model: path to the trained model to use for prediction (path to the nnunettrainer folder)

Author: Pierre-Louis Benveniste
"""

import argparse
import os
import sys

os.environ['nnUNet_raw'] = "./nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "./nnUNet_preprocessed"
os.environ['nnUNet_results'] = "./nnUNet_results"

import importlib
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join  # noqa: E402
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent / "post-processing"))
from image import Image, get_orientation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the std of predicted lesion volume under subtle augmentations.")
    parser.add_argument("-i", "--input-folder", type=str, required=True, help="Input folder containing nifti files")
    parser.add_argument("-o", "--output-folder", type=str, required=True, help="Output folder to save predictions and csv file")
    parser.add_argument("-model", "--path-model", type=str, required=True, help="Path to the trained nnUNet model folder")

    return parser.parse_args()


def load_trainer_class_if_available(path_model):
    """
    Load a custom nnUNet trainer class if available in the model folder.
    """
    trainer_file = os.path.join(path_model, "trainer_class.py")
    if os.path.exists(trainer_file):
        spec = importlib.util.spec_from_file_location("trainer_class", trainer_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "get_trainer_class"):
            print(f"Custom trainer class found in {trainer_file}, using it for prediction.")
            return module.get_trainer_class()
    return None


def initialize_predictor(path_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    folds = [0, 1, 2, 3, 4]
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
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


def augment_identity(data, rng):
    """No change applied to the image."""
    return data.copy()


def augment_gaussian_noise(data, rng):
    """Add very subtle Gaussian noise, scaled to the image intensity range."""
    sigma = 0.01 * (np.percentile(data, 99) - np.percentile(data, 1))
    noise = rng.normal(loc=0.0, scale=sigma, size=data.shape)
    return data + noise


def augment_gaussian_blur(data, rng):
    """Apply a very light Gaussian blur."""
    sigma = rng.uniform(0.3, 0.6)
    return gaussian_filter(data, sigma=sigma)


def augment_intensity_shift(data, rng):
    """Apply a small global intensity shift."""
    shift = rng.uniform(-0.02, 0.02) * (np.percentile(data, 99) - np.percentile(data, 1))
    return data + shift


def augment_intensity_scale(data, rng):
    """Apply a small global intensity scaling (contrast change)."""
    scale = rng.uniform(0.97, 1.03)
    mean = data.mean()
    return (data - mean) * scale + mean


def augment_gamma(data, rng):
    """Apply a very subtle gamma correction."""
    gamma = rng.uniform(0.95, 1.05)
    data_min = data.min()
    data_max = data.max()
    if data_max - data_min < 1e-8:
        return data.copy()
    normalized = (data - data_min) / (data_max - data_min)
    corrected = np.power(normalized, gamma)
    return corrected * (data_max - data_min) + data_min


def augment_bias_field(data, rng):
    """Apply a smooth, low-amplitude multiplicative bias field."""
    field = rng.normal(loc=0.0, scale=1.0, size=data.shape)
    field = gaussian_filter(field, sigma=np.array(data.shape) / 4)
    field = field / (np.abs(field).max() + 1e-8)
    bias = 1.0 + 0.03 * field
    return data * bias


AUGMENTATIONS = {
    "original": augment_identity,
    "gaussian_noise": augment_gaussian_noise,
    "gaussian_blur": augment_gaussian_blur,
    "intensity_shift": augment_intensity_shift,
    "intensity_scale": augment_intensity_scale,
    "gamma": augment_gamma,
    "bias_field": augment_bias_field,
}


def run_prediction(predictor, img_in, data_aug):
    """Run nnUNet prediction on an augmented version of the (already reoriented) image and return the binary lesion mask in RPI."""
    data = data_aug.transpose([2, 1, 0])
    data = np.expand_dims(data, axis=0).astype(np.float32)

    pred = predictor.predict_single_npy_array(
        input_image=data,
        image_properties={'spacing': img_in.dim[6:3:-1]},
        save_or_return_probabilities=True,
        return_logits_per_fold=False
    )
    _, prob_maps = pred
    pred_soft = np.where(prob_maps[1] > prob_maps[0], prob_maps[1], 0)
    pred_bin = np.where(pred_soft > 0.5, 1, 0)
    return pred_bin.transpose([2, 1, 0])


def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    path_model = args.path_model

    os.makedirs(output_folder, exist_ok=True)

    input_images = sorted(str(p) for p in Path(input_folder).rglob("*.nii.gz"))

    # Remove files with derivatives in the name to avoid processing segmentation files
    input_images = [p for p in input_images if "derivative" not in p]
    # Remove SHA256 files
    input_images = [p for p in input_images if "SHA256" not in p]

    predictor = initialize_predictor(path_model)
    model_orientation = "RPI"

    csv_path = os.path.join(output_folder, "lesion_volume_std.csv")
    fieldnames = ["image"] + list(AUGMENTATIONS.keys())

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for img_idx, img_path in enumerate(tqdm(input_images, desc="Evaluating lesion volume std under subtle augmentations")):
            orig_orientation = get_orientation(Image(img_path))

            img_in = Image(img_path)
            if orig_orientation != model_orientation:
                img_in.change_orientation(model_orientation)

            voxel_volume = float(np.prod(img_in.dim[4:7]))
            img_name = os.path.basename(img_path).replace(".nii.gz", "")

            row = {"image": img_name}
            # The seed is based on the image name to ensure that the same augmentations are applied to the same image
            rng = np.random.default_rng(seed=hash(img_name) % (2 ** 32))

            for aug_name, aug_fn in AUGMENTATIONS.items():
                aug_data = aug_fn(img_in.data, rng)
                pred_bin = run_prediction(predictor, img_in, aug_data)

                # For the first image only, the output images are also saved to visually check the effect of the augmentations and the predictions
                if img_idx == 0:
                    img_out_bin = img_in.copy()
                    img_out_bin.data = pred_bin
                    if orig_orientation != model_orientation:
                        img_out_bin.change_orientation(orig_orientation)

                    out_path = os.path.join(output_folder, "predictions", img_name, f"{img_name}_{aug_name}.nii.gz")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    img_out_bin.save(out_path)

                lesion_volume = float(pred_bin.sum()) * voxel_volume
                row[aug_name] = lesion_volume

            writer.writerow(row)
            csv_file.flush()
            print(f"Done with image {img_path}")


if __name__ == "__main__":
    main()
