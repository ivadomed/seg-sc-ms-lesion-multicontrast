"""
This script generates 5 versions of N randomly selected images (and of their lesion masks) to evaluate the variability of
the predictions of segmentation models: a copy of the input image and 4 versions with one subtle random MONAI augmentation each:
    - affine: in-plane rotation (±3°), translation (±2 voxels) and scaling (±5%), also applied to the mask
    - bias: smooth multiplicative bias field
    - contrast: gamma contrast change (gamma in [0.9, 1.1])
    - noise: Gaussian noise (std up to 5% of the image std)

Input:
    --image-folder: Folder containing the images (.nii.gz)
    --label-folder: Folder containing the lesion masks (same name as the image, without the nnUNet "_0000" suffix if any)
    --output-folder: Folder to save the images (images_<version>/), masks (labels_<version>/) and the list of selected images (selected_images.json)
    --n-images: Number of images to select (default: 50)
    --seed: Random seed for the selection and the augmentations (default: 42)

Example:
    python generate_augmented_images.py --image-folder /path/to/imagesTs --label-folder /path/to/labelsTs --output-folder /path/to/output

Author: Pierre-Louis Benveniste
"""
import argparse
import json
import random
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
from monai.transforms import Compose, Identityd, RandAffined, RandBiasFieldd, RandAdjustContrastd, RandGaussianNoised


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a copy and 4 slightly augmented versions of images and lesion masks to evaluate prediction variability.")
    parser.add_argument("--image-folder", required=True, help="Folder containing the images")
    parser.add_argument("--label-folder", required=True, help="Folder containing the lesion masks")
    parser.add_argument("--output-folder", required=True, help="Folder to save the augmented images and masks")
    parser.add_argument("--n-images", type=int, default=50, help="Number of images to select (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Randomly select the images and save the list
    images = random.Random(args.seed).sample(sorted(Path(args.image_folder).glob("*.nii.gz")), args.n_images)
    with open(output_folder / "selected_images.json", "w") as f:
        json.dump([str(image) for image in images], f, indent=4)

    for i, image_path in enumerate(tqdm(images)):
        label_name = image_path.name.replace("_0000.nii.gz", ".nii.gz")
        image, label = nib.load(image_path), nib.load(Path(args.label_folder) / label_name)
        data = {"image": image.get_fdata(dtype=np.float32)[None], "label": label.get_fdata(dtype=np.float32)[None]}
        # Spatial changes stay in-plane (rotation around the thickest axis, no shift/scaling along it) to avoid interpolating across thick slices
        thick = np.arange(3) == np.argmax(image.header.get_zooms()[:3])
        versions = {
            "copy": Identityd(keys=["image", "label"]),
            "affine": RandAffined(keys=["image", "label"], prob=1, rotate_range=0.05 * thick, translate_range=2 * ~thick, scale_range=0.05 * ~thick, mode=("bilinear", "nearest")),
            "bias": RandBiasFieldd(keys="image", prob=1, coeff_range=(-0.02, 0.02)),
            "contrast": RandAdjustContrastd(keys="image", prob=1, gamma=(0.9, 1.1)),
            "noise": RandGaussianNoised(keys="image", prob=1, std=0.05 * data["image"].std()),
        }
        for version, transform in versions.items():
            output = Compose(transform).set_random_state(seed=args.seed + i)(data)
            (output_folder / f"images_{version}").mkdir(exist_ok=True)
            (output_folder / f"labels_{version}").mkdir(exist_ok=True)
            # Saved with the original header: the content moves within the same voxel grid, as if the subject had moved in the scanner
            nib.save(nib.Nifti1Image(np.asarray(output["image"][0]), image.affine, image.header, dtype=np.float32), output_folder / f"images_{version}" / image_path.name)
            nib.save(nib.Nifti1Image(np.asarray(output["label"][0]).astype(np.uint8), label.affine, label.header, dtype=np.uint8), output_folder / f"labels_{version}" / label_name)


if __name__ == "__main__":
    main()
