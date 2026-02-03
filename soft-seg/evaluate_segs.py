"""
This code evaluates the soft segmentations at various thresholds and computes evaluation metrics.

Input:
    --msd: Path to the MSD json file
    --pred-folder: Folder containing the predictions of the model on the test set
    -o : Folder to save the evaluation results

Author: Pierre-Louis Benveniste
"""
import os
import argparse
from pathlib import Path
import json
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm


def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate soft segmentations at various thresholds.")
    parser.add_argument("--msd", required=True, type=str, help="Path to the MSD json file")
    parser.add_argument("-pred-folder", required=True, type=str, help="Folder containing the predictions of the model on the test set")
    parser.add_argument("-o", required=True, type=str, help="Folder to save the evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    msd_path = args.msd
    pred_folder = args.pred_folder
    output_folder = args.o

    # Load MSD json file
    with open(msd_path, "r") as f:
        msd_dict = json.load(f)

    msd_data = msd_dict["test"]

    # Get all soft segmentation in the pred_folder
    soft_seg_files = list(Path(pred_folder).rglob("*_lesion.nii.gz"))
    soft_seg_files = [str(f) for f in soft_seg_files if "/soft/" in str(f)]

    dice_scores = {}
    all_dices = []

    for pred_img in tqdm(soft_seg_files):
        # img name
        image_name = pred_img.split("/")[-3]
        # For this image, get the corresponding entry in the msd_data
        corresponding_img = None
        for msd_image in msd_data:
            if image_name in msd_image["image"]:
                corresponding_img = msd_image
                break

        # Load ground truth
        gt_file = corresponding_img["label"]
        gt_img = nib.load(gt_file)
        gt_data = gt_img.get_fdata()

        # Load soft segmentation
        soft_img = nib.load(pred_img)
        soft_data = soft_img.get_fdata()
        ## Resample the soft segmentation to match the ground truth if needed
        if soft_data.shape != gt_data.shape:
            zoom_factors = [gt_data.shape[i] / soft_data.shape[i] for i in range(3)]
            soft_data = zoom(soft_data, zoom_factors, order=1)  # Linear interpolation
        ## Binarize soft segmentation
        soft_data_thresh = soft_data.copy()
        soft_data_thresh[soft_data_thresh > 0] = 1

        # Compute Dice score
        if dice_scores.get(image_name) is None:
            dice_scores[image_name] = {}
        dice = dice_score(soft_data_thresh, gt_data)
        dice_scores[image_name][pred_img.split("/")[-1].replace(".nii.gz", "")] = dice
        all_dices.append(dice)

    # Compute average dice score
    avg_dice = sum(all_dices) / len(all_dices)
    dice_scores["avg"] = avg_dice

    # Save dice scores
    os.makedirs(output_folder, exist_ok=True)
    dice_output_path = os.path.join(output_folder, "dice_scores.json")
    with open(dice_output_path, "w") as f:
        json.dump(dice_scores, f, indent=4)

    # Print average dice score
    avg_dice = dice_scores["avg"]
    print(f"Average Dice score: {avg_dice}")


if __name__ == "__main__":
    main()