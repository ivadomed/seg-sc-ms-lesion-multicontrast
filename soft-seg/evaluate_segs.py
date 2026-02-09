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
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


def dice_score(prediction, groundtruth, smooth=1.):
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate soft segmentations at various thresholds.")
    parser.add_argument("--msd", required=True, type=str, help="Path to the MSD json file")
    parser.add_argument("-pred-folder", required=True, type=str, help="Folder containing the predictions of the model on the test set")
    parser.add_argument("-o", required=True, type=str, help="Folder to save the evaluation results")
    return parser.parse_args()


def main_dice_eval(msd_path, pred_folder, output_folder):

    # We also want to save all results to log file
    log_path = os.path.join(output_folder, "dice_analysis.log")
    ## remove existing log file if it exists
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.add(log_path, format="{time} {level} {message}", level="INFO")

    # Load MSD json file
    with open(msd_path, "r") as f:
        msd_dict = json.load(f)

    msd_data = msd_dict["images"]

    # Get all soft segmentation in the pred_folder
    soft_seg_files = list(Path(pred_folder).rglob("*_lesion.nii.gz"))
    soft_seg_files = [str(f) for f in soft_seg_files if "/soft/" in str(f)]

    dice_scores_df = pd.DataFrame(columns=["subject", "resamp_factor", "dice_at_0.5", "dice_at_1e-6"])

    for pred_img in tqdm(soft_seg_files):
        # img name
        image_name = pred_img.split("/")[-3]
        # For this image, get the corresponding entry in the msd_data
        corresponding_img = None
        for msd_image in msd_data:
            if image_name in msd_image["image"]:
                corresponding_img = msd_image
                break

        if corresponding_img is None:
            return ValueError(f"No corresponding entry found in MSD json for image {image_name}")

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
        
        # Compute dice at 0.5
        ## Binarize soft segmentation
        soft_data_thresh = soft_data.copy()
        soft_data_thresh[soft_data_thresh > 0.5] = 1
        soft_data_thresh[soft_data_thresh <= 0.5] = 0
        dice_at_05 = dice_score(soft_data_thresh, gt_data)
        
        # Compute dice at 1e-6
        soft_data_thresh = soft_data.copy()
        soft_data_thresh[soft_data_thresh > 1e-6] = 1
        soft_data_thresh[soft_data_thresh <= 1e-6] = 0
        dice_at_1e_6 = dice_score(soft_data_thresh, gt_data)

        # Update dice scores dataframe
        resamp_factor = pred_img.split("/")[-1].split("_")[-2]
        new_line = pd.DataFrame({"subject": [image_name], "resamp_factor": [resamp_factor], "dice_at_0.5": [dice_at_05], "dice_at_1e-6": [dice_at_1e_6]})
        dice_scores_df = pd.concat([dice_scores_df, new_line], ignore_index=True)

    # Save dice scores dataframe to csv
    dice_scores_df.to_csv(os.path.join(output_folder, "dice_scores.csv"), index=False)
    logger.info(f"Saved {os.path.join(output_folder, 'dice_scores.csv')}")

    # Compute average dice score per image
    ## At 0.5
    img_avg_dice_05 = dice_scores_df.groupby("subject")["dice_at_0.5"].mean()
    ## At 1e-6
    img_avg_dice_1e_6 = dice_scores_df.groupby("subject")["dice_at_1e-6"].mean()
    logger.info(f" ------------ Avg dice score per subject ------------ ")
    logger.info(f"Average Dice at 0.5: {img_avg_dice_05.mean():.4f}")
    logger.info(f"Average Dice at 1e-6: {img_avg_dice_1e_6.mean():.4f}")

    # Compute average dice score across all images
    avg_dice_05 = dice_scores_df["dice_at_0.5"].mean()
    avg_dice_1e_6 = dice_scores_df["dice_at_1e-6"].mean()
    logger.info(f" ------------ Avg dice score across all subjects ------------ ")
    logger.info(f"Average Dice at 0.5: {avg_dice_05:.4f}")
    logger.info(f"Average Dice at 1e-6: {avg_dice_1e_6:.4f}")
    
    # Create a dict for resampling factors
    resamp_factors = ['0.5x0.5x0.5', '0.5x0.5x1.0', '0.5x1.0x0.5', '1.0x0.5x0.5', '0.6x0.6x0.6', '0.6x0.6x1.0', '0.6x1.0x0.6',
                      '1.0x0.6x0.6', '0.7x0.7x0.7', '0.7x0.7x1.0', '0.7x1.0x0.7', '1.0x0.7x0.7', '0.8x0.8x0.8', '0.8x0.8x1.0',
                      '0.8x1.0x0.8', '1.0x0.8x0.8', '0.9x0.9x0.9', '0.9x0.9x1.0', '0.9x1.0x0.9', '1.0x0.9x0.9']
    resamp_factors_dict = {resamp_factor: idx for idx, resamp_factor in enumerate(resamp_factors)}
    dice_scores_df["resamp_factor"] = dice_scores_df["resamp_factor"].map(resamp_factors_dict)

    # Plot dice score at 0.5 per resampling factor
    dice_scores_df['resamp_factor'] = dice_scores_df['resamp_factor'].astype(str)
    df_plot = dice_scores_df.groupby("resamp_factor")["dice_at_0.5"].mean().reset_index()
    plt.plot(df_plot["resamp_factor"], df_plot["dice_at_0.5"], marker='o', linestyle='-', label='Dice at 0.5')
    df_plot = dice_scores_df.groupby("resamp_factor")["dice_at_1e-6"].mean().reset_index()
    plt.plot(df_plot["resamp_factor"], df_plot["dice_at_1e-6"], marker='o', linestyle='-', label='Dice at 1e-6')
    plt.xlabel("Resampling Factor")
    plt.ylabel("Dice Score")
    plt.title("Dice Score per Resampling Factor")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(output_folder, "dice_score_per_resampling_factor.png"))
    logger.info(f"Saved {os.path.join(output_folder, 'dice_score_per_resampling_factor.png')}")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    msd_path = args.msd
    pred_folder = args.pred_folder
    output_folder = args.o

    # Run main evaluation
    main_dice_eval(msd_path, pred_folder, output_folder)