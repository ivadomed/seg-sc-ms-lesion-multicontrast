"""
This script analyzes the stability of predicted lesion volumes across subtle data augmentations.
It takes as input the lesion_volume_std.csv file produced by run_eval_lesion_std_vol.py and computes,
for each image, the std and coefficient of variation (CV) of the predicted lesion volume across
augmentations. It then breaks down these stability metrics per contrast and saves summary csvs and figures.

Input:
    -i: path to the lesion_volume_std.csv file produced by run_eval_lesion_std_vol.py
    -o: output folder to save the summary csv and figures

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the stability of predicted lesion volumes across subtle data augmentations.")
    parser.add_argument("-i", "--input-csv", type=str, required=True, help="Path to the lesion_volume_std.csv file")
    parser.add_argument("-o", "--output-folder", type=str, required=True, help="Output folder to save the summary csv and figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    aug_columns = [col for col in df.columns if col != "image"]

    # Per-image stability metrics across augmentations
    summary = pd.DataFrame({"image": df["image"]})
    summary["mean_volume"] = df[aug_columns].mean(axis=1)
    summary["std_volume"] = df[aug_columns].std(axis=1)
    summary["cv_volume"] = summary["std_volume"] / summary["mean_volume"].replace(0, np.nan)
    summary["min_volume"] = df[aug_columns].min(axis=1)
    summary["max_volume"] = df[aug_columns].max(axis=1)
    summary["range_volume"] = summary["max_volume"] - summary["min_volume"]

    # Retrieve the contrast of each image from the last element of the image name (assuming the format is something like "sub-01_ses-01_T1.nii.gz")
    summary["contrast"] = summary["image"].apply(lambda x: x.split("_")[-1].split(".")[0])

    summary_path = os.path.join(args.output_folder, "lesion_volume_std_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved per-image summary to {summary_path}")

    # Add a count of images per contrast for the x-axis labels, plus an "Overall" group containing all images
    contrast_counts = summary['contrast'].value_counts()
    summary['contrast_count'] = summary['contrast'].apply(lambda x: f"{x} (n={contrast_counts[x]})")

    overall_label = f"Overall (n={len(summary)})"
    summary_with_overall = pd.concat([
        summary.assign(contrast_count=overall_label),
        summary
    ], ignore_index=True)

    # Order: Overall first, then contrasts in alphabetical order
    contrast_order = [overall_label] + sorted(summary['contrast_count'].unique())

    # Plot: std of lesion volume per contrast (+ overall)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=summary_with_overall, x="contrast_count", y="std_volume", order=contrast_order)
    sns.stripplot(data=summary_with_overall, x="contrast_count", y="std_volume", order=contrast_order, color="black", alpha=0.4, size=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Contrast")
    plt.ylabel("Std of predicted lesion volume across augmentations (mm^3)")
    plt.title("Stability (std) of predicted lesion volume per contrast")
    plt.tight_layout()
    std_path = os.path.join(args.output_folder, "lesion_volume_std_per_contrast.png")
    plt.savefig(std_path, dpi=300)
    plt.close()
    print(f"Saved std per contrast plot to {std_path}")

    # Plot: CV of lesion volume per contrast (+ overall)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=summary_with_overall, x="contrast_count", y="cv_volume", order=contrast_order)
    sns.stripplot(data=summary_with_overall, x="contrast_count", y="cv_volume", order=contrast_order, color="black", alpha=0.4, size=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Contrast")
    plt.ylabel("CV of predicted lesion volume across augmentations")
    plt.title("Stability (CV) of predicted lesion volume per contrast")
    plt.tight_layout()
    cv_path = os.path.join(args.output_folder, "lesion_volume_cv_per_contrast.png")
    plt.savefig(cv_path, dpi=300)
    plt.close()
    print(f"Saved CV per contrast plot to {cv_path}")

    # Overall and per-contrast stats (mean +/- std) for std_volume and cv_volume
    stats = summary_with_overall.groupby('contrast_count')[['std_volume', 'cv_volume']].agg(['mean', 'std'])
    stats = stats.reindex(contrast_order)
    stats_path = os.path.join(args.output_folder, "lesion_volume_std_stats_per_contrast.csv")
    stats.to_csv(stats_path)
    print(f"Saved per-contrast stats to {stats_path}")

    print("\nLesion volume stability per contrast (mean +/- std)")
    for contrast_count, row in stats.iterrows():
        print(f"  {contrast_count}: std_volume = {row[('std_volume', 'mean')]:.4f} +/- {row[('std_volume', 'std')]:.4f}, "
              f"cv_volume = {row[('cv_volume', 'mean')]:.4f} +/- {row[('cv_volume', 'std')]:.4f}")


if __name__ == "__main__":
    main()
