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
    parser.add_argument("--data-json-path", type=str, required=True, help="Path to the MSD dataset json file")
    parser.add_argument("--conversion-dict-path", type=str, required=True, help="Path to the conversion dict json file mapping original BIDS paths to nnUNet paths")
    return parser.parse_args()


def get_contrast_per_image(image_names, data_json_path, conversion_dict_path):
    """
    For each nnUNet image name (e.g. "msLesionAgnostic_001_0000"), retrieve the contrast from the MSD dataset json file.
    The conversion dict (original BIDS path -> nnUNet path) is used to map the nnUNet name back to the original
    BIDS image path, which is then looked up in the MSD dataset json file.
    Replaces the MEGRE contrast by T2star, similarly to plot_performance.py.
    """
    with open(data_json_path, 'r') as f:
        jsondata = json.load(f)
    jsondata = jsondata['train'] + jsondata['validation'] + jsondata['test'] + jsondata['externalValidation']

    # Build a lookup from original BIDS image path to contrast
    contrast_lookup_by_path = {data['image']: data['contrast'] for data in jsondata}

    with open(conversion_dict_path, 'r') as f:
        conversion_dict = json.load(f)

    # Build a lookup from nnUNet image name (without extension) to contrast, via the original BIDS path
    contrast_lookup = {}
    n_matched_path = 0
    for orig_path, nnunet_path in conversion_dict.items():
        nnunet_name = os.path.basename(nnunet_path).replace(".nii.gz", "")
        if orig_path in contrast_lookup_by_path:
            n_matched_path += 1
            contrast_lookup[nnunet_name] = contrast_lookup_by_path[orig_path]
            
    contrasts = [contrast_lookup.get(name, "unknown") for name in image_names]
    contrasts = ["T2star" if c == "MEGRE" else c for c in contrasts]
    return contrasts


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

    # Retrieve the contrast of each image from the MSD dataset json file
    summary["contrast"] = get_contrast_per_image(summary["image"], args.data_json_path, args.conversion_dict_path)

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

    # Per-augmentation analysis: how much does each individual augmentation shift the lesion volume
    # away from the original (unaugmented) prediction, relative to the original volume.
    non_original_aug_columns = [col for col in aug_columns if col != "original"]
    deviation_long = []
    for aug_name in non_original_aug_columns:
        deviation = df[aug_name] - df["original"]
        cv = deviation / df["original"].replace(0, np.nan)
        deviation_long.append(pd.DataFrame({
            "image": df["image"],
            "augmentation": aug_name,
            "abs_deviation": deviation.abs(),
            "cv_deviation": cv.abs(),
        }))

    # "All" combines the per-image std/cv across all augmentations (i.e. the same metrics as the
    # contrast-level analysis above), to compare individual augmentations against the combined effect.
    deviation_long.append(pd.DataFrame({
        "image": summary["image"],
        "augmentation": "All",
        "abs_deviation": summary["std_volume"],
        "cv_deviation": summary["cv_volume"],
    }))
    deviation_long = pd.concat(deviation_long, ignore_index=True)

    aug_order = non_original_aug_columns + ["All"]

    # Plot: absolute deviation in lesion volume per augmentation
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=deviation_long, x="augmentation", y="abs_deviation", order=aug_order)
    sns.stripplot(data=deviation_long, x="augmentation", y="abs_deviation", order=aug_order, color="black", alpha=0.4, size=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Augmentation")
    plt.ylabel("Absolute deviation in predicted lesion volume (mm^3)")
    plt.title("Lesion volume deviation from original prediction, per augmentation")
    plt.tight_layout()
    aug_abs_path = os.path.join(args.output_folder, "lesion_volume_deviation_per_augmentation.png")
    plt.savefig(aug_abs_path, dpi=300)
    plt.close()
    print(f"Saved deviation per augmentation plot to {aug_abs_path}")

    # Plot: CV of the deviation in lesion volume per augmentation
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=deviation_long, x="augmentation", y="cv_deviation", order=aug_order)
    sns.stripplot(data=deviation_long, x="augmentation", y="cv_deviation", order=aug_order, color="black", alpha=0.4, size=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Augmentation")
    plt.ylabel("CV of predicted lesion volume deviation")
    plt.title("Lesion volume CV from original prediction, per augmentation")
    plt.tight_layout()
    aug_cv_path = os.path.join(args.output_folder, "lesion_volume_cv_per_augmentation.png")
    plt.savefig(aug_cv_path, dpi=300)
    plt.close()
    print(f"Saved CV per augmentation plot to {aug_cv_path}")

    # Stats (mean +/- std) per augmentation
    aug_stats = deviation_long.groupby('augmentation')[['abs_deviation', 'cv_deviation']].agg(['mean', 'std'])
    aug_stats = aug_stats.reindex(aug_order)
    aug_stats_path = os.path.join(args.output_folder, "lesion_volume_stats_per_augmentation.csv")
    aug_stats.to_csv(aug_stats_path)
    print(f"Saved per-augmentation stats to {aug_stats_path}")

    print("\nLesion volume deviation per augmentation (mean +/- std)")
    for aug_name, row in aug_stats.iterrows():
        print(f"  {aug_name}: abs_deviation = {row[('abs_deviation', 'mean')]:.4f} +/- {row[('abs_deviation', 'std')]:.4f}, "
              f"cv_deviation = {row[('cv_deviation', 'mean')]:.4f} +/- {row[('cv_deviation', 'std')]:.4f}")


if __name__ == "__main__":
    main()
