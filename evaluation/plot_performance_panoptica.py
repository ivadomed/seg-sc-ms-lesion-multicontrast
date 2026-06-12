"""
Plot panoptica evaluation results per contrast/site, analogous to plot_performance.py.

Reads the results_panoptica.json file produced by evaluate_predictions_with_panoptica.py
and the dataset JSON split file.

Usage:
    python plot_performance_panoptica.py \
        --pred-dir-path /path/to/results_dir \
        --data-json-path /path/to/dataset.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_parser():
    parser = argparse.ArgumentParser(description="Plot panoptica evaluation results per contrast")
    parser.add_argument("--pred-dir-path", required=True,
                        help="Directory containing results_panoptica.json")
    parser.add_argument("--data-json-path", required=True,
                        help="Path to the dataset JSON split file")
    return parser


def main():
    args = get_parser().parse_args()

    # Load panoptica results
    results_file = os.path.join(args.pred_dir_path, "results_panoptica.json")
    with open(results_file, "r") as f:
        raw = json.load(f)

    # null -> NaN
    for name, metrics in raw.items():
        for k, v in metrics.items():
            if v is None:
                metrics[k] = np.nan

    df = pd.DataFrame.from_dict(raw, orient="index").reset_index()
    df = df.rename(columns={"index": "name"})

    # Load dataset JSON
    with open(args.data_json_path, "r") as f:
        jsondata = json.load(f)

    all_entries = (
        jsondata.get("train", [])
        + jsondata.get("validation", [])
        + jsondata.get("test", [])
        + jsondata.get("externalValidation", [])
    )
    entry_by_name = {e["image"]: e for e in all_entries}

    for col in ["contrast", "site", "nb_lesions", "total_lesion_volume"]:
        df[col] = None

    for idx, row in df.iterrows():
        entry = entry_by_name.get(row["name"])
        if entry:
            df.at[idx, "contrast"] = entry.get("contrast")
            df.at[idx, "site"] = entry.get("site")
            df.at[idx, "nb_lesions"] = entry.get("nb_lesions")
            df.at[idx, "total_lesion_volume"] = entry.get("total_lesion_volume")

    # Normalise contrast labels
    df["contrast"] = df["contrast"].apply(lambda x: "T2star" if x == "MEGRE" else x)

    # Sort by contrast
    df = df.sort_values(by="contrast").reset_index(drop=True)

    # Add count-annotated labels
    contrast_counts = df["contrast"].value_counts()
    df["contrast_count"] = df["contrast"].apply(
        lambda x: f"{x} (n={contrast_counts[x]})" if pd.notna(x) else "Unknown"
    )

    metrics_to_plot = [
        ("global_bin_dsc", "Global Binary Dice (DSC)", (-0.2, 1.2)),
        ("rq",             "Recognition Quality / F1-Score (RQ)", (-0.2, 1.2)),
        ("pq_dsc",         "Panoptic Quality DSC (PQ-DSC)", (-0.2, 1.2)),
        ("sq_dsc",         "Segmentation Quality DSC (SQ-DSC)", (-0.2, 1.2)),
        ("sq_assd",        "Segmentation Quality ASSD (SQ-ASSD)", None),
        ("sq_rvd",         "Segmentation Quality Relative Volume Difference (SQ-RVD)", None),
    ]

    stats_lines = []

    for metric, title, ylim in metrics_to_plot:
        plot_df = df.dropna(subset=[metric, "contrast_count"])
        if plot_df.empty:
            print(f"No data for {metric}, skipping.")
            continue

        plt.figure(figsize=(20, 10))
        plt.grid(True)
        sns.violinplot(x="contrast_count", y=metric, data=plot_df)
        if ylim:
            plt.ylim(*ylim)
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        out_path = os.path.join(args.pred_dir_path, f"{metric}_contrast.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")

        stats_lines.append(f"\n{title} per contrast (mean ± std)")
        line = f"  Global: {plot_df[metric].mean():.4f} ± {plot_df[metric].std():.4f}"
        print(line)
        stats_lines.append(line)

        grouped = plot_df.groupby("contrast_count")[metric].agg(["mean", "std"])
        for contrast, row in grouped.iterrows():
            line = f"  {contrast}: {row['mean']:.4f} ± {row['std']:.4f}"
            print(line)
            stats_lines.append(line)

    stats_path = os.path.join(args.pred_dir_path, "panoptica_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines) + "\n")
    print(f"\nSaved stats to {stats_path}")

    print(f"\nTotal files evaluated: {len(df)}")
    print(f"  With contrast metadata: {df['contrast'].notna().sum()}")
    print(f"  TP sum: {df['tp'].sum():.0f}  FP sum: {df['fp'].sum():.0f}  FN sum: {df['fn'].sum():.0f}")


if __name__ == "__main__":
    main()
