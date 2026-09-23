"""
Aggregates every volume_stability.csv produced by evaluate_lesion_stability.py under a
root folder into a single CSV.

Expected layout:
    <root>/<run>/<acquisition>/volume_stability.csv
    e.g. output_eval_upper_lower_tta/r20250909/lower/volume_stability.csv

Each row is tagged with the "run" (top-level subfolder name, e.g. r20250909) and
"acquisition" (immediate parent folder name, e.g. upper/lower) it was found under.

Arguments:
    -i / --root     Root folder to search for volume_stability.csv files
    -o / --output   Path to the merged output CSV

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate volume_stability.csv files into a single CSV.")
    parser.add_argument("-i", "--root", required=True, help="Root folder to search for volume_stability.csv files")
    parser.add_argument("-o", "--output", required=True, help="Path to the merged output CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output).resolve()

    frames = []
    for csv_path in sorted(root.rglob("volume_stability.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["run"] = csv_path.parent.parent.name
        df["acquisition"] = csv_path.parent.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No volume_stability.csv files found under {root}")

    merged_df = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged {len(frames)} volume_stability.csv file(s) -> {len(merged_df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
