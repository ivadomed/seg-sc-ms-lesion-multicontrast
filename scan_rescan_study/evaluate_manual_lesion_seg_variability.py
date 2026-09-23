"""
Evaluates the scan-rescan variability of the manual lesion segmentations, for each pair of the input
json which is not excluded and has a manual lesion segmentation for both runs:
    - lesion volume and number of lesions of each run
    - lesion volume difference between runs
    - Dice between the run-01 lesion seg and the run-02 lesion seg registered to run-01
      (using the warping field of generate_study_files.py)

Outputs (in the output folder):
    - lesion_seg_variability.csv: one line per pair (including the has_lesion field of the input json)
    - lesion_seg_variability.png: variability plots of all pairs
    - lesion_seg_variability_by_lesion.png: same plots, labeled by the has_lesion field

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -j / --json         Path to the input json (with the has_lesion field)
    -o / --output       Path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage
from tqdm import tqdm

VOL_1, VOL_2, VOL_DIFF = "lesion_volume_run-01_mm3", "lesion_volume_run-02_mm3", "lesion_volume_diff_mm3"


def run(cmd):
    assert os.system(cmd) == 0, f"Command failed: {cmd}"


def load_seg(path):
    """Loads a binary segmentation and its voxel volume."""
    img = nib.load(path)
    return np.asarray(img.dataobj) > 0.5, np.prod(img.header.get_zooms()[:3])


def process_pair(entry, files, dataset, study_dir):
    les_1, les_2 = dataset / entry["lesion_seg_run-01"], dataset / entry["lesion_seg_run-02"]

    # Register the run-02 lesion seg to run-01
    les_2_reg = (study_dir / files["sc_seg_run-02"]).parent / les_2.name.replace(".nii.gz", "_space-run01.nii.gz")
    if not les_2_reg.exists():
        run(f"sct_apply_transfo -i {les_2} -d {dataset / entry['scan_run-01']} -w {study_dir / files['warp_run-02_to_run-01']} -x nn -o {les_2_reg}")

    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2), (seg_2_reg, _) = load_seg(les_1), load_seg(les_2), load_seg(les_2_reg)
    vol_1, vol_2 = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2
    n_voxels = seg_1.sum() + seg_2_reg.sum()

    return {
        "subject": entry["subject"],
        "session": entry["session"],
        "acquisition": entry["acquisition"],
        "has_lesion": entry["has_lesion"],
        VOL_1: vol_1,
        VOL_2: vol_2,
        VOL_DIFF: vol_2 - vol_1,
        "lesion_volume_abs_diff_percent": abs(vol_2 - vol_1) / ((vol_1 + vol_2) / 2) * 100 if vol_1 + vol_2 > 0 else np.nan,
        "lesion_count_run-01": ndimage.label(seg_1, structure=np.ones((3, 3, 3)))[1],
        "lesion_count_run-02": ndimage.label(seg_2, structure=np.ones((3, 3, 3)))[1],
        "dice_run-02_registered_to_run-01": 2 * (seg_1 & seg_2_reg).sum() / n_voxels if n_voxels > 0 else np.nan,
    }


def plot_variability(df, output, hue=None):
    """Plots the lesion volumes of both runs, their Bland-Altman plot and the Dice scores."""
    df = df.assign(mean_volume=(df[VOL_1] + df[VOL_2]) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(data=df, x=VOL_1, y=VOL_2, hue=hue, ax=axes[0])
    max_vol = df[[VOL_1, VOL_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set_title("Lesion volume")

    sns.scatterplot(data=df, x="mean_volume", y=VOL_DIFF, hue=hue, ax=axes[1], legend=False)
    mean_diff, std_diff = df[VOL_DIFF].mean(), df[VOL_DIFF].std()
    for y in (mean_diff - 1.96 * std_diff, mean_diff, mean_diff + 1.96 * std_diff):
        axes[1].axhline(y, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Bland-Altman of lesion volume (run-02 - run-01)")

    sns.boxplot(data=df, x=hue, y="dice_run-02_registered_to_run-01", ax=axes[2], showfliers=False)
    sns.stripplot(data=df, x=hue, y="dice_run-02_registered_to_run-01", ax=axes[2], color="k")
    axes[2].set_title("Dice (run-02 registered to run-01)")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the manual lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-j", "--json", required=True, type=Path, help="Path to the input json (with the has_lesion field)")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    with open(study_dir / "study_files.json") as f:
        study = json.load(f)
    with open(args.json) as f:
        entries = json.load(f)
    files = {(p["subject"], p["session"], p["acquisition"]): p for p in study["pairs"] if p["status"] == "success"}

    results = []
    for entry in tqdm([e for e in entries if not e["excluded"] and e["lesion_seg_run-01"] and e["lesion_seg_run-02"]]):
        try:
            results.append(process_pair(entry, files[(entry["subject"], entry["session"], entry["acquisition"])], Path(study["dataset"]), study_dir))
        except Exception as e:
            print(f"Failed for {entry['subject']}/{entry['session']}/acq-{entry['acquisition']}: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.output / "lesion_seg_variability.csv", index=False)
    plot_variability(df, args.output / "lesion_seg_variability.png")
    df["lesion"] = df["has_lesion"].map({True: "lesion", False: "no lesion"}).fillna("unknown")
    plot_variability(df, args.output / "lesion_seg_variability_by_lesion.png", hue="lesion")
    print(f"Results of {len(results)} pairs saved to {args.output}")


if __name__ == "__main__":
    main()
