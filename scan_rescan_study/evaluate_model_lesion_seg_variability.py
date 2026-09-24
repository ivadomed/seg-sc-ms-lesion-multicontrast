"""
Evaluates the scan-rescan variability of the lesion segmentations predicted by the SCT model, for each
pair of study_files.json (output of generate_study_files.py), with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction> [flags of the mode]

Modes:
    default         no flag
    tta             -test-time-aug
    single-fold     -single-fold
    soft            -soft-seg (metrics computed on the soft segmentations)
    soft-bin        -soft-seg, binarized at 0.5

Metrics:
    - lesion volume and number of lesions of each run
    - lesion volume difference (mm3 and %) between run-01 and run-02, and between run-01 and run-02
      registered to run-01 (using the warping field of generate_study_files.py)
    - Dice between the run-01 prediction and the run-02 prediction registered to run-01
    - Dice of each prediction relative to the manual segmentation when available
    (soft volumes and soft Dice in the soft mode)

Outputs (in the output folder), with pairs labeled by focal lesion visible or uncertain (has_lesion field):
    - lesion_seg_variability.csv: one line per pair
    - lesion_seg_variability_native.png: lesion volumes and Bland-Altman plots (mm3 and %), run-01 vs run-02
    - lesion_seg_variability_registered.png: same, run-01 vs run-02 registered to run-01, and Dice
    - lesion_seg_variability_native_visible.png and lesion_seg_variability_registered_visible.png: same
      plots, only for the pairs with visible focal lesions

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    -m / --mode         Inference mode (default: default)
    --exclude-stitch    Exclude the stitch acquisitions (included by default)

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

MODES = {"default": "", "tta": "-test-time-aug", "single-fold": "-single-fold", "soft": "-soft-seg", "soft-bin": "-soft-seg"}
VOL_1 = "lesion_volume_run-01_mm3"
DICE = "dice_run-02_registered_to_run-01"
PALETTE = {"visible": "tab:orange", "uncertain": "tab:blue"}


def run(cmd):
    assert os.system(cmd) == 0, f"Command failed: {cmd}"


def load_seg(path, soft):
    """Loads a segmentation (binarized at 0.5 unless soft) and its voxel volume."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=float)
    return (data if soft else (data > 0.5).astype(float)), np.prod(img.header.get_zooms()[:3])


def dice(seg_1, seg_2):
    """(Soft) Dice score, 1 if both segmentations are empty."""
    denominator = seg_1.sum() + seg_2.sum()
    return 2 * (seg_1 * seg_2).sum() / denominator if denominator > 0 else 1.0


def volume_diff(vol_1, vol_2, prefix):
    """Absolute volume difference |vol_2 - vol_1| in mm3 and in % of the mean volume (0 % if both volumes are 0)."""
    return {
        f"{prefix}_abs_diff_mm3": abs(vol_2 - vol_1),
        f"{prefix}_abs_diff_percent": abs(vol_2 - vol_1) / ((vol_1 + vol_2) / 2) * 100 if vol_1 + vol_2 > 0 else 0.0,
    }


def process_pair(pair, dataset, study_dir, mode):
    out_dir = (study_dir / pair["sc_seg_run-01"]).parent
    flags = MODES[mode]

    # Inference on both runs (shared between modes with the same flags) and registration of run-02 to run-01
    preds = []
    for key in ("scan_run-01", "rescan_run-02"):
        pred = out_dir / Path(pair[key]).name.replace(".nii.gz", f"_label-lesion_desc-{flags.lstrip('-') or 'default'}_seg.nii.gz")
        if not pred.exists():
            run(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {dataset / pair[key]} -o {pred} {flags}")
        preds.append(pred)
    pred_2_reg = Path(str(preds[1]).replace(".nii.gz", "_space-run01.nii.gz"))
    if not pred_2_reg.exists():
        run(f"sct_apply_transfo -i {preds[1]} -d {dataset / pair['scan_run-01']} -w {study_dir / pair['warp_run-02_to_run-01']} -x nn -o {pred_2_reg}")

    soft = mode == "soft"
    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2), (seg_2_reg, _) = (load_seg(p, soft) for p in (*preds, pred_2_reg))
    vol_1, vol_2, vol_2_reg = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2, seg_2_reg.sum() * voxel_vol_1

    # Dice relative to the manual segmentations (if available)
    dice_manual = {r: dice(seg, load_seg(dataset / pair[f"lesion_seg_run-{r}"], soft=False)[0]) if pair[f"lesion_seg_run-{r}"] else np.nan
                   for r, seg in (("01", seg_1), ("02", seg_2))}

    return {
        "subject": pair["subject"],
        "session": pair["session"],
        "acquisition": pair["acquisition"],
        "has_lesion": pair["has_lesion"],
        "lesion_count_run-01": ndimage.label(seg_1 > 0.5, structure=np.ones((3, 3, 3)))[1],
        "lesion_count_run-02": ndimage.label(seg_2 > 0.5, structure=np.ones((3, 3, 3)))[1],
        VOL_1: vol_1,
        "lesion_volume_run-02_mm3": vol_2,
        **volume_diff(vol_1, vol_2, "lesion_volume"),
        "lesion_volume_run-02_registered_mm3": vol_2_reg,
        **volume_diff(vol_1, vol_2_reg, "lesion_volume_registered"),
        DICE: dice(seg_1, seg_2_reg),
        "dice_vs_manual_run-01": dice_manual["01"],
        "dice_vs_manual_run-02": dice_manual["02"],
    }


def plot_variability(df, prefix, vol_2, run_2_name, output, dice=False):
    """Plots the lesion volumes, their Bland-Altman plots (mm3 and %) and optionally the Dice, labeled by focal lesion."""
    hue = "Focal lesion"
    mean_volume = (df[VOL_1] + df[vol_2]) / 2
    fig, axes = plt.subplots(1, 4 if dice else 3, figsize=(24 if dice else 18, 5))

    sns.scatterplot(data=df, x=VOL_1, y=vol_2, hue=hue, palette=PALETTE, ax=axes[0])
    max_vol = df[[VOL_1, vol_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set(title="Lesion volume", xlabel="Lesion volume run-01 (mm³)", ylabel=f"Lesion volume {run_2_name} (mm³)")

    # Bland-Altman plots of the volume difference in mm3 and in %
    for ax, diff, unit in ((axes[1], df[f"{prefix}_abs_diff_mm3"], "mm³"), (axes[2], df[f"{prefix}_abs_diff_percent"], "%")):
        sns.scatterplot(x=mean_volume, y=diff, hue=df[hue], palette=PALETTE, ax=ax)
        mean_diff, std_diff = diff.mean(), diff.std()
        for y, name in ((mean_diff, "mean"), (mean_diff + 1.96 * std_diff, "+1.96 SD")):
            ax.axhline(y, color="k", linestyle="--", linewidth=1)
            ax.annotate(f"{name}: {y:.1f}", xy=(1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8)
        ax.set(title=f"Bland-Altman of lesion volume ({unit})", xlabel="Mean lesion volume (mm³)", ylabel=f"Absolute lesion volume difference |{run_2_name} - run-01| ({unit})")

    if dice:
        sns.boxplot(data=df, x=hue, y=DICE, hue=hue, palette=PALETTE, legend=False, ax=axes[3], showfliers=False)
        sns.stripplot(data=df, x=hue, y=DICE, ax=axes[3], color="k")
        axes[3].set(title="Dice (run-02 registered to run-01)", ylabel="Dice")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the model lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("-m", "--mode", default="default", choices=MODES, help="Inference mode")
    parser.add_argument("--exclude-stitch", action="store_true", help="Exclude the stitch acquisitions")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    with open(study_dir / "study_files.json") as f:
        study = json.load(f)

    results = []
    for pair in tqdm([p for p in study["pairs"] if p["status"] == "success" and not (args.exclude_stitch and p["acquisition"] == "stitch")]):
        try:
            results.append(process_pair(pair, Path(study["dataset"]), study_dir, args.mode))
        except Exception as e:
            print(f"Failed for {pair['subject']}/{pair['session']}/acq-{pair['acquisition']}: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)

    # Focal lesion visible or not, from the has_lesion field (bool or string)
    labels = {"true": "visible", "false": "uncertain"}
    df["Focal lesion"] = df["has_lesion"].astype(str).str.strip().str.lower().map(labels)

    df.to_csv(args.output / "lesion_seg_variability.csv", index=False)

    # Only plot the labeled pairs, so that the Bland-Altman lines are computed on the plotted points
    # (all labeled pairs, and only the pairs with visible focal lesions)
    df = df.dropna(subset=["Focal lesion"])
    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        plot_variability(df_plot, "lesion_volume", "lesion_volume_run-02_mm3", "run-02", args.output / f"lesion_seg_variability_native{suffix}.png")
        plot_variability(df_plot, "lesion_volume_registered", "lesion_volume_run-02_registered_mm3", "run-02 registered",
                         args.output / f"lesion_seg_variability_registered{suffix}.png", dice=True)
    print(f"Results of {len(results)} pairs saved to {args.output}")


if __name__ == "__main__":
    main()
