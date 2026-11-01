"""
Evaluates the scan-rescan variability of the soft lesion segmentations predicted by the SCT model, for
each pair of study_files.json (output of generate_study_files.py), with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction> -average-per-fold

Modes:
    soft            one inference per image
    soft-tta        test-time augmentation: the inference is run on 5 versions of the image (copy, affine,
                    bias field, contrast, noise) and the 5 soft predictions are averaged. The prediction
                    of the affine version is brought back to the native space before averaging.

Metrics (computed on the soft segmentations):
    - soft lesion volume of each run
    - lesion volume difference (mm3 and %) between run-01 and run-02, and between run-01 and run-02
      registered to run-01 (using the warping field of generate_study_files.py)
    - soft Dice between the run-01 prediction and the run-02 prediction registered to run-01
    - soft Dice of each prediction relative to the manual segmentation when available

Outputs (in the output folder), with pairs labeled by focal lesion visible or uncertain (has_lesion field):
    - lesion_seg_variability.csv: one line per pair
    - lesion_seg_variability_native.png: lesion volumes and Bland-Altman plots (mm3 and %), run-01 vs run-02
    - lesion_seg_variability_registered.png: same, run-01 vs run-02 registered to run-01, and Dice
    - lesion_seg_variability_<native/registered>_visible.png: same plots, only for the pairs with
      visible focal lesions
    - lesion_seg_variability_thresholds.csv and .png: same metrics on the soft segmentations without
      threshold and binarized at each threshold (0.1, 0.2, ..., 1.0): one line per pair and threshold,
      and boxplots per threshold

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    -m / --mode         Inference mode: soft or soft-tta (default: soft)
    --exclude-stitch    Exclude the stitch acquisitions (included by default)

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from common import DICE, PALETTE, VOL_1, VOL_2, dice, focal_lesion, load_seg, load_study, pair_name, plot_variability, predict_soft, volume_diff, warp_to_run01

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def process_pair(pair, dataset, study_dir, mode):
    out_dir = (study_dir / pair["sc_seg_run-01"]).parent

    # Inference on both runs and registration of run-02 to run-01
    preds = [predict_soft(dataset / pair[key], out_dir, tta=mode == "soft-tta") for key in ("scan_run-01", "rescan_run-02")]
    pred_2_reg = warp_to_run01(preds[1], Path(str(preds[1]).replace(".nii.gz", "_space-run01.nii.gz")), pair, dataset, study_dir, interp="linear")

    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2), (seg_2_reg, _) = (load_seg(p, soft=True) for p in (*preds, pred_2_reg))

    # Manual segmentations (if available)
    manual_1, manual_2 = (load_seg(dataset / pair[f"lesion_seg_run-{r}"])[0] if pair[f"lesion_seg_run-{r}"] else None for r in ("01", "02"))

    def metrics(seg_1, seg_2, seg_2_reg):
        vol_1, vol_2, vol_2_reg = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2, seg_2_reg.sum() * voxel_vol_1
        return {
            "subject": pair["subject"],
            "session": pair["session"],
            "acquisition": pair["acquisition"],
            "has_lesion": pair["has_lesion"],
            VOL_1: vol_1,
            VOL_2: vol_2,
            **volume_diff(vol_1, vol_2, "lesion_volume"),
            "lesion_volume_run-02_registered_mm3": vol_2_reg,
            **volume_diff(vol_1, vol_2_reg, "lesion_volume_registered"),
            DICE: dice(seg_1, seg_2_reg),
            "dice_vs_manual_run-01": dice(seg_1, manual_1) if manual_1 is not None else np.nan,
            "dice_vs_manual_run-02": dice(seg_2, manual_2) if manual_2 is not None else np.nan,
        }

    # Metrics on the soft segmentations (no threshold), and on the segmentations binarized at each threshold
    soft_metrics = metrics(seg_1, seg_2, seg_2_reg)
    thresholds_metrics = [{"threshold": str(thr), **metrics(seg_1 >= thr, seg_2 >= thr, seg_2_reg >= thr)} for thr in THRESHOLDS]
    return soft_metrics, [{"threshold": "no threshold", **soft_metrics}] + thresholds_metrics


def plot_thresholds(df, output):
    """Plots the variability metrics of the segmentations binarized at each threshold, labeled by focal lesion."""
    df = df.assign(dice_vs_manual=df[["dice_vs_manual_run-01", "dice_vs_manual_run-02"]].mean(axis=1))
    panels = {
        "lesion_volume_abs_diff_percent": "Absolute lesion volume difference |run-02 - run-01| (%)",
        "lesion_volume_registered_abs_diff_percent": "Absolute lesion volume difference |run-02 registered - run-01| (%)",
        DICE: "Dice (run-02 registered to run-01)",
        "dice_vs_manual": "Dice relative to the manual segmentation (mean of both runs)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    for ax, (metric, title) in zip(axes.flat, panels.items()):
        sns.boxplot(data=df, x="threshold", y=metric, hue="Focal lesion", palette=PALETTE, ax=ax, showfliers=False,
                    order=["no threshold"] + [str(thr) for thr in THRESHOLDS])
        ax.set(title=title, xlabel="Binarization threshold", ylabel="")
        if "volume" in metric:
            sns.move_legend(ax, "upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the model soft lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("-m", "--mode", default="soft", choices=["soft", "soft-tta"], help="Inference mode")
    parser.add_argument("--exclude-stitch", action="store_true", help="Exclude the stitch acquisitions")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    dataset, pairs = load_study(study_dir)

    results, results_thresholds = [], []
    for pair in tqdm([p for p in pairs if not (args.exclude_stitch and p["acquisition"] == "stitch")]):
        try:
            soft_metrics, thresholds_metrics = process_pair(pair, dataset, study_dir, args.mode)
            results.append(soft_metrics)
            results_thresholds += thresholds_metrics
        except Exception as e:
            print(f"Failed for {pair_name(pair)}: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    df, df_thresholds = pd.DataFrame(results), pd.DataFrame(results_thresholds)
    for d in (df, df_thresholds):
        d["Focal lesion"] = d["has_lesion"].map(focal_lesion)
    df.to_csv(args.output / "lesion_seg_variability.csv", index=False)
    df_thresholds.to_csv(args.output / "lesion_seg_variability_thresholds.csv", index=False)
    plot_thresholds(df_thresholds.dropna(subset=["Focal lesion"]), args.output / "lesion_seg_variability_thresholds.png")

    # Only plot the labeled pairs (all of them, and only the pairs with visible focal lesions)
    df = df.dropna(subset=["Focal lesion"])
    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        plot_variability(df_plot, "lesion_volume", VOL_1, VOL_2, "run-01", "run-02",
                         args.output / f"lesion_seg_variability_native{suffix}.png", soft=True)
        plot_variability(df_plot, "lesion_volume_registered", VOL_1, "lesion_volume_run-02_registered_mm3", "run-01", "run-02 registered",
                         args.output / f"lesion_seg_variability_registered{suffix}.png", dice=True, soft=True)
    print(f"Results of {len(results)} pairs saved to {args.output}")


if __name__ == "__main__":
    main()
