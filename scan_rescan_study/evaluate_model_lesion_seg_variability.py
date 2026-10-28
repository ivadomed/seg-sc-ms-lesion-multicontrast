"""
Evaluates the scan-rescan variability of the lesion segmentations predicted by the SCT model, for each
pair of study_files.json (output of generate_study_files.py), with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction> [flags of the mode]

Modes:
    default         no flag
    tta             -test-time-aug
    single-fold     -single-fold

Metrics:
    - lesion volume and number of lesions of each run
    - lesion volume difference (mm3 and %) between run-01 and run-02, and between run-01 and run-02
      registered to run-01 (using the warping field of generate_study_files.py)
    - lesion volume difference (mm3 and %) between run-01 and run-02, each in its straightened space
      (sct_straighten_spinalcord with the SC seg)
    - Dice between the run-01 prediction and the run-02 prediction registered to run-01
    - Dice of each prediction relative to the manual segmentation when available

Outputs (in the output folder), with pairs labeled by focal lesion visible or uncertain (has_lesion field):
    - lesion_seg_variability.csv: one line per pair
    - lesion_seg_variability_native.png: lesion volumes and Bland-Altman plots (mm3 and %), run-01 vs run-02
    - lesion_seg_variability_registered.png: same, run-01 vs run-02 registered to run-01, and Dice
    - lesion_seg_variability_straight.png: same, run-01 vs run-02 in the straightened spaces
    - lesion_seg_variability_<native/registered/straight>_visible.png: same plots, only for the pairs
      with visible focal lesions

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    -m / --mode         Inference mode (default: default)
    --exclude-stitch    Exclude the stitch acquisitions (included by default)

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import (DICE, VOL_1, VOL_2, dice, focal_lesion, lesion_count, load_seg, load_study, pair_name, plot_variability, predict_binary,
                    straighten, volume_diff, warp_to_run01)

MODES = {"default": "", "tta": "-test-time-aug", "single-fold": "-single-fold"}


def process_pair(pair, dataset, study_dir, mode):
    out_dir = (study_dir / pair["sc_seg_run-01"]).parent

    # Inference on both runs and registration of run-02 to run-01
    preds = [predict_binary(dataset / pair[key], out_dir, MODES[mode]) for key in ("scan_run-01", "rescan_run-02")]
    pred_2_reg = warp_to_run01(preds[1], Path(str(preds[1]).replace(".nii.gz", "_space-run01.nii.gz")), pair, dataset, study_dir)

    # Predictions in the straightened space of each run
    preds_straight = [straighten(pred, dataset / pair[key], study_dir / pair[f"sc_seg_run-{r}"], study_dir / "qc")
                      for pred, key, r in zip(preds, ("scan_run-01", "rescan_run-02"), ("01", "02"))]

    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2), (seg_2_reg, _) = (load_seg(p) for p in (*preds, pred_2_reg))
    (seg_1_straight, voxel_vol_1_straight), (seg_2_straight, voxel_vol_2_straight) = (load_seg(p) for p in preds_straight)
    vol_1, vol_2, vol_2_reg = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2, seg_2_reg.sum() * voxel_vol_1
    vol_1_straight, vol_2_straight = seg_1_straight.sum() * voxel_vol_1_straight, seg_2_straight.sum() * voxel_vol_2_straight

    # Dice relative to the manual segmentations (if available)
    dice_manual = {r: dice(seg, load_seg(dataset / pair[f"lesion_seg_run-{r}"])[0]) if pair[f"lesion_seg_run-{r}"] else np.nan
                   for r, seg in (("01", seg_1), ("02", seg_2))}

    return {
        "subject": pair["subject"],
        "session": pair["session"],
        "acquisition": pair["acquisition"],
        "has_lesion": pair["has_lesion"],
        "lesion_count_run-01": lesion_count(seg_1),
        "lesion_count_run-02": lesion_count(seg_2),
        VOL_1: vol_1,
        VOL_2: vol_2,
        **volume_diff(vol_1, vol_2, "lesion_volume"),
        "lesion_volume_run-02_registered_mm3": vol_2_reg,
        **volume_diff(vol_1, vol_2_reg, "lesion_volume_registered"),
        "lesion_volume_run-01_straight_mm3": vol_1_straight,
        "lesion_volume_run-02_straight_mm3": vol_2_straight,
        **volume_diff(vol_1_straight, vol_2_straight, "lesion_volume_straight"),
        DICE: dice(seg_1, seg_2_reg),
        "dice_vs_manual_run-01": dice_manual["01"],
        "dice_vs_manual_run-02": dice_manual["02"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the model lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("-m", "--mode", default="default", choices=MODES, help="Inference mode")
    parser.add_argument("--exclude-stitch", action="store_true", help="Exclude the stitch acquisitions")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    dataset, pairs = load_study(study_dir)

    results = []
    for pair in tqdm([p for p in pairs if not (args.exclude_stitch and p["acquisition"] == "stitch")]):
        try:
            results.append(process_pair(pair, dataset, study_dir, args.mode))
        except Exception as e:
            print(f"Failed for {pair_name(pair)}: {e}")

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df["Focal lesion"] = df["has_lesion"].map(focal_lesion)
    df.to_csv(args.output / "lesion_seg_variability.csv", index=False)

    # Only plot the labeled pairs, so that the Bland-Altman lines are computed on the plotted points
    # (all labeled pairs, and only the pairs with visible focal lesions)
    df = df.dropna(subset=["Focal lesion"])
    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        plot_variability(df_plot, "lesion_volume", VOL_1, VOL_2, "run-01", "run-02",
                         args.output / f"lesion_seg_variability_native{suffix}.png")
        plot_variability(df_plot, "lesion_volume_registered", VOL_1, "lesion_volume_run-02_registered_mm3", "run-01", "run-02 registered",
                         args.output / f"lesion_seg_variability_registered{suffix}.png", dice=True)
        plot_variability(df_plot, "lesion_volume_straight", "lesion_volume_run-01_straight_mm3", "lesion_volume_run-02_straight_mm3",
                         "run-01 straight", "run-02 straight", args.output / f"lesion_seg_variability_straight{suffix}.png")
    print(f"Results of {len(results)} pairs saved to {args.output}")


if __name__ == "__main__":
    main()
