"""
Evaluates the scan-rescan variability of the manual lesion segmentations, for each pair of
study_files.json (output of generate_study_files.py) which has a manual lesion segmentation for both runs:
    - lesion volume and number of lesions of each run
    - lesion volume difference (mm3 and %) between run-01 and run-02, and between run-01 and run-02
      registered to run-01 (using the warping field of generate_study_files.py)
    - lesion volume difference (mm3 and %) between run-01 and run-02, each in its straightened space
      (sct_straighten_spinalcord with the SC seg)
    - Dice between the run-01 lesion seg and the run-02 lesion seg registered to run-01

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

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from common import DICE, VOL_1, VOL_2, dice, focal_lesion, lesion_count, load_seg, load_study, pair_name, plot_variability, straighten, volume_diff, warp_to_run01


def process_pair(pair, dataset, study_dir):
    les_1, les_2 = dataset / pair["lesion_seg_run-01"], dataset / pair["lesion_seg_run-02"]

    # Register the run-02 lesion seg to run-01
    les_2_reg = warp_to_run01(les_2, (study_dir / pair["sc_seg_run-02"]).parent / les_2.name.replace(".nii.gz", "_space-run01.nii.gz"), pair, dataset, study_dir)

    # Lesion segs in the straightened space of each run
    les_1_straight = straighten(les_1, dataset / pair["scan_run-01"], study_dir / pair["sc_seg_run-01"], study_dir / "qc")
    les_2_straight = straighten(les_2, dataset / pair["rescan_run-02"], study_dir / pair["sc_seg_run-02"], study_dir / "qc")

    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2), (seg_2_reg, _) = load_seg(les_1), load_seg(les_2), load_seg(les_2_reg)
    (seg_1_straight, voxel_vol_1_straight), (seg_2_straight, voxel_vol_2_straight) = load_seg(les_1_straight), load_seg(les_2_straight)
    vol_1, vol_2, vol_2_reg = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2, seg_2_reg.sum() * voxel_vol_1
    vol_1_straight, vol_2_straight = seg_1_straight.sum() * voxel_vol_1_straight, seg_2_straight.sum() * voxel_vol_2_straight

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
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the manual lesion segmentations.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    args = parser.parse_args()

    study_dir = args.input.resolve()
    dataset, pairs = load_study(study_dir)

    results = []
    for pair in tqdm([p for p in pairs if p["lesion_seg_run-01"] and p["lesion_seg_run-02"]]):
        try:
            results.append(process_pair(pair, dataset, study_dir))
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
