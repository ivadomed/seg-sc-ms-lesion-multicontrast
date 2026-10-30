"""
Saves screenshots comparing the lesion segmentations of two models (default: r20250909 = MSJ and r20261029 =
MSJ+Auglab) across scan and rescan, for each lower and upper pair of study_files.json (output of
generate_study_files.py). The predictions must already exist (computed by evaluate_models_variability.py, named
<image>_label-lesion_desc-<model>_seg.nii.gz).

The run-02 predictions are registered to run-01 (using the warping field of generate_study_files.py). Lesion groups
are the 3D connected components of the union of the 4 segmentations (both models, both runs), so that each screenshot
shows the same lesion area for both runs and both models. Each screenshot shows the axial and sagittal views through
the largest section of the group (rows: axial and sagittal; columns: model 1 run-01, model 1 run-02 registered,
model 2 run-01, model 2 run-02 registered), with the lesions in red.

Outputs (in the output folder):
    - <visible/uncertain/unknown>/<subject>_<session>_acq-<acquisition>_group-<lesion_group>.png: one screenshot per
      lesion group, in a sub-folder per focal lesion (has_lesion field)
    - lesion_groups.csv: one line per lesion group, with the volume of each model and run (run-02 in the registered
      space) and the scan-rescan Dice of each model in the group
    - log.txt: log of the script

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    --models            Names of the two models in the prediction file names (default: r20250909 r20261029)
    --names             Names of the two models in the screenshots (default: MSJ MSJ+Auglab)

Author: Pierre-Louis Benveniste
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import ndimage
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import CHUNKS, STRUCTURE, dice, focal_lesion, load, load_study, pair_name, warp_to_run01

RUNS = ("run-01", "run-02 registered")


def save_screenshot(images, segs, names, group, zooms, title, output, margin=15):
    """Saves the axial (top row) and sagittal (bottom row) views (through the largest section of a lesion group) of both
    runs of each model side by side (columns: model 1 run-01, model 1 run-02, model 2 run-01, model 2 run-02), with the
    lesions in red."""
    x, z = group.sum(axis=(1, 2)).argmax(), group.sum(axis=(0, 1)).argmax()
    xs, ys, zs = (np.nonzero(group.any(axis=axis))[0] for axis in ((1, 2), (0, 2), (0, 1)))
    crop = lambda c: slice(max(c.min() - margin, 0), c.max() + margin + 1)
    views = {"axial": (lambda a: a[crop(xs), crop(ys), z], zooms[1] / zooms[0]),
             "sagittal": (lambda a: a[x, crop(ys), crop(zs)], zooms[2] / zooms[1])}

    fig, axes = plt.subplots(2, 2 * len(names), figsize=(8 * len(names), 8))
    for m, name in enumerate(names):
        for v, (view, (cut, aspect)) in enumerate(views.items()):
            for r, (run_name, image, seg) in enumerate(zip(RUNS, images, segs[m])):
                ax = axes[v, 2 * m + r]
                ax.imshow(cut(image).T, cmap="gray", origin="lower", aspect=aspect)
                if cut(seg).any():
                    ax.contour(cut(seg).T.astype(float), levels=[0.5], colors="red", linewidths=1)
                ax.set_title(f"{name} - {run_name} ({view})")
                ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save screenshots comparing the lesion segmentations of two models across scan and rescan.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("--models", nargs=2, default=["r20250909", "r20261029"], help="Names of the two models in the prediction file names")
    parser.add_argument("--names", nargs=2, default=["MSJ", "MSJ+Auglab"], help="Names of the two models in the screenshots")
    args = parser.parse_args()

    # Save the log in the output folder
    args.output.mkdir(parents=True, exist_ok=True)
    logger.add(args.output / "log.txt", mode="w")
    logger.info(f"Arguments: {vars(args)}")

    study_dir = args.input.resolve()
    dataset, pairs = load_study(study_dir)

    rows = []
    for pair in tqdm([p for p in pairs if p["acquisition"] in CHUNKS]):
        try:
            out_dir = (study_dir / pair["sc_seg_run-01"]).parent
            # Predictions of both runs for each model, run-02 registered to run-01
            seg_paths = []
            for model in args.models:
                pred_1, pred_2 = (out_dir / Path(pair[key]).name.replace(".nii.gz", f"_label-lesion_desc-{model}_seg.nii.gz")
                                  for key in ("scan_run-01", "rescan_run-02"))
                pred_2_reg = warp_to_run01(pred_2, Path(str(pred_2).replace(".nii.gz", "_space-run01.nii.gz")), pair, dataset, study_dir)
                seg_paths.append((pred_1, pred_2_reg))

            (img_1, zooms), (img_2, _) = load(dataset / pair["scan_run-01"]), load(study_dir / pair["rescan_run-02_reg_to_run-01"])
            segs = [tuple(load(p)[0] > 0.5 for p in paths) for paths in seg_paths]
            voxel_vol = np.prod(zooms)
            # Lesion groups: connected components of the union of the 4 segmentations
            groups, _ = ndimage.label(np.any([s for model_segs in segs for s in model_segs], axis=0), structure=STRUCTURE)

            name = pair_name(pair).replace("/", "_")
            for g, box in enumerate(ndimage.find_objects(groups), start=1):
                group = groups[box] == g
                row = {**{k: pair[k] for k in ("subject", "session", "acquisition", "has_lesion")}, "lesion_group": g}
                title = []
                for model, model_name, (seg_1, seg_2) in zip(args.models, args.names, segs):
                    group_1, group_2 = seg_1[box] & group, seg_2[box] & group
                    vol_1, vol_2 = group_1.sum() * voxel_vol, group_2.sum() * voxel_vol
                    row.update({f"{model}_volume_run-01_mm3": vol_1, f"{model}_volume_run-02_registered_mm3": vol_2,
                                f"{model}_dice_run-02_registered_to_run-01": dice(group_1, group_2)})
                    title.append(f"{model_name}: {vol_1:.1f} vs {vol_2:.1f} mm³, Dice {row[f'{model}_dice_run-02_registered_to_run-01']:.2f}")
                rows.append(row)
                save_screenshot((img_1, img_2), segs, args.names, groups == g, zooms, f"{name} - group {g}\n" + "\n".join(title),
                                args.output / (focal_lesion(pair["has_lesion"]) or "unknown") / f"{name}_group-{g:02d}.png")
        except Exception as e:
            logger.error(f"Failed for {pair_name(pair)}: {e}")

    pd.DataFrame(rows).to_csv(args.output / "lesion_groups.csv", index=False)
    logger.info(f"Screenshots of {len(rows)} lesion groups saved to {args.output}")


if __name__ == "__main__":
    main()
