"""
Evaluates the scan-rescan variability of the lesion segmentations predicted by the SCT model lesion-wise,
for each pair of study_files.json (output of generate_study_files.py), with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction>
or, with --soft-tta, with the soft segmentations with MONAI test-time augmentation (average of the predictions of
`sct_deepseg lesion_ms -average-per-fold` on 5 augmented versions of the image, same as
evaluate_model_soft_lesion_seg_variability.py), binarized at --threshold in the native space.

The run-02 prediction is registered to run-01 (using the warping field of generate_study_files.py). Lesions
are the 3D connected components of each segmentation. Lesions of both runs which overlap are grouped together
(connected components of the union of both segmentations), so that a lesion seen as one blob on one run and
as several smaller blobs on the other run is compared as a whole. A group is matched if the IoU between its
run-01 and run-02 lesions is at least --iou-threshold; otherwise, its lesions are reported as detected only
on run-01 and only on run-02. The run-02 lesion volumes are computed in the run-02 native space (each native
run-02 lesion is assigned to the groups it falls in once registered, in proportion to its registered voxels).

Outputs (in the output folder), with lesions labeled by focal lesion visible or uncertain (has_lesion field):
    - lesions.csv: one line per lesion group (matched, only on run-01 or only on run-02), with the number of
      lesions of each run in the group
    - matched_lesions.png: volumes of both runs, Bland-Altman plots (mm³ and %) and Dice vs mean
      lesion volume of the matched lesions
    - unmatched_lesions.png: number of lesion groups and volume of the lesions detected only on one run (with the matched
      lesions as reference)
    - matched_lesions_visible.png and unmatched_lesions_visible.png: same plots, only for the visible focal lesions
    - log.txt: log of the script
    - matched/ and not-matched/: screenshots of each lesion group (axial and sagittal views of run-01 and run-02
      registered, with the lesions in red), in a visible or uncertain sub-folder (focal lesion), named
      <subject>_<session>_acq-<acquisition>_group-<lesion_group>.png

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder
    --iou-threshold     Minimum IoU to match two lesions (default: 0.1)
    --exclude-stitch    Exclude the stitch acquisitions (included by default)
    --soft-tta          Use the soft segmentations with MONAI TTA (default: sct_deepseg lesion_ms default inference)
    --threshold         Binarization threshold of the soft segmentations (default: 0.5)

Author: Pierre-Louis Benveniste
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import binarize, label_and_register, load_study, match_lesions, pair_name, predict_binary, predict_soft, save_lesion_wise_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scan-rescan variability of the model lesion segmentations lesion-wise.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    parser.add_argument("--iou-threshold", default=0.1, type=float, help="Minimum IoU to match two lesions (default: 0.1)")
    parser.add_argument("--exclude-stitch", action="store_true", help="Exclude the stitch acquisitions")
    parser.add_argument("--soft-tta", action="store_true", help="Use the soft segmentations with MONAI TTA (binarized at --threshold)")
    parser.add_argument("--threshold", default=0.5, type=float, help="Binarization threshold of the soft segmentations (default: 0.5)")
    args = parser.parse_args()

    # Save the log in the output folder
    args.output.mkdir(parents=True, exist_ok=True)
    logger.add(args.output / "log.txt", mode="w")
    logger.info(f"Arguments: {vars(args)}")

    study_dir = args.input.resolve()
    dataset, pairs = load_study(study_dir)

    rows = []
    for pair in tqdm([p for p in pairs if not (args.exclude_stitch and p["acquisition"] == "stitch")]):
        try:
            # Inference on both runs (shared with evaluate_model_lesion_seg_variability.py and
            # evaluate_model_soft_lesion_seg_variability.py)
            preds = []
            for r, key in (("01", "scan_run-01"), ("02", "rescan_run-02")):
                out_dir = (study_dir / pair[f"sc_seg_run-{r}"]).parent
                if args.soft_tta:
                    # Soft segmentation with TTA, binarized in the native space
                    soft_pred = predict_soft(dataset / pair[key], out_dir, tta=True)
                    pred = Path(str(soft_pred).replace("_seg.nii.gz", f"-thr{args.threshold}_seg.nii.gz"))
                    if not pred.exists():
                        binarize(soft_pred, pred, args.threshold)
                else:
                    pred = predict_binary(dataset / pair[key], out_dir)
                preds.append(pred)
            # Label the run-02 lesions in the native space and register the label map to run-01
            labels_2 = Path(str(preds[1]).replace(".nii.gz", "_lesion-labels.nii.gz"))
            labels_2_reg = Path(str(labels_2).replace(".nii.gz", "_space-run01.nii.gz"))
            label_and_register(preds[1], labels_2, labels_2_reg, pair, dataset, study_dir)
            img_paths = (dataset / pair["scan_run-01"], study_dir / pair["rescan_run-02_reg_to_run-01"])
            rows += match_lesions(pair, img_paths, (preds[0], labels_2_reg), labels_2, args.iou_threshold, args.output)
        except Exception as e:
            logger.error(f"Failed for {pair_name(pair)}: {e}")

    save_lesion_wise_results(rows, args.output, logger)


if __name__ == "__main__":
    main()
