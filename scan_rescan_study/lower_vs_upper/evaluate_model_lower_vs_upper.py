"""
Compares the scan-rescan variability of the total lesion volume of the lesion segmentations predicted by the SCT
model between the lower and upper chunks (acquisition field), for each pair of study_files.json (output of
generate_study_files.py), with:
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <prediction>
(predictions shared with evaluate_model_lesion_seg_variability.py)

The total lesion volume of each run is computed in its native space, and the variability of a pair is the absolute
difference |run-02 - run-01| (mm³ and % of the mean volume, 0 % if both volumes are 0).

Outputs (in the output folder):
    - lower_vs_upper.csv: one line per pair (lower and upper chunks)
    - lower_vs_upper.png: total lesion volumes of both runs, absolute volume difference (mm³ and %) per chunk, and
      % difference of the lower and upper chunks of the same subject/session
    - lower_vs_upper_visible.png: same plots, only for the pairs with visible focal lesions (has_lesion field)
    - log.txt: log of the script, with the variability per chunk and the paired Wilcoxon test lower vs upper

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -o / --output       Path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import CHUNKS, load_study, pair_name, predict_binary, save_lower_vs_upper_results, total_volume_row


def main():
    parser = argparse.ArgumentParser(description="Compare the total lesion volume variability of the model lesion segmentations between the lower and upper chunks.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
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
            preds = [predict_binary(dataset / pair[key], out_dir) for key in ("scan_run-01", "rescan_run-02")]
            rows.append(total_volume_row(pair, *preds))
        except Exception as e:
            logger.error(f"Failed for {pair_name(pair)}: {e}")

    save_lower_vs_upper_results(rows, args.output, logger)


if __name__ == "__main__":
    main()
