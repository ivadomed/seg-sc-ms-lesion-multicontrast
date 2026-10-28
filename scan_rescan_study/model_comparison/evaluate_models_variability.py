"""
Compares the scan-rescan variability of the total lesion volume across several lesion segmentation models, for each
lower and upper pair of study_files.json (output of generate_study_files.py).

The models are listed in a JSON config file (e.g. models.json) mapping each model name (used in the prediction file
names, so no spaces) to its release and the links of the release assets of its folds:
    {"model_v1": {"release": "r20250909", "url": {"model_fold0": ["https://.../model_fold0.zip"], ...}}}
Each model is installed in SCT with all its folds and run on both runs of each pair (predictions which already exist
are not recomputed, and the model is only installed if some predictions are missing):
    sct_deepseg lesion_ms -install -custom-url <fold0 link> <fold1 link> ...
    SCT_USE_GPU=1 sct_deepseg lesion_ms -i <image> -o <image>_label-lesion_desc-<model>_seg.nii.gz
The default lesion_ms model is reinstalled at the end if another model was installed.

The total lesion volume of each run is computed in its native space, and the variability of a pair is the absolute
difference |run-02 - run-01| (mm³ and % of the mean volume, 0 % if both volumes are 0). The Dice of each prediction
relative to the manual lesion segmentation of the same run (when available, dice_score of evaluation/utils.py) is
also computed, and averaged over both runs of each pair.

Outputs (in the output folder):
    - models_variability.csv: one line per pair and model
    - models_variability.png: total lesion volumes of both runs (colored by model), absolute volume difference
      (mm³ and %) and Dice relative to the manual segmentation per model (points colored by chunk)
    - models_variability_visible.png: same plots, only for the pairs with visible focal lesions (has_lesion field)
    - log.txt: log of the script, with the variability and Dice per model and the paired Wilcoxon tests between
      models (on the pairs processed by both models)

Arguments:
    -i / --input        Path to the output folder of generate_study_files.py
    -c / --config       Path to the JSON config file of the models
    -o / --output       Path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import wilcoxon
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import CHUNK_PALETTE, CHUNKS, DIFF, VOL_1, VOL_2, dice, focal_lesion, load_seg, load_study, pair_name, run, total_volume_row

METRICS = ["lesion_volume_abs_diff_mm3", DIFF, "dice_vs_manual"]


def plot_models(df, models, output):
    """Plots the total lesion volumes of both runs (colored by model), and the absolute volume difference (mm³ and %)
    and the Dice relative to the manual segmentation per model (points colored by chunk)."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    sns.scatterplot(data=df, x=VOL_1, y=VOL_2, hue="model", hue_order=models, ax=axes[0])
    max_vol = df[[VOL_1, VOL_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set(title="Total lesion volume", xlabel="Lesion volume run-01 (mm³)", ylabel="Lesion volume run-02 (mm³)")

    titles = {
        METRICS[0]: ("Total lesion volume difference (mm³)", "Absolute lesion volume difference |run-02 - run-01| (mm³)"),
        METRICS[1]: ("Total lesion volume difference (%)", "Absolute lesion volume difference |run-02 - run-01| (%)"),
        METRICS[2]: ("Dice relative to the manual segmentation", "Dice (mean of both runs)"),
    }
    for ax, (metric, (title, ylabel)) in zip(axes[1:], titles.items()):
        sns.boxplot(data=df, x="model", y=metric, order=models, color="lightgray", ax=ax, showfliers=False)
        sns.stripplot(data=df, x="model", y=metric, order=models, hue="acquisition", hue_order=CHUNKS, palette=CHUNK_PALETTE, ax=ax)
        ax.set(title=title, xlabel="Model", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare the scan-rescan variability of the total lesion volume across several lesion segmentation models.")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to the output folder of generate_study_files.py")
    parser.add_argument("-c", "--config", required=True, type=Path, help="Path to the JSON config file of the models")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    args = parser.parse_args()

    # Save the log in the output folder
    args.output.mkdir(parents=True, exist_ok=True)
    logger.add(args.output / "log.txt", mode="w")
    logger.info(f"Arguments: {vars(args)}")

    with open(args.config) as f:
        models = json.load(f)
    logger.info(f"Models: {models}")

    study_dir = args.input.resolve()
    dataset, pairs = load_study(study_dir)
    pairs = [p for p in pairs if p["acquisition"] in CHUNKS]

    rows, installed = [], False
    for model, config in models.items():
        urls = " ".join(url for fold_urls in config["url"].values() for url in fold_urls)
        # Prediction paths of both runs of each pair
        preds = {pair_name(p): [(study_dir / p["sc_seg_run-01"]).parent / Path(p[key]).name.replace(".nii.gz", f"_label-lesion_desc-{model}_seg.nii.gz")
                                for key in ("scan_run-01", "rescan_run-02")] for p in pairs}
        if not all(pred.exists() for paths in preds.values() for pred in paths):
            try:
                logger.info(f"Installing {model} (release {config['release']}, {len(config['url'])} folds)")
                run(f"sct_deepseg lesion_ms -install -custom-url {urls}")
                installed = True
            except Exception as e:
                logger.error(f"Installation failed for {model}, skipped: {e}")
                continue

        for pair in tqdm(pairs, desc=model):
            try:
                for key, pred in zip(("scan_run-01", "rescan_run-02"), preds[pair_name(pair)]):
                    if not pred.exists():
                        run(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {dataset / pair[key]} -o {pred}")
                # Dice relative to the manual segmentation of each run (if available)
                dice_manual = {f"dice_vs_manual_run-{r}": dice(load_seg(pred)[0], load_seg(dataset / pair[f"lesion_seg_run-{r}"])[0])
                               if pair[f"lesion_seg_run-{r}"] else np.nan for r, pred in zip(("01", "02"), preds[pair_name(pair)])}
                rows.append({"model": model, **total_volume_row(pair, *preds[pair_name(pair)]), **dice_manual,
                             "dice_vs_manual": np.nanmean(list(dice_manual.values())) if any(pair[f"lesion_seg_run-{r}"] for r in ("01", "02")) else np.nan})
            except Exception as e:
                logger.error(f"Failed for {model} {pair_name(pair)}: {e}")

    # Restore the default lesion_ms model
    if installed:
        run("sct_deepseg lesion_ms -install")
        logger.info("Default lesion_ms model reinstalled")

    df = pd.DataFrame(rows)
    df["Focal lesion"] = df["has_lesion"].map(focal_lesion)
    df.to_csv(args.output / "models_variability.csv", index=False)

    models = [m for m in models if m in set(df["model"])]
    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        logger.info(f"Pairs{suffix or ' (all)'}: total lesion volume variability and Dice per model:\n"
                    f"{df_plot.groupby('model')[METRICS].describe().T.round(2)}")
        # Paired Wilcoxon tests between models, on the pairs processed by both models
        for metric in METRICS:
            paired = df_plot.pivot_table(index=["subject", "session", "acquisition"], columns="model", values=metric).reindex(columns=models)
            for model_1, model_2 in combinations(models, 2):
                both = paired[[model_1, model_2]].dropna()
                try:
                    p = wilcoxon(both[model_1], both[model_2]).pvalue
                except ValueError as e:  # e.g. no pair, or all differences equal to 0
                    p = f"not computed ({e})"
                logger.info(f"Pairs{suffix or ' (all)'}: {metric}, paired Wilcoxon {model_1} vs {model_2} (n = {len(both)}): p = {p}")
        plot_models(df_plot, models, args.output / f"models_variability{suffix}.png")
    logger.info(f"Results of {len(df)} pairs x models saved to {args.output}")


if __name__ == "__main__":
    main()
