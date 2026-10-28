"""
Functions shared by the scan-rescan study scripts (study files, SCT commands, inference, metrics, plots and
lesion-wise matching).

Author: Pierre-Louis Benveniste
"""

import json
import os
import sys
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parent.parent / "evaluation"))
from utils import dice_score

PALETTE = {"visible": "tab:orange", "uncertain": "tab:blue"}
VOL_1, VOL_2 = "lesion_volume_run-01_mm3", "lesion_volume_run-02_mm3"
DICE = "dice_run-02_registered_to_run-01"
DIFF = "lesion_volume_abs_diff_percent"
STATUS = ["matched", "only run-01", "only run-02"]
STRUCTURE = np.ones((3, 3, 3))  # 26-connectivity of the lesions (3D connected components)
GROUPS_EXPLANATION = (
    "A lesion group is a connected component of the union of the run-01 and run-02 (registered) segmentations, so it\n"
    "can contain several lesions of each run. Each matched group gives one 'matched' row, and each not matched group\n"
    "gives one 'only run-01' row and/or one 'only run-02' row: the plots count these rows (lesion groups), not lesions.\n"
    "Example of a not matched group with 3 lesions:\n"
    "    run-01:   ##        ##          two small lesions\n"
    "    run-02:     ########            one elongated lesion touching both\n"
    "    union:    ############          -> one lesion group\n"
    "If IoU < threshold: one 'only run-01' row (2 lesions) and one 'only run-02' row (1 lesion), so 2 rows for 3 lesions.\n"
    "'only run-0x' rows may thus still overlap lesions of the other run (0 < IoU < threshold)."
)


# ---------------------------------------------------------------- Study files and SCT commands

def run(cmd):
    """Runs a shell command and raises an error if it fails."""
    if os.system(cmd) != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def load_study(study_dir):
    """Loads study_files.json (output of generate_study_files.py) and returns the dataset path and the pairs processed successfully."""
    with open(study_dir / "study_files.json") as f:
        study = json.load(f)
    return Path(study["dataset"]), [p for p in study["pairs"] if p["status"] == "success"]


def pair_name(pair):
    return f"{pair['subject']}/{pair['session']}/acq-{pair['acquisition']}"


def focal_lesion(has_lesion):
    """Focal lesion visible or uncertain from the has_lesion field (bool or string), None if not labeled."""
    return {"true": "visible", "false": "uncertain"}.get(str(has_lesion).strip().lower())


def warp_to_run01(seg, output, pair, dataset, study_dir, interp="nn"):
    """Registers a run-02 image/segmentation to run-01 with the warping field of generate_study_files.py."""
    if not output.exists():
        run(f"sct_apply_transfo -i {seg} -d {dataset / pair['scan_run-01']} -w {study_dir / pair['warp_run-02_to_run-01']} -x {interp} -o {output}")
    return output


def straighten(seg, image, sc_seg, qc_dir):
    """Warps a segmentation to the straightened space of its image (straightening computed from the SC seg)."""
    folder = Path(str(sc_seg).replace(".nii.gz", "_straightening"))
    if not (folder / "warp_curve2straight.nii.gz").exists():
        run(f"sct_straighten_spinalcord -i {image} -s {sc_seg} -disable-straight2curved -ofolder {folder} -qc {qc_dir}")
    seg_straight = folder / seg.name.replace(".nii.gz", "_straight.nii.gz")
    if not seg_straight.exists():
        run(f"sct_apply_transfo -i {seg} -d {folder / 'straight_ref.nii.gz'} -w {folder / 'warp_curve2straight.nii.gz'} -x nn -o {seg_straight}")
    return seg_straight


# ---------------------------------------------------------------- Inference

def predict_binary(image, out_dir, flags=""):
    """Predicts the lesion segmentation of an image with sct_deepseg lesion_ms [flags] (if it does not exist yet)."""
    pred = out_dir / image.name.replace(".nii.gz", f"_label-lesion_desc-{flags.lstrip('-') or 'default'}_seg.nii.gz")
    if not pred.exists():
        run(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {image} -o {pred} {flags}")
    return pred


def predict(image, pred):
    """Predicts the soft lesion segmentation of an image."""
    run(f"SCT_USE_GPU=1 sct_deepseg lesion_ms -i {image} -o {pred} -average-per-fold")


def predict_tta(image, pred, seed=0):
    """Averages the soft predictions of 5 augmented versions of an image."""
    import torch
    from monai.data import MetaTensor
    from monai.transforms import Identityd, RandAdjustContrastd, RandAffined, RandBiasFieldd, RandGaussianNoised

    nii = nib.load(image)
    header = nii.header.copy()
    header.set_data_dtype(np.float32)
    data = {"image": MetaTensor(torch.from_numpy(nii.get_fdata(dtype=np.float32)[None]))}
    zooms = np.array(nii.header.get_zooms()[:3])
    thick = zooms > 1.5 * zooms.min()  # thick-slice axes: rotation only around them, translation and scaling only along the others
    versions = {
        "copy": Identityd(keys="image"),
        "affine": RandAffined(keys="image", prob=1, rotate_range=0.05 * thick, translate_range=2 * ~thick, scale_range=0.05 * ~thick, mode="bilinear"),
        "bias": RandBiasFieldd(keys="image", prob=1, coeff_range=(-0.02, 0.02)),
        "contrast": RandAdjustContrastd(keys="image", prob=1, gamma=(0.9, 1.1)),
        "noise": RandGaussianNoised(keys="image", prob=1, std=0.05 * float(data["image"].std())),
    }

    preds = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, transform in versions.items():
            if name != "copy":
                transform.set_random_state(seed=seed)
            augmented = transform({"image": data["image"].clone()})["image"]
            augmented_path, augmented_pred = Path(tmp_dir) / f"{name}.nii.gz", Path(tmp_dir) / f"{name}_pred.nii.gz"
            nib.save(nib.Nifti1Image(augmented[0].numpy(), nii.affine, header), augmented_path)
            predict(augmented_path, augmented_pred)
            pred_data = nib.load(augmented_pred).get_fdata(dtype=np.float32)
            if name == "affine":
                # Bring the prediction back to the native space with the inverse of the affine transform
                pred_data = MetaTensor(torch.from_numpy(pred_data[None]), applied_operations=augmented.applied_operations)
                pred_data = transform.inverse({"image": pred_data})["image"][0].numpy()
            preds.append(pred_data)
    nib.save(nib.Nifti1Image(np.mean(preds, axis=0), nii.affine, header), pred)


def predict_soft(image, out_dir, tta=False):
    """Predicts the soft lesion segmentation of an image, with or without TTA (if it does not exist yet)."""
    pred = out_dir / image.name.replace(".nii.gz", f"_label-lesion_desc-{'soft-tta' if tta else 'soft'}_seg.nii.gz")
    if not pred.exists():
        (predict_tta if tta else predict)(image, pred)
    return pred


def binarize(soft_pred, pred, threshold):
    """Binarizes a soft segmentation (voxels >= threshold)."""
    nii = nib.load(soft_pred)
    header = nii.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image((nii.get_fdata() >= threshold).astype(np.uint8), nii.affine, header), pred)


# ---------------------------------------------------------------- Metrics

def load_seg(path, soft=False):
    """Loads a segmentation (binarized at 0.5, or soft) and its voxel volume."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=float)
    return (data if soft else data > 0.5), np.prod(img.header.get_zooms()[:3])


def dice(seg_1, seg_2):
    """(Soft) Dice score of evaluation/utils.py (smooth=1, so 1 if both segmentations are empty)."""
    return dice_score(np.asarray(seg_1, dtype=float), np.asarray(seg_2, dtype=float))


def volume_diff(vol_1, vol_2, prefix):
    """Absolute volume difference |vol_2 - vol_1| in mm3 and in % of the mean volume (0 % if both volumes are 0)."""
    return {
        f"{prefix}_abs_diff_mm3": abs(vol_2 - vol_1),
        f"{prefix}_abs_diff_percent": abs(vol_2 - vol_1) / ((vol_1 + vol_2) / 2) * 100 if vol_1 + vol_2 > 0 else 0.0,
    }


def lesion_count(seg):
    return ndimage.label(seg, structure=STRUCTURE)[1]


# ---------------------------------------------------------------- Plots

def bland_altman(ax, mean_volume, diff, hue, unit, run_names=("run-01", "run-02")):
    """Bland-Altman plot of the absolute volume difference (mean and +1.96 SD lines)."""
    sns.scatterplot(x=mean_volume, y=diff, hue=hue, palette=PALETTE, ax=ax)
    mean_diff, std_diff = diff.mean(), diff.std()
    for y, name in ((mean_diff, "mean"), (mean_diff + 1.96 * std_diff, "+1.96 SD")):
        ax.axhline(y, color="k", linestyle="--", linewidth=1)
        ax.annotate(f"{name}: {y:.1f}", xy=(1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8)
    ax.set(title=f"Bland-Altman of lesion volume ({unit})", xlabel="Mean lesion volume (mm³)",
           ylabel=f"Absolute lesion volume difference |{run_names[1]} - {run_names[0]}| ({unit})")
    sns.move_legend(ax, "upper right")


def plot_variability(df, prefix, vol_1, vol_2, run_1_name, run_2_name, output, dice=False, soft=False):
    """Plots the lesion volumes, their Bland-Altman plots (mm3 and %) and optionally the Dice, labeled by focal lesion."""
    hue = "Focal lesion"
    fig, axes = plt.subplots(1, 4 if dice else 3, figsize=(24 if dice else 18, 5))

    sns.scatterplot(data=df, x=vol_1, y=vol_2, hue=hue, palette=PALETTE, ax=axes[0])
    max_vol = df[[vol_1, vol_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set(title=f"{'Soft l' if soft else 'L'}esion volume", xlabel=f"Lesion volume {run_1_name} (mm³)", ylabel=f"Lesion volume {run_2_name} (mm³)")

    for ax, unit in ((axes[1], "mm3"), (axes[2], "percent")):
        bland_altman(ax, (df[vol_1] + df[vol_2]) / 2, df[f"{prefix}_abs_diff_{unit}"], df[hue], {"mm3": "mm³", "percent": "%"}[unit], (run_1_name, run_2_name))

    if dice:
        sns.boxplot(data=df, x=hue, y=DICE, hue=hue, palette=PALETTE, legend=False, ax=axes[3], showfliers=False)
        sns.stripplot(data=df, x=hue, y=DICE, ax=axes[3], color="k")
        axes[3].set(title=f"{'Soft Dice' if soft else 'Dice'} (run-02 registered to run-01)", ylabel="Dice")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Lesion-wise matching

def load(path):
    """Loads an image in RAS orientation and its voxel size."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj, dtype=float), np.array(img.header.get_zooms()[:3])


def save_screenshot(images, segs, group, zooms, title, output, margin=15):
    """Saves axial and sagittal views (through the largest section of a lesion group) of both runs, with the lesions in red."""
    x, z = group.sum(axis=(1, 2)).argmax(), group.sum(axis=(0, 1)).argmax()
    xs, ys, zs = (np.nonzero(group.any(axis=axis))[0] for axis in ((1, 2), (0, 2), (0, 1)))
    crop = lambda c: slice(max(c.min() - margin, 0), c.max() + margin + 1)
    views = {"axial": (lambda a: a[crop(xs), crop(ys), z], zooms[1] / zooms[0]),
             "sagittal": (lambda a: a[x, crop(ys), crop(zs)], zooms[2] / zooms[1])}

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for row, (view, (cut, aspect)) in enumerate(views.items()):
        for col, (run_name, image, seg) in enumerate(zip(("run-01", "run-02 registered"), images, segs)):
            axes[row, col].imshow(cut(image).T, cmap="gray", origin="lower", aspect=aspect)
            if cut(seg).any():
                axes[row, col].contour(cut(seg).T.astype(float), levels=[0.5], colors="red", linewidths=1)
            axes[row, col].set_title(f"{run_name} ({view})")
            axes[row, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def label_and_register(seg, labels, labels_reg, pair, dataset, study_dir):
    """Labels the lesions (3D connected components) of the run-02 seg in its native space and registers the
    label map to run-01."""
    if not labels.exists():
        nii = nib.load(seg)
        data, _ = ndimage.label(nii.get_fdata() > 0.5, structure=STRUCTURE)
        header = nii.header.copy()
        header.set_data_dtype(np.int16)
        nib.save(nib.Nifti1Image(data.astype(np.int16), nii.affine, header), labels)
    warp_to_run01(labels, labels_reg, pair, dataset, study_dir)


def match_lesions(pair, img_paths, seg_paths, labels_2_native, iou_threshold, output):
    """Groups the run-01 and run-02 lesions (registered to run-01), saves a screenshot of each group in the
    matched or not-matched folder, and returns one row per lesion group. The run-02 volumes are computed in the
    run-02 native space: each native run-02 lesion adds its native volume to the groups it falls in once
    registered, in proportion to its registered voxels in each group."""
    (img_1, zooms), (img_2, _) = (load(p) for p in img_paths)
    seg_1 = load(seg_paths[0])[0] > 0.5
    labels_2 = np.rint(load(seg_paths[1])[0]).astype(int)  # run-02 lesion labels registered to run-01
    seg_2 = labels_2 > 0
    native_labels_2, zooms_2 = load(labels_2_native)
    native_vol_2 = np.bincount(np.rint(native_labels_2).astype(int).ravel()) * np.prod(zooms_2)
    reg_count_2 = np.bincount(labels_2.ravel(), minlength=len(native_vol_2))
    voxel_vol = np.prod(zooms)
    # Lesion groups: connected components of the union of both segmentations, so that overlapping lesions of both
    # runs (e.g. one lesion on run-01 seen as several smaller lesions on run-02) are in the same group
    groups, _ = ndimage.label(seg_1 | seg_2, structure=STRUCTURE)

    name = pair_name(pair).replace("/", "_")
    rows = []
    for g, box in enumerate(ndimage.find_objects(groups), start=1):
        base = {**{k: pair[k] for k in ("subject", "session", "acquisition", "has_lesion")}, "lesion_group": g}
        group = groups[box] == g
        group_1, group_2 = seg_1[box] & group, seg_2[box] & group
        n_1 = lesion_count(group_1)
        vol_1 = group_1.sum() * voxel_vol
        count_2 = np.bincount((labels_2[box] * group).ravel(), minlength=len(native_vol_2))[1:]
        in_group = count_2 > 0
        n_2 = in_group.sum()
        vol_2 = np.sum(native_vol_2[1:][in_group] * count_2[in_group] / reg_count_2[1:][in_group])
        iou = (group_1 & group_2).sum() / group.sum()
        matched = iou >= iou_threshold
        save_screenshot((img_1, img_2), (seg_1, seg_2), groups == g, zooms,
                        f"{name} - group {g} - IoU {iou:.2f} - {vol_1:.1f} mm³ (run-01) vs {vol_2:.1f} mm³ (run-02)",
                        output / ("matched" if matched else "not-matched") / (focal_lesion(pair["has_lesion"]) or "unknown") / f"{name}_group-{g:02d}.png")
        if matched:
            rows.append({
                **base,
                "status": "matched",
                "lesion_count_run-01": n_1,
                "lesion_count_run-02": n_2,
                VOL_1: vol_1,
                VOL_2: vol_2,
                "lesion_volume_abs_diff_mm3": abs(vol_2 - vol_1),
                DIFF: abs(vol_2 - vol_1) / ((vol_1 + vol_2) / 2) * 100,
                "lesion_dice": dice(group_1, group_2),
                "lesion_iou": iou,
            })
            continue
        # Group not matched: its lesions are reported as detected only on run-01 and only on run-02
        if n_1:
            rows.append({**base, "status": "only run-01", "lesion_count_run-01": n_1, "lesion_count_run-02": 0, VOL_1: vol_1, VOL_2: 0.0})
        if n_2:
            rows.append({**base, "status": "only run-02", "lesion_count_run-01": 0, "lesion_count_run-02": n_2, VOL_1: 0.0, VOL_2: vol_2})
    return rows


def plot_matched(df, output):
    """Plots the volumes of both runs, the Bland-Altman plots (mm³ and %) and the Dice vs mean lesion volume of
    the matched lesions."""
    hue = "Focal lesion"
    mean_volume = (df[VOL_1] + df[VOL_2]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    sns.scatterplot(data=df, x=VOL_1, y=VOL_2, hue=hue, palette=PALETTE, ax=axes[0])
    max_vol = df[[VOL_1, VOL_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set(title="Volume of the matched lesions", xlabel="Lesion volume run-01 (mm³)", ylabel="Lesion volume run-02 (mm³)")

    bland_altman(axes[1], mean_volume, df["lesion_volume_abs_diff_mm3"], df[hue], "mm³")
    bland_altman(axes[2], mean_volume, df[DIFF], df[hue], "%")

    sns.scatterplot(x=mean_volume, y=df["lesion_dice"], hue=df[hue], palette=PALETTE, ax=axes[3])
    axes[3].set(title="Lesion Dice (run-02 registered to run-01) vs lesion volume", xlabel="Mean lesion volume (mm³)", ylabel="Dice")
    sns.move_legend(axes[3], "lower right", title=hue)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_unmatched(df, output):
    """Plots the number of lesion groups and the volume of the lesions detected only on one run (matched lesions as reference)."""
    hue = "Focal lesion"
    df = df.assign(lesion_volume_mm3=df[[VOL_1, VOL_2]].max(axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(data=df, x="status", hue=hue, palette=PALETTE, order=STATUS, ax=axes[0])
    axes[0].set(title="Number of lesion groups", xlabel="", ylabel="Number of lesion groups")

    sns.stripplot(data=df, x="status", y="lesion_volume_mm3", hue=hue, palette=PALETTE, order=STATUS, dodge=True, ax=axes[1])
    axes[1].set(title="Lesion volume (largest of both runs for matched lesions)", xlabel="", ylabel="Lesion volume (mm³)")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Lower vs upper chunks

CHUNKS = ["lower", "upper"]
CHUNK_PALETTE = {"lower": "tab:green", "upper": "tab:purple"}


def total_volume_row(pair, seg_path_1, seg_path_2):
    """Total lesion volume and number of lesions of both runs (each in its native space) and their difference."""
    (seg_1, voxel_vol_1), (seg_2, voxel_vol_2) = load_seg(seg_path_1), load_seg(seg_path_2)
    vol_1, vol_2 = seg_1.sum() * voxel_vol_1, seg_2.sum() * voxel_vol_2
    return {
        **{k: pair[k] for k in ("subject", "session", "acquisition", "has_lesion")},
        "lesion_count_run-01": lesion_count(seg_1),
        "lesion_count_run-02": lesion_count(seg_2),
        VOL_1: vol_1,
        VOL_2: vol_2,
        **volume_diff(vol_1, vol_2, "lesion_volume"),
    }


def plot_lower_vs_upper(df, output):
    """Plots the total lesion volumes of both runs and their absolute difference (mm³ and %) for the lower and upper
    chunks, and the % difference of the lower and upper chunks of the same subject/session."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    sns.scatterplot(data=df, x=VOL_1, y=VOL_2, hue="acquisition", hue_order=CHUNKS, palette=CHUNK_PALETTE, ax=axes[0])
    max_vol = df[[VOL_1, VOL_2]].max().max()
    axes[0].plot([0, max_vol], [0, max_vol], "k--", linewidth=1)
    axes[0].set(title="Total lesion volume", xlabel="Lesion volume run-01 (mm³)", ylabel="Lesion volume run-02 (mm³)")

    for ax, unit in ((axes[1], "mm3"), (axes[2], "percent")):
        y, label = f"lesion_volume_abs_diff_{unit}", {"mm3": "mm³", "percent": "%"}[unit]
        sns.boxplot(data=df, x="acquisition", y=y, hue="acquisition", order=CHUNKS, palette=CHUNK_PALETTE, legend=False, ax=ax, showfliers=False)
        sns.stripplot(data=df, x="acquisition", y=y, order=CHUNKS, ax=ax, color="k")
        ax.set(title=f"Total lesion volume difference ({label})", xlabel="Chunk", ylabel=f"Absolute lesion volume difference |run-02 - run-01| ({label})")

    # Lower vs upper chunk of the same subject/session
    paired = df.pivot_table(index=["subject", "session"], columns="acquisition", values=DIFF).reindex(columns=CHUNKS).dropna()
    for _, row in paired.iterrows():
        axes[3].plot(CHUNKS, row[CHUNKS], "o-", color="gray", alpha=0.6)
    axes[3].set(title=f"Paired lower vs upper chunks (n = {len(paired)})", xlabel="Chunk", ylabel="Absolute lesion volume difference |run-02 - run-01| (%)")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_lower_vs_upper_results(rows, output, logger):
    """Saves lower_vs_upper.csv, logs the variability of each chunk and the paired Wilcoxon test (lower vs upper chunks
    of the same subject/session), and plots all pairs and only the pairs with visible focal lesions."""
    from scipy.stats import wilcoxon

    df = pd.DataFrame(rows)
    df["Focal lesion"] = df["has_lesion"].map(focal_lesion)
    df.to_csv(output / "lower_vs_upper.csv", index=False)

    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        metrics = ["lesion_volume_abs_diff_mm3", DIFF]
        logger.info(f"Pairs{suffix or ' (all)'}: total lesion volume variability per chunk:\n"
                    f"{df_plot.groupby('acquisition')[metrics].describe().T.round(2)}")
        for metric in metrics:
            paired = df_plot.pivot_table(index=["subject", "session"], columns="acquisition", values=metric).reindex(columns=CHUNKS).dropna()
            try:
                p = wilcoxon(paired["lower"], paired["upper"]).pvalue
            except ValueError as e:  # e.g. no pair, or all differences equal to 0
                p = f"not computed ({e})"
            logger.info(f"Pairs{suffix or ' (all)'}: {metric}, paired Wilcoxon lower vs upper (n = {len(paired)}): p = {p}")
        plot_lower_vs_upper(df_plot, output / f"lower_vs_upper{suffix}.png")
    logger.info(f"Results of {len(df)} pairs saved to {output}")


def save_lesion_wise_results(rows, output, logger):
    """Saves lesions.csv, logs the number of lesion groups and of lesions, and plots the matched and unmatched lesions
    (all labeled lesions, and only the visible focal lesions)."""
    df = pd.DataFrame(rows)
    df["Focal lesion"] = df["has_lesion"].map(focal_lesion)
    df.to_csv(output / "lesions.csv", index=False)
    # Number of lesion groups, and number of lesions (connected components) of each run in these groups
    summary = df.groupby(["Focal lesion", "status"]).agg(**{
        "lesion_groups": ("status", "size"),
        "lesions_run-01": ("lesion_count_run-01", "sum"),
        "lesions_run-02": ("lesion_count_run-02", "sum"),
    })
    logger.info(f"Number of lesion groups and of lesions of each run:\n{summary}")
    logger.info(GROUPS_EXPLANATION)

    df = df.dropna(subset=["Focal lesion"])
    for suffix, df_plot in (("", df), ("_visible", df[df["Focal lesion"] == "visible"])):
        plot_matched(df_plot[df_plot["status"] == "matched"], output / f"matched_lesions{suffix}.png")
        plot_unmatched(df_plot, output / f"unmatched_lesions{suffix}.png")
    n_groups = len(df[["subject", "session", "acquisition", "lesion_group"]].drop_duplicates())
    logger.info(f"Results of {n_groups} lesion groups saved to {output}")
