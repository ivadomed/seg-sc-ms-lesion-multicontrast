"""
Lesion-wise comparison of the model trained on full images vs the model trained on SC-cropped
images, to check whether one model over-segments and the other under-segments.

Both models are compared against the ground truth (GT), lesion by lesion:
    - Everything is evaluated on the full GT grid. The cropped prediction (and the crop field of
      view, taken from the cropped GT) are put back onto that grid with nearest-neighbour
      resampling, and the analysis is restricted to the crop field of view for both models.
    - Lesions = 26-connected components larger than -min-vol mm3.
    - A GT lesion is matched to the predicted lesion with the highest IoU, if IoU > -iou.
    - For each matched lesion: predicted vs GT volume, log2(pred/GT) volume ratio (> 0 means
      over-segmentation, < 0 under-segmentation), extra voxels (pred not GT) and missed voxels
      (GT not pred), and the mean signed distance from the predicted boundary to the GT boundary
      (> 0 means the predicted boundary lies outside the GT one).
    - Lesions touching the crop/image border, and lesions merged with another lesion by a model,
      are flagged and excluded from the statistics.

Inputs: four folders with identical file names (e.g. case.nii.gz), all binary masks:
    -gt-full     GT on the full images
    -gt-crop     GT on the cropped images (used to get the crop field of view, and as a QC)
    -pred-full   predictions of the model trained on full images (same space as -gt-full)
    -pred-crop   predictions of the model trained on cropped images (same space as -gt-crop)

Outputs (in -output-folder):
    lesions.csv          one row per GT lesion, columns full_* / crop_* for each model
    subjects.csv         one row per case and model: detection metrics, total volumes, Dice
    stats.json           volume bias of each model and paired comparison (overall and per size)
    plot_volume_bias.png, plot_paired_log_ratio.png, plot_log_ratio_per_size.png

Example:
    python compare_lesion_volumes_crop_vs_full.py -gt-full /path/labelsTs -gt-crop /path/labelsTs_crop \
        -pred-full /path/pred_full -pred-crop /path/pred_crop -output-folder /path/output

Author: Pierre-Louis Benveniste
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.stats import wilcoxon
from tqdm import tqdm

MODELS = ["full", "crop"]
STRUCT = np.ones((3, 3, 3))  # 26-connectivity
SIZE_BINS = [0, 10, 100, np.inf]  # mm3
SIZE_LABELS = ["<10 mm3", "10-100 mm3", ">100 mm3"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-gt-full", required=True, type=Path, help="GT segmentations on full images")
    parser.add_argument("-gt-crop", required=True, type=Path, help="GT segmentations on cropped images")
    parser.add_argument("-pred-full", required=True, type=Path, help="Predictions of the model trained on full images")
    parser.add_argument("-pred-crop", required=True, type=Path, help="Predictions of the model trained on cropped images")
    parser.add_argument("-output-folder", required=True, type=Path, help="Folder to save the results")
    parser.add_argument("-iou", type=float, default=0.1, help="IoU threshold to match lesions (default: 0.1)")
    parser.add_argument("-min-vol", type=float, default=3, help="Minimum lesion volume in mm3 (default: 3)")
    parser.add_argument("-workers", type=int, default=8, help="Number of cases processed in parallel (default: 8)")
    return parser.parse_args()


def on_grid(img, ref):
    """Binary mask of img on the grid of ref (nearest neighbour, 0 outside of img)."""
    if img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3):
        return np.asarray(img.dataobj) > 0.5
    return np.asarray(resample_from_to(img, ref, order=0, cval=0).dataobj) > 0.5


def label_lesions(mask, min_vox):
    """Connected components, removing the ones smaller than min_vox, relabelled 1..n."""
    labels, n = ndimage.label(mask, STRUCT)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.flatnonzero(sizes >= min_vox)
    keep = keep[keep > 0]
    lut = np.zeros(n + 1, dtype=np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels], len(keep)


def signed_boundary_distance(gt, pred, spacing):
    """Mean signed distance (mm) from the boundary of pred to the boundary of gt (> 0: pred outside gt)."""
    # Work in a padded bounding box around both lesions
    coords = np.argwhere(gt | pred)
    lo = np.maximum(coords.min(0) - 3, 0)
    hi = np.minimum(coords.max(0) + 4, gt.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    g, p = gt[sl], pred[sl]
    g_border = g & ~ndimage.binary_erosion(g, STRUCT)
    p_border = p & ~ndimage.binary_erosion(p, STRUCT)
    dist = ndimage.distance_transform_edt(~g_border, sampling=spacing)
    signed = np.where(g, -dist, dist)
    return float(signed[p_border].mean())


def process_case(name, args):
    gt_full = nib.load(args.gt_full / name)
    gt_crop = nib.load(args.gt_crop / name)
    spacing = gt_full.header.get_zooms()[:3]
    vox_vol = float(np.prod(spacing))
    min_vox = int(np.ceil(args.min_vol / vox_vol))

    # Crop field of view on the full grid, and everything restricted to it
    fov_img = nib.Nifti1Image(np.ones(gt_crop.shape[:3], dtype=np.uint8), gt_crop.affine)
    fov = on_grid(fov_img, gt_full)
    fov_border = fov & ~ndimage.binary_erosion(fov, border_value=0)  # crop border + image border
    gt = (np.asarray(gt_full.dataobj) > 0.5) & fov
    preds = {
        "full": on_grid(nib.load(args.pred_full / name), gt_full) & fov,
        "crop": on_grid(nib.load(args.pred_crop / name), gt_full) & fov,
    }

    # QC: cropped GT put back on the full grid should match the full GT
    gt_crop_back = on_grid(gt_crop, gt_full)
    denom = gt_crop_back.sum() + gt.sum()
    qc_dice = 2 * (gt_crop_back & gt).sum() / denom if denom else 1.0

    gt_lab, n_gt = label_lesions(gt, min_vox)
    gt_vox = np.bincount(gt_lab.ravel(), minlength=n_gt + 1)
    at_border = np.bincount(gt_lab[fov_border], minlength=n_gt + 1) > 0

    lesions = [{"case": name, "gt_lesion": i, "gt_vol": gt_vox[i] * vox_vol, "at_border": bool(at_border[i])}
               for i in range(1, n_gt + 1)]
    subjects = []

    for model, pred in preds.items():
        pred_lab, n_pred = label_lesions(pred, min_vox)
        pred_vox = np.bincount(pred_lab.ravel(), minlength=n_pred + 1)

        # Overlap and IoU between every GT lesion and every predicted lesion
        inter = np.bincount((gt_lab * (n_pred + 1) + pred_lab).ravel(),
                            minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)
        inter[0, :] = inter[:, 0] = 0
        union = gt_vox[:, None] + pred_vox[None, :] - inter
        iou = np.divide(inter, union, out=np.zeros(inter.shape), where=union > 0)
        n_gt_per_pred = (inter > 0).sum(0)  # > 1: the predicted lesion merges several GT lesions

        for i in range(1, n_gt + 1):
            row = lesions[i - 1]
            j = int(iou[i].argmax())
            matched = n_pred > 0 and iou[i, j] > args.iou
            row[f"{model}_matched"] = matched
            row[f"{model}_split"] = bool((inter[i, 1:] > 0).sum() > 1)
            if not matched:
                row[f"{model}_merge"] = False
                continue
            row[f"{model}_merge"] = bool(n_gt_per_pred[j] > 1)
            row[f"{model}_iou"] = iou[i, j]
            row[f"{model}_pred_vol"] = pred_vox[j] * vox_vol
            row[f"{model}_log2_ratio"] = np.log2(pred_vox[j] / gt_vox[i])
            row[f"{model}_extra_vol"] = (pred_vox[j] - inter[i, j]) * vox_vol
            row[f"{model}_missed_vol"] = (gt_vox[i] - inter[i, j]) * vox_vol
            row[f"{model}_signed_dist"] = signed_boundary_distance(gt_lab == i, pred_lab == j, spacing)

        tp = int(sum(r[f"{model}_matched"] for r in lesions))
        fp = int(sum(iou[1:, j].max(initial=0) <= args.iou for j in range(1, n_pred + 1)))
        gt_bin, pred_bin = gt_lab > 0, pred_lab > 0
        denom = gt_bin.sum() + pred_bin.sum()
        subjects.append({
            "case": name, "model": model, "n_gt": n_gt, "n_pred": n_pred, "tp": tp, "fn": n_gt - tp, "fp": fp,
            "n_split": int(sum(r[f"{model}_split"] for r in lesions)),
            "n_merge": int((n_gt_per_pred[1:] > 1).sum()),
            "gt_total_vol": gt_bin.sum() * vox_vol, "pred_total_vol": pred_bin.sum() * vox_vol,
            "dice": 2 * (gt_bin & pred_bin).sum() / denom if denom else 1.0,
            "qc_gt_crop_vs_full_dice": qc_dice,
        })
    return lesions, subjects


def median_iqr(x):
    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None,
            "q1": float(np.percentile(x, 25)) if len(x) else None,
            "q3": float(np.percentile(x, 75)) if len(x) else None}


def wilcoxon_p(x):
    x = np.asarray(x)
    return float(wilcoxon(x).pvalue) if len(x) > 0 and np.any(x != 0) else None


def compute_stats(df):
    """Volume bias of each model vs GT, and paired comparison on lesions found by both models."""
    stats = {}
    for subset_name, subset in [("all", df)] + [(lab, df[df.size_bin == lab]) for lab in SIZE_LABELS]:
        s = {}
        for model in MODELS:
            valid = subset[subset[f"{model}_matched"] & ~subset[f"{model}_merge"] & ~subset.at_border]
            s[model] = {
                "log2_ratio": {**median_iqr(valid[f"{model}_log2_ratio"]),
                               "wilcoxon_p_vs_0": wilcoxon_p(valid[f"{model}_log2_ratio"])},
                "signed_dist_mm": {**median_iqr(valid[f"{model}_signed_dist"]),
                                   "wilcoxon_p_vs_0": wilcoxon_p(valid[f"{model}_signed_dist"])},
            }
        paired = subset[subset.paired]
        s["paired_full_vs_crop"] = {
            "n": int(len(paired)),
            "log2_ratio_diff_full_minus_crop": {
                **median_iqr(paired.full_log2_ratio - paired.crop_log2_ratio),
                "wilcoxon_p": wilcoxon_p(paired.full_log2_ratio - paired.crop_log2_ratio)},
            "signed_dist_diff_full_minus_crop": {
                **median_iqr(paired.full_signed_dist - paired.crop_signed_dist),
                "wilcoxon_p": wilcoxon_p(paired.full_signed_dist - paired.crop_signed_dist)},
        }
        stats[subset_name] = s
    return stats


def make_plots(df, out):
    colors = {"full": "tab:blue", "crop": "tab:orange"}

    # 1. Volume bias vs lesion size, for each model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        valid = df[df[f"{model}_matched"] & ~df[f"{model}_merge"] & ~df.at_border]
        ax.scatter(valid.gt_vol, valid[f"{model}_log2_ratio"], s=8, alpha=0.4, color=colors[model])
        ax.axhline(0, color="k", lw=1)
        ax.axhline(valid[f"{model}_log2_ratio"].median(), color=colors[model], ls="--",
                   label=f"median = {valid[f'{model}_log2_ratio'].median():.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("GT lesion volume (mm3)")
        ax.set_title(f"Model trained on {model} images (n={len(valid)})")
        ax.legend()
    axes[0].set_ylabel("log2(predicted / GT volume)\n> 0 over-segmentation, < 0 under-segmentation")
    fig.tight_layout()
    fig.savefig(out / "plot_volume_bias.png", dpi=200)
    plt.close(fig)

    # 2. Paired log ratio: full vs crop, on the same GT lesions
    paired = df[df.paired]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(paired.full_log2_ratio, paired.crop_log2_ratio, s=8, alpha=0.4, color="tab:gray")
    lim = np.nanmax(np.abs(paired[["full_log2_ratio", "crop_log2_ratio"]].values)) if len(paired) else 1
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=1, ls=":")
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("log2(pred / GT volume), full-image model")
    ax.set_ylabel("log2(pred / GT volume), cropped model")
    ax.set_title(f"Lesions found by both models (n={len(paired)})")
    fig.tight_layout()
    fig.savefig(out / "plot_paired_log_ratio.png", dpi=200)
    plt.close(fig)

    # 3. Log ratio per lesion size, both models side by side (paired lesions)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, model in enumerate(MODELS):
        data = [paired[paired.size_bin == lab][f"{model}_log2_ratio"] for lab in SIZE_LABELS]
        ax.boxplot(data, positions=np.arange(len(SIZE_LABELS)) * 3 + k, widths=0.8, patch_artist=True,
                   boxprops={"facecolor": colors[model], "alpha": 0.6}, showfliers=False)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(np.arange(len(SIZE_LABELS)) * 3 + 0.5)
    ax.set_xticklabels([f"{lab}\n(n={(paired.size_bin == lab).sum()})" for lab in SIZE_LABELS])
    ax.set_ylabel("log2(predicted / GT volume)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=colors[m], alpha=0.6) for m in MODELS],
              [f"{m} model" for m in MODELS])
    fig.tight_layout()
    fig.savefig(out / "plot_log_ratio_per_size.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in args.gt_full.glob("*.nii.gz"))
    missing = [n for n in names if not all((d / n).exists() for d in [args.gt_crop, args.pred_full, args.pred_crop])]
    if missing:
        print(f"WARNING: {len(missing)} case(s) missing in at least one folder, skipped: {missing[:5]}...")
    names = [n for n in names if n not in missing]

    lesions, subjects = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for les, sub in tqdm(ex.map(process_case, names, [args] * len(names)), total=len(names)):
            lesions += les
            subjects += sub

    df = pd.DataFrame(lesions)
    df["size_bin"] = pd.cut(df.gt_vol, SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    df["paired"] = (df.full_matched & df.crop_matched & ~df.full_merge & ~df.crop_merge & ~df.at_border)
    df.to_csv(args.output_folder / "lesions.csv", index=False)
    df_sub = pd.DataFrame(subjects)
    df_sub.to_csv(args.output_folder / "subjects.csv", index=False)

    stats = compute_stats(df)
    with open(args.output_folder / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    make_plots(df, args.output_folder)

    # Summary
    bad_qc = df_sub[(df_sub.model == "full") & (df_sub.qc_gt_crop_vs_full_dice < 0.9)]
    print(f"\n{len(names)} cases, {len(df)} GT lesions ({df.at_border.sum()} at the crop border), "
          f"{df.paired.sum()} found by both models and used for the paired comparison.")
    if len(bad_qc):
        print(f"WARNING: {len(bad_qc)} case(s) where the cropped GT does not match the full GT (Dice < 0.9), "
              f"check the crop alignment (see qc_gt_crop_vs_full_dice in subjects.csv).")
    print("\nDetection (sum over cases):")
    print(df_sub.groupby("model")[["n_gt", "tp", "fn", "fp", "n_split", "n_merge", "gt_total_vol", "pred_total_vol"]]
          .sum().round(0).to_string())
    print("\nVolume bias, log2(pred / GT) on matched lesions (> 0 over-, < 0 under-segmentation):")
    for model in MODELS:
        s = stats["all"][model]["log2_ratio"]
        print(f"  {model:5s}: median {s['median']:.3f} [IQR {s['q1']:.3f}, {s['q3']:.3f}], n={s['n']}, "
              f"Wilcoxon vs 0 p={s['wilcoxon_p_vs_0']}")
    p = stats["all"]["paired_full_vs_crop"]["log2_ratio_diff_full_minus_crop"]
    print(f"  paired full - crop: median {p['median']:.3f}, n={p['n']}, Wilcoxon p={p['wilcoxon_p']}")
    print(f"\nResults saved in {args.output_folder}")


if __name__ == "__main__":
    main()
