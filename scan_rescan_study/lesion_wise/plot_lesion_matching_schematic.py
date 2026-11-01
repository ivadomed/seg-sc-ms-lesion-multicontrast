"""
Generates a schematic figure explaining how lesions are matched across scan-rescan in
evaluate_manual_lesion_wise_variability.py and evaluate_model_lesion_wise_variability.py. The figure is drawn on
synthetic 2D lesions, with the same grouping and matching as in these scripts.

Arguments:
    -o / --output       Path to the output figure (.png, .svg or .pdf)
    --iou-threshold     Minimum IoU to match two lesions (default: 0.1)

Author: Pierre-Louis Benveniste
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import ndimage

H, W = 200, 80
COLORS = {"run-01": np.array([0.85, 0.15, 0.15]), "run-02": np.array([0.15, 0.35, 0.9]), "both": np.array([0.6, 0.2, 0.75])}
STATUS_COLORS = {"matched": "tab:green", "not matched": "tab:orange"}


def ellipse(cx, cy, rx, ry):
    y, x = np.mgrid[:H, :W]
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def background():
    """Synthetic sagittal view of the spinal cord."""
    y, x = np.mgrid[:H, :W]
    cord = np.abs(x - 40 - 2 * np.sin(y / 30)) <= 15
    noise = ndimage.gaussian_filter(np.random.default_rng(0).normal(0, 0.15, (H, W)), 1.5)
    return np.clip(np.where(cord, 0.55, 0.12) + noise, 0, 1)


def overlay(image, masks_colors):
    """Colors the masks on the grayscale image."""
    rgb = np.repeat(image[..., None], 3, axis=2)
    for mask, color in masks_colors:
        rgb[mask] = 0.15 * rgb[mask] + 0.85 * color
    return rgb


def main():
    parser = argparse.ArgumentParser(description="Generate a schematic figure of the lesion matching across scan-rescan.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output figure")
    parser.add_argument("--iou-threshold", default=0.1, type=float, help="Minimum IoU to match two lesions")
    args = parser.parse_args()

    # Synthetic lesions: same lesion slightly moved, lesion split in two blobs on the rescan, lesion only on the
    # scan, lesion only on the rescan, and two lesions barely overlapping (IoU below the threshold)
    seg_1 = ellipse(40, 30, 7, 9) | ellipse(40, 75, 6, 14) | ellipse(37, 115, 5, 6) | ellipse(40, 182, 8, 9)
    seg_2 = ellipse(41, 33, 7, 8) | ellipse(39, 65, 4.5, 5) | ellipse(41, 86, 4.5, 5) | ellipse(44, 150, 4, 4) | ellipse(47, 175, 3, 3)
    image = background()

    # Grouping and matching (same as in the evaluation scripts)
    groups, n_groups = ndimage.label(seg_1 | seg_2, structure=np.ones((3, 3)))

    fig, axes = plt.subplots(1, 4, figsize=(17, 8), gridspec_kw={"width_ratios": [80, 80, 175, 120]})
    axes[0].imshow(overlay(image, [(seg_1, COLORS["run-01"])]))
    axes[0].set_title("1. Scan (run-01)", fontsize=13)
    axes[1].imshow(overlay(image, [(seg_2, COLORS["run-02"])]))
    axes[1].set_title("2. Rescan (run-02)\nregistered to the scan", fontsize=13)
    axes[2].imshow(overlay(image, [(seg_1 & ~seg_2, COLORS["run-01"]), (seg_2 & ~seg_1, COLORS["run-02"]), (seg_1 & seg_2, COLORS["both"])]))
    axes[2].set_title(f"3. Lesion groups (overlapping lesions of both runs)\nmatched if IoU ≥ {args.iou_threshold}", fontsize=13)

    for g in range(1, n_groups + 1):
        group = groups == g
        group_1, group_2 = seg_1 & group, seg_2 & group
        n_1, n_2 = ndimage.label(group_1, structure=np.ones((3, 3)))[1], ndimage.label(group_2, structure=np.ones((3, 3)))[1]
        iou = (group_1 & group_2).sum() / group.sum()
        if iou >= args.iou_threshold:
            status, text = "matched", f"matched ({n_1} vs {n_2} lesion{'s' if max(n_1, n_2) > 1 else ''})"
        else:
            status = "not matched"
            text = " + ".join(name for name, n in (("only run-01", n_1), ("only run-02", n_2)) if n)
        color = STATUS_COLORS[status]
        axes[2].contour(ndimage.binary_dilation(group, iterations=3).astype(float), levels=[0.5], colors=color, linestyles="--", linewidths=1.5)
        ys, xs = np.nonzero(group)
        axes[2].annotate(f"G{g}: IoU = {iou:.2f}\n→ {text}", xy=(xs.max() + 4, ys.mean()), xytext=(88, ys.mean()), va="center", fontsize=11,
                         color=color, fontweight="bold", arrowprops={"arrowstyle": "->", "color": color})
    axes[2].set_xlim(-0.5, 175)

    for ax in axes[:3]:
        ax.axis("off")
    axes[2].legend(handles=[Patch(color=COLORS["run-01"], label="Lesion on the scan only"), Patch(color=COLORS["run-02"], label="Lesion on the rescan only"),
                            Patch(color=COLORS["both"], label="Overlap"), Line2D([], [], color=STATUS_COLORS["matched"], linestyle="--", label="Matched group"),
                            Line2D([], [], color=STATUS_COLORS["not matched"], linestyle="--", label="Not matched group")],
                   loc="upper center", bbox_to_anchor=(0.3, -0.01), ncol=2, fontsize=10, frameon=False)

    # Matching rules
    axes[3].axis("off")
    axes[3].text(0, 0.95, "How lesions are matched", fontsize=14, fontweight="bold", va="top")
    axes[3].text(0, 0.86, (
        "• The rescan lesions are registered to the scan\n\n"
        "• Lesion group = connected component of the\n"
        "  union of both segmentations: a lesion seen as\n"
        "  one blob on one run and several blobs on the\n"
        "  other run is compared as a whole (G2)\n\n"
        "• IoU = |scan ∩ rescan| / |scan ∪ rescan|\n"
        "  within the group\n\n"
        f"• IoU ≥ {args.iou_threshold} → matched: volume difference\n"
        "  and Dice are computed on the group (G1, G2)\n\n"
        f"• IoU < {args.iou_threshold} → lesions reported as detected\n"
        "  only on the scan and/or only on the rescan\n"
        "  (G3, G4, G5)"
    ), fontsize=12, va="top", linespacing=1.4)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {args.output}")


if __name__ == "__main__":
    main()
