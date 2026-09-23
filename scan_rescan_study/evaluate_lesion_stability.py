"""
Evaluates the scan-rescan stability of predicted lesion volumes between run-01 and
run-02 UNIT1 predictions, for every model subfolder produced by predict_lesion.py.

Two levels of stability can be computed:
    - Whole-scan lesion volume (always computed): computed independently in each run's
      native space, no registration needed. Coefficient of variation (CV) is computed
      per subject/session.
    - Lesion-wise volume (only with --lesion-wise): run-02 is registered onto run-01
      space with sct_register_multimodal, the resulting warp is applied to the run-02
      lesion mask with sct_apply_transfo (nearest-neighbor). Both masks are then split
      into 26-connected components (max connectivity) with scipy.ndimage.label, and
      components are matched greedily by decreasing IoU, keeping only pairs with
      IoU > 0.1. CV is computed per matched lesion pair.

Arguments:
    -i / --pred-root    Path to the predictions output folder (contains one subfolder per model)
    -bids / --bids-root Path to the BIDS dataset root (for the run-01/run-02 UNIT1 images)
    -o / --output       Path to the output folder for results and registration files
    -a / --acquisition  Which acquisition(s) to evaluate: "stitch" (default) or "upper-lower"
                         (evaluates acq-upper and acq-lower separately, each saved in its own subfolder)
    --lesion-wise        Also register run-02 onto run-01 and compute lesion-wise CV
                         (off by default, only whole-scan volume CV is computed)

Author: Pierre-Louis Benveniste
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


IOU_THRESHOLD = 0.1
EXCLUDE_DIRS = {"sc_seg", "qc", "disc_seg"}
# Corrupted acquisitions to skip during evaluation
EXCLUDE_CASES = {"sub-003_ses-M0_acq-lower"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate scan-rescan stability of predicted lesion volumes (run-01 vs run-02).")
    parser.add_argument("-i", "--pred-root", required=True, help="Path to the predictions output folder (one subfolder per model)")
    parser.add_argument("-bids", "--bids-root", required=True, help="Path to the BIDS dataset root")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder for results and registration files")
    parser.add_argument("-a", "--acquisition", choices=["stitch", "upper-lower"], default="stitch",
                         help='Evaluate acq-stitch (default) or acq-upper and acq-lower separately ("upper-lower")')
    parser.add_argument("--lesion-wise", action="store_true",
                         help="Also register run-02 onto run-01 and compute lesion-wise CV (off by default)")
    return parser.parse_args()


def load_mask_and_voxel_volume(path: Path):
    img = nib.load(str(path))
    mask = img.get_fdata() > 0
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))
    return mask, voxel_volume


def lesion_volume_mm3(path: Path) -> float:
    mask, voxel_volume = load_mask_and_voxel_volume(path)
    return float(mask.sum()) * voxel_volume


def coefficient_of_variation(vol1: float, vol2: float) -> float:
    mean_vol = np.mean([vol1, vol2])
    if mean_vol == 0:
        return 0.0
    return float(np.std([vol1, vol2], ddof=1) / mean_vol * 100)


def compute_iou_matrix(labeled1, n1, labeled2, n2):
    """IoU matrix (1-indexed) between the connected components of two labeled volumes."""
    l1 = labeled1.ravel().astype(np.int64)
    l2 = labeled2.ravel().astype(np.int64)
    combined = l1 * (n2 + 1) + l2
    intersections = np.bincount(combined, minlength=(n1 + 1) * (n2 + 1)).reshape(n1 + 1, n2 + 1)
    voxels1 = np.bincount(l1, minlength=n1 + 1)
    voxels2 = np.bincount(l2, minlength=n2 + 1)

    iou = np.zeros((n1 + 1, n2 + 1))
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            inter = intersections[i, j]
            if inter == 0:
                continue
            union = voxels1[i] + voxels2[j] - inter
            iou[i, j] = inter / union
    return iou, voxels1, voxels2


def match_lesions(iou_matrix, threshold):
    """Greedily match connected components by decreasing IoU, one-to-one, above threshold."""
    n1, n2 = iou_matrix.shape[0] - 1, iou_matrix.shape[1] - 1
    candidates = [(iou_matrix[i, j], i, j) for i in range(1, n1 + 1) for j in range(1, n2 + 1) if iou_matrix[i, j] > threshold]
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched1, matched2 = set(), set()
    matches = []
    for iou, i, j in candidates:
        if i in matched1 or j in matched2:
            continue
        matches.append((i, j, iou))
        matched1.add(i)
        matched2.add(j)
    return matches, matched1, matched2


def evaluate_model(model_dir: Path, bids_root: Path, output_dir: Path, acquisition: str, qc_dir: Path, lesion_wise: bool):
    reg_dir = output_dir / "registration"
    reg_dir.mkdir(parents=True, exist_ok=True)

    run1_segs = sorted(model_dir.rglob(f"*_acq-{acquisition}_run-01_UNIT1_label-lesion_seg.nii.gz"))
    print(f"  Found {len(run1_segs)} run-01 lesion segmentation(s)")

    volume_rows = []
    lesion_rows = []

    for run1_seg in tqdm(run1_segs, desc=f"{model_dir.name}/{acquisition}"):
        run2_seg = Path(str(run1_seg).replace("run-01", "run-02"))
        if not run2_seg.exists():
            print(f"  WARNING no matching run-02 segmentation for {run1_seg.name}, skipping")
            continue

        relative_path = run1_seg.relative_to(model_dir)
        subject, session = relative_path.parts[0], relative_path.parts[1]

        case_key = f"{subject}_{session}_acq-{acquisition}"
        if case_key in EXCLUDE_CASES:
            print(f"  Skipping {case_key} (excluded, corrupted file)")
            continue

        run1_img = bids_root / relative_path.parent / run1_seg.name.replace("_label-lesion_seg.nii.gz", ".nii.gz")
        run2_img = bids_root / relative_path.parent / run2_seg.name.replace("_label-lesion_seg.nii.gz", ".nii.gz")
        if not run1_img.exists() or not run2_img.exists():
            print(f"  WARNING missing source UNIT1 image for {subject}/{session}, skipping")
            continue

        # --- Whole-lesion volume stability (native space) ---
        vol1 = lesion_volume_mm3(run1_seg)
        vol2 = lesion_volume_mm3(run2_seg)
        volume_rows.append({
            "subject": subject,
            "session": session,
            "model": model_dir.name,
            "volume_run1_mm3": vol1,
            "volume_run2_mm3": vol2,
            "mean_volume_mm3": np.mean([vol1, vol2]),
            "abs_vol_diff_mm3": np.abs(vol2 - vol1),
            "cv_percent": coefficient_of_variation(vol1, vol2),
        })

        if not lesion_wise:
            continue

        # --- Register run-02 onto run-01 space ---
        case_id = relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "")
        case_reg_dir = reg_dir / relative_path.parent
        case_reg_dir.mkdir(parents=True, exist_ok=True)

        run2_img_reg = case_reg_dir / f"{case_id}_run-02_UNIT1_reg.nii.gz"
        warp_path = case_reg_dir / f"{case_id}_warp_run-02_to_run-01.nii.gz"
        run2_seg_reg = case_reg_dir / f"{case_id}_run-02_UNIT1_label-lesion_seg_reg.nii.gz"

        # Find the run-01 and run-02 SC segs for QC, in the sc folder
        run1_sc_seg = model_dir.parent / "sc_seg" / relative_path.parent / relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "_run-01_UNIT1_label-sc_seg.nii.gz")
        run2_sc_seg = model_dir.parent / "sc_seg" / relative_path.parent / relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "_run-02_UNIT1_label-sc_seg.nii.gz")
        if not run1_sc_seg.exists() or not run2_sc_seg.exists():
            print(f"  WARNING missing SC segmentation for {subject}/{session}, skipping registration")
            # Raise en error and stop the script
            raise FileNotFoundError(f"Missing SC segmentation for {subject}/{session}: {run1_sc_seg} or {run2_sc_seg} not found")
        
        # Find the run-01 and run-02 disc segs for QC, in the disc folder
        run1_disc_seg = model_dir.parent / "disc_seg" / relative_path.parent /relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "_run-01_UNIT1_label-disc_seg.nii.gz")
        run2_disc_seg = model_dir.parent / "disc_seg" / relative_path.parent /relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "_run-02_UNIT1_label-disc_seg.nii.gz")
        if not run1_disc_seg.exists() or not run2_disc_seg.exists():
            print(f"  WARNING missing disc segmentation for {subject}/{session}, skipping registration")
            # Raise en error and stop the script
            raise FileNotFoundError(f"Missing disc segmentation for {subject}/{session}: {run1_disc_seg} or {run2_disc_seg} not found")
        
        # We need to have the same amount of the labels in the run-01 and run-02 disc segs
        run1_disc_seg_common = model_dir.parent / "disc_seg" / relative_path.parent /relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "_run-01_UNIT1_label-disc_seg_common.nii.gz")
        run2_disc_seg_common = model_dir.parent / "disc_seg" / relative_path.parent /relative_path.name.replace("_run-01_UNIT1_label-lesion_seg.nii.gz", "_run-02_UNIT1_label-disc_seg_common.nii.gz")
        if not run1_disc_seg_common.exists() or not run2_disc_seg_common.exists():
            # We build the common disc segs
            assert os.system(f"sct_label_utils -i {run1_disc_seg} -remove-sym {run2_disc_seg} -o {run1_disc_seg_common} {run2_disc_seg_common}") == 0, f"Failed to build common disc seg for {subject}/{session} run-01"

        if not warp_path.exists():
            assert os.system(
                f"sct_register_multimodal -i {run2_img} -d {run1_img} -iseg {run2_sc_seg} -dseg {run1_sc_seg} -ilabel {run2_disc_seg_common} -dlabel {run1_disc_seg_common} -o {run2_img_reg} "
                f"-owarp {warp_path} -param step=0,type=label,dof=Tx_Ty_Tz:step=1,type=seg,algo=centermassrot -qc {qc_dir}"
            ) == 0, f"Registration failed for {subject}/{session}"

        assert os.system(
            f"sct_apply_transfo -i {run2_seg} -d {run1_img} -w {warp_path} -x nn -o {run2_seg_reg}"
        ) == 0, f"Warp application failed for {subject}/{session}"

        # --- Lesion-wise coefficient of variation ---
        mask1, voxel_volume = load_mask_and_voxel_volume(run1_seg)
        mask2_reg, _ = load_mask_and_voxel_volume(run2_seg_reg)

        labeled1, n1 = ndimage.label(mask1, structure=np.ones((3, 3, 3)))
        labeled2, n2 = ndimage.label(mask2_reg, structure=np.ones((3, 3, 3)))

        iou_matrix, voxels1, voxels2 = compute_iou_matrix(labeled1, n1, labeled2, n2)
        matches, matched1, matched2 = match_lesions(iou_matrix, IOU_THRESHOLD)

        for i, j, iou in matches:
            lvol1 = voxels1[i] * voxel_volume
            lvol2 = voxels2[j] * voxel_volume
            lesion_rows.append({
                "subject": subject, "session": session, "model": model_dir.name,
                "lesion_id": f"{i}-{j}", "iou": iou, "matched": True,
                "volume_run1_mm3": lvol1, "volume_run2_mm3": lvol2,
                "cv_percent": coefficient_of_variation(lvol1, lvol2),
            })
        for i in set(range(1, n1 + 1)) - matched1:
            lesion_rows.append({
                "subject": subject, "session": session, "model": model_dir.name,
                "lesion_id": f"{i}-none", "iou": 0.0, "matched": False,
                "volume_run1_mm3": voxels1[i] * voxel_volume, "volume_run2_mm3": None, "cv_percent": None,
            })
        for j in set(range(1, n2 + 1)) - matched2:
            lesion_rows.append({
                "subject": subject, "session": session, "model": model_dir.name,
                "lesion_id": f"none-{j}", "iou": 0.0, "matched": False,
                "volume_run1_mm3": None, "volume_run2_mm3": voxels2[j] * voxel_volume, "cv_percent": None,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    volume_df = pd.DataFrame(volume_rows)
    volume_df.to_csv(output_dir / "volume_stability.csv", index=False)

    if len(volume_df) > 0:
        print(f"  Mean whole-lesion volume CV: {volume_df['cv_percent'].mean():.2f}% +- {volume_df['cv_percent'].std():.2f}% ({len(volume_df)} subject/session pairs)")

    if lesion_wise:
        lesion_df = pd.DataFrame(lesion_rows)
        lesion_df.to_csv(output_dir / "lesionwise_stability.csv", index=False)

        if len(lesion_df) > 0:
            matched_cv = lesion_df.loc[lesion_df["matched"], "cv_percent"]
            print(f"  Mean matched-lesion CV: {matched_cv.mean():.2f}% ({len(matched_cv)} matched, "
                  f"{(~lesion_df['matched']).sum()} unmatched lesions)")


def main():
    args = parse_args()
    pred_root = Path(args.pred_root).resolve()
    bids_root = Path(args.bids_root).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    qc_dir = output_root / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in pred_root.iterdir() if d.is_dir() and d.name not in EXCLUDE_DIRS)
    print(f"Found {len(model_dirs)} model(s): {[d.name for d in model_dirs]}")

    acquisitions = ["stitch"] if args.acquisition == "stitch" else ["upper", "lower"]

    for model_dir in model_dirs:
        print(f"\nEvaluating model '{model_dir.name}' ...")
        for acquisition in acquisitions:
            print(f"  Acquisition: acq-{acquisition}")
            model_output_dir = output_root / model_dir.name if acquisitions == ["stitch"] else output_root / model_dir.name / acquisition
            evaluate_model(model_dir, bids_root, model_output_dir, acquisition, qc_dir, args.lesion_wise)

    print("\nDone.")


if __name__ == "__main__":
    main()
