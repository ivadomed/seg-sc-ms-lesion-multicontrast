"""
Takes an nnU-Net format dataset and produces a duplicate with SC-cropped images and labels.

Detection is run on the _0000 channel of each case; the same bounding box is applied to all
channels of that case and to the corresponding label.  After cropping, every label is validated
with check_label_crop and a CSV + JSON quality-control report is written alongside the output.

Usage:
    python crop_nnunet_dataset.py -i /path/to/DatasetXXX_Name -o /path/to/output [options]

Arguments:
    -i / --input        Path to the source nnU-Net dataset directory
    -o / --output       Path for the cropped output dataset (created if absent)
    --pad-sup           Superior padding in mm              (default: 40)
    --pad-inf           Inferior padding in mm              (default: 100)
    --pad-lr            Left-Right padding in mm            (default: 20)
    --pad-ap            Anterior-Posterior padding in mm    (default: 20)
    --device            PyTorch device for SC detection, e.g. "cuda", "cuda:0", "mps"
                        (default: CPU via ONNX Runtime)

Example:
    python crop_nnunet_dataset.py \\
        -i /data/Dataset101_msLesionAgnostic \\
        -o /data/Dataset101_msLesionAgnostic_CropSC
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import tqdm

from sc_crop import CropReport, check_label_crop, crop, detect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_cases(split_dir: Path):
    """Return a dict  case_id -> sorted list of channel paths  for a split folder."""
    if not split_dir.exists():
        return {}
    cases = defaultdict(list)
    for p in sorted(split_dir.glob("*.nii.gz")):
        # strip _XXXX channel suffix → base id
        stem = p.name.replace(".nii.gz", "")
        if "_0000" in stem:
            case_id = stem.rsplit("_", 1)[0]
        else:
            case_id = stem
        cases[case_id].append(p)
    return dict(cases)


def _process_case(
    case_id: str,
    image_paths: list,
    label_path: Path | None,
    out_images_dir: Path,
    out_labels_dir: Path | None,
    pad: dict,
    device: str | None = None,
):
    """
    Detect SC on the first channel, crop all channels and the label, run QC.
    Returns (case_id, qc_dict | None, error_message | None).
    """
    try:
        # Detect on the first (or only) channel
        use_onnx = device is None
        bbox = detect(image_paths[0], **pad, use_onnx=use_onnx, device=device)

        # Crop every channel
        for img_path in image_paths:
            cropped = crop(nib.load(img_path), bbox)
            nib.save(cropped, out_images_dir / img_path.name)

        qc = None
        if label_path is not None and out_labels_dir is not None:
            label_nii = nib.load(label_path)
            qc = check_label_crop(label_nii, bbox)

            if not qc["ok"]:
                # Copy original files unchanged
                for img_path in image_paths:
                    shutil.copy2(img_path, out_images_dir / img_path.name)
                shutil.copy2(label_path, out_labels_dir / label_path.name)
                return case_id, qc, None

            cropped_label = crop(label_nii, bbox)
            nib.save(cropped_label, out_labels_dir / label_path.name)

        return case_id, qc, None

    except Exception as exc:
        return case_id, None, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Crop an nnU-Net dataset to the spinal cord.")
    p.add_argument("-i", "--input", required=True, help="Source nnU-Net dataset directory")
    p.add_argument("-o", "--output", required=True, help="Output directory for cropped dataset")
    p.add_argument("--pad-sup",  type=float, default=40,  help="Superior padding mm")
    p.add_argument("--pad-inf",  type=float, default=100, help="Inferior padding mm")
    p.add_argument("--pad-rl", type=float, default=20,  help="Left-Right padding mm")
    p.add_argument("--pad-ap",  type=float, default=20,  help="Anterior-Posterior padding mm")
    p.add_argument("--device", type=str, default=None,
                   help='PyTorch device for SC detection (e.g. "cuda", "cuda:0", "mps"). '
                        'Omit to use CPU via ONNX Runtime.')
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.exists():
        sys.exit(f"Input dataset not found: {src}")

    pad = dict(
        pad_superior=args.pad_sup,
        pad_inferior=args.pad_inf,
        pad_rl=args.pad_rl,
        pad_ap=args.pad_ap,
    )

    # Discover splits present in the source dataset
    splits = [
        ("imagesTr", "labelsTr"),
        ("imagesTs", "labelsTs"),
    ]

    report = CropReport()
    errors = []

    for images_split, labels_split in splits:
        src_images_dir = src / images_split
        src_labels_dir = src / labels_split
        dst_images_dir = dst / images_split
        dst_labels_dir = dst / labels_split

        if not src_images_dir.exists():
            continue

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        if src_labels_dir.exists():
            dst_labels_dir.mkdir(parents=True, exist_ok=True)

        image_cases = _find_cases(src_images_dir)
        label_cases = _find_cases(src_labels_dir) if src_labels_dir.exists() else {}
        print(f"\nProcessing {images_split}: {len(image_cases)} cases")

        tasks = []
        for case_id, image_paths in sorted(image_cases.items()):
            label_paths = label_cases.get(case_id, [])
            label_path = label_paths[0] if label_paths else None
            out_lbl_dir = dst_labels_dir if src_labels_dir.exists() else None
            tasks.append((case_id, image_paths, label_path, dst_images_dir, out_lbl_dir))

        # Run sc detection, cropping and QC
        for case_id, image_paths, label_path, out_img, out_lbl in tqdm.tqdm(tasks, desc=images_split):
            cid, qc, err = _process_case(case_id, image_paths, label_path, out_img, out_lbl, pad, args.device)
            if qc is not None and label_path is not None:
                report.add(label_path, qc)
            if err:
                errors.append(f"{cid}: {err}")
                print(f"  WARNING {cid}: {err}")

    # Copy dataset.json if present
    ds_json = src / "dataset.json"
    if ds_json.exists():
        shutil.copy2(ds_json, dst / "dataset.json")

    # Write QC report
    if len(report) > 0:
        dst.mkdir(parents=True, exist_ok=True)
        report.save(dst / "crop_qc_report.csv")
        report.save_summary(dst / "crop_qc_summary.json")
        print(f"\nQC report: {dst / 'crop_qc_report.csv'}")
        print(f"QC summary: {dst / 'crop_qc_summary.json'}")
        print(f"  Total cases checked : {len(report)}")
        print(f"  Failed (voxels lost): {report.n_failed()}")

    if errors:
        print(f"\n{len(errors)} case(s) with errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
