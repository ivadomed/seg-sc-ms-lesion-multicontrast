"""
Crop every case of an nnU-Net dataset around the spinal canal or spinal cord.

The structure is detected with YOLO (via sc_crop.py) on channel _0000, and the
same physical bounding box is then applied to every channel and to the label.

Arguments:
    -i, --input: Source nnU-Net dataset directory
    -o, --output: Output directory for cropped dataset
    -m, --model: Path to YOLO weights (best.pt)
    -s, --structure: Structure to crop around (canal or sc)
    --rgb: Pseudo-RGB slices (if model was trained on RGB)
    --conf: Confidence threshold(s) for YOLO predictions
    --margin: Margin added in all three directions (mm)
    --sc-crop-dir: Directory containing sc_crop.py

Author: Pierre-Louis Benveniste
"""

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# nnU-Net image files are named <case_id>_<4-digit channel>.nii.gz; labels are <case_id>.nii.gz
CHANNEL_RE = re.compile(r"^(?P<case>.+)_\d{4}\.nii\.gz$")
SPLITS = [("imagesTr", "labelsTr"), ("imagesTs", "labelsTs")]


def parse_args():
    p = argparse.ArgumentParser(description="Crop an nnU-Net dataset using sc_crop.py")
    p.add_argument("-i", "--input", required=True, type=Path, help="Source nnU-Net dataset directory")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output directory for cropped dataset")
    p.add_argument("-m", "--model", required=True, type=Path, help="Path to YOLO weights (best.pt)")
    p.add_argument("-s", "--structure", choices=["canal", "sc"], default="canal", help="Structure to crop around")
    p.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (if model was trained on RGB)")
    p.add_argument("--conf", type=float, nargs="+", default=[0.25], help="Confidence threshold(s)")
    p.add_argument("--margin", type=float, nargs="+", default=[0], help="Margin added in all three directions (mm)")
    p.add_argument("--sc-crop-dir", required=True, type=Path, help="Directory containing sc_crop.py")
    return p.parse_args()


def find_cases(images_dir):
    """Return {case_id: [channel paths, sorted so that _0000 comes first]}."""
    cases = defaultdict(list)
    for path in sorted(images_dir.glob("*.nii.gz")):
        match = CHANNEL_RE.match(path.name)
        if match is None:
            tqdm.write(f"  Skipping {path.name}: no _XXXX channel suffix")
            continue
        cases[match["case"]].append(path)
    return cases


def crop_case(sc_crop, model, args, image_paths, label_path, out_images, out_labels):
    """Detect the structure on channel _0000, then crop all channels and the label with that box."""
    ref = sc_crop.resample_image(image_paths[0])
    slices = sc_crop.cut_slices(np.asarray(ref.dataobj), args.rgb)
    preds = sc_crop.predict_slices(model, slices, sc_crop.get_thresholds(args.conf, model.names))
    box = sc_crop.find_box(preds, sc_crop.STRUCTURES[args.structure], ref.shape, args.margin)
    if box is None:
        raise ValueError(f"no {args.structure} detected on {image_paths[0].name}")

    targets = [(p, out_images) for p in image_paths]
    if label_path is not None:
        targets.append((label_path, out_labels))

    # Compute every crop before writing anything, so a failing case leaves no partial output
    crops = []
    for path, out_dir in targets:
        img = nib.load(path)
        crop = sc_crop.to_image_voxels(*box, ref.affine, img)
        if any(s.stop <= s.start for s in crop):
            raise ValueError(f"crop box falls outside {path.name}")
        crops.append((img.slicer[crop], out_dir / path.name))

    for cropped, out_path in crops:
        nib.save(cropped, out_path)


def main():
    args = parse_args()

    sc_crop_dir = args.sc_crop_dir.resolve()
    sys.path.insert(0, str(sc_crop_dir))
    import sc_crop

    src, dst = args.input.resolve(), args.output.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))  # load once, not once per case
    errors = []

    for images_name, labels_name in SPLITS:
        images_dir, labels_dir = src / images_name, src / labels_name
        out_images, out_labels = dst / images_name, dst / labels_name
        out_images.mkdir(exist_ok=True)
        out_labels.mkdir(exist_ok=True)

        cases = find_cases(images_dir)
        for case_id, image_paths in tqdm(sorted(cases.items()), desc=images_name):
            label_path = labels_dir / f"{case_id}.nii.gz"
            try:
                crop_case(sc_crop, model, args, image_paths, label_path, out_images, out_labels)
            except Exception as exc:
                errors.append(f"{case_id}: {exc}")
                tqdm.write(f"  WARNING {errors[-1]}")

    if (src / "dataset.json").exists():
        shutil.copy2(src / "dataset.json", dst / "dataset.json")

    if errors:
        print(f"\n{len(errors)} case(s) with errors:")
        print("\n".join(f"  {e}" for e in errors))
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()