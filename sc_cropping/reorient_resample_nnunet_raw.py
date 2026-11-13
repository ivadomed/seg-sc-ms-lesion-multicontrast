"""
Reorient to RPI and resample to 1 mm isotropic all images and labels of an nnUNet dataset
(DatasetXXX_Name folder), writing them to a new folder with the same structure.

- Images (imagesTr, imagesTs, imagesExternalTs; one channel: _0000):
  sct_image -setorient + sct_resample (linear), saved as float32.
- Labels (labelsTr, labelsTs, labelsExternalTs; binary MS lesions): brought onto the
  processed image grid with sct_register_multimodal -identity 1 (nearest neighbour), so image and
  label grids match exactly.
- Other files (dataset.json, conversion_dict*.json, ...) are copied as is.

Cases are processed in parallel. Cases whose outputs already exist are skipped, so the script
can be re-run after an interruption.

Author: Pierre-Louis Benveniste
"""

import argparse
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

ORIENTATION = "RPI"
RESOLUTION = "1x1x1"
# (images folder, labels folder), relative to the dataset folder
SPLITS = [
    ("imagesTr", "labelsTr"),
    ("imagesTs", "labelsTs"),
    ("imagesExternalTs", "labelsExternalTs"),
]


def run(cmd):
    # Capture SCT output to keep the progress bar readable; show it only on failure
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{res.stdout}\n{res.stderr}")


def process_case(image_file, label_file, image_out, label_out):
    # Everything is written to a temporary folder and moved at the end, so an interrupted case
    # never leaves a partial output behind (and is not wrongly skipped on re-run)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_image, tmp_label = f"{tmp}/image.nii.gz", f"{tmp}/label.nii.gz"

        # Image: reorient, resample, cast to float32
        run(f"sct_image -i {image_file} -setorient {ORIENTATION} -o {tmp_image} -v 0")
        run(f"sct_resample -i {tmp_image} -mm {RESOLUTION} -x linear -o {tmp_image} -v 0")
        img = nib.load(tmp_image)
        img.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine, img.header), tmp_image)

        # Label: bring onto the processed image grid
        if label_file is not None:
            run(f"sct_register_multimodal -i {label_file} -d {tmp_image} -identity 1 -x nn "
                f"-ofolder {tmp} -o {tmp_label} -v 0")
            shutil.move(tmp_label, label_out)
        shutil.move(tmp_image, image_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input DatasetXXX_Name folder.")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output dataset folder. Such as DatasetYYY_Name.")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of cases processed in parallel (default: 8).")
    args = parser.parse_args()

    in_dir, out_dir = args.input.resolve(), args.output.resolve()

    # Copy non-NIfTI files (dataset.json, conversion_dict*.json, ...)
    for f in in_dir.rglob("*"):
        if f.is_file() and not f.name.endswith(".nii.gz"):
            (out_dir / f.relative_to(in_dir)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out_dir / f.relative_to(in_dir))

    # List the cases to process (skip the ones already done)
    cases, n_done = [], 0
    for images_folder, labels_folder in SPLITS:
        if not (in_dir / images_folder).exists():
            continue
        (out_dir / images_folder).mkdir(parents=True, exist_ok=True)
        for image_file in sorted((in_dir / images_folder).glob("*_0000.nii.gz")):
            label_name = image_file.name.replace("_0000.nii.gz", ".nii.gz")
            label_file = in_dir / labels_folder / label_name
            if label_file.exists():
                (out_dir / labels_folder).mkdir(parents=True, exist_ok=True)
            else:
                label_file = None
            image_out = out_dir / images_folder / image_file.name
            label_out = out_dir / labels_folder / label_name
            if image_out.exists() and (label_file is None or label_out.exists()):
                n_done += 1
                continue
            cases.append((image_file, label_file, image_out, label_out))
    print(f"{len(cases)} cases to process ({n_done} already done).")

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_case, *case): case[0] for case in cases}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                fut.result()
            except Exception as e:
                errors.append((futures[fut], e))

    if errors:
        for image_file, e in errors:
            print(f"\nFAILED {image_file}:\n{e}")
        raise SystemExit(f"{len(errors)} case(s) failed. Re-run the script to retry them.")
    print(f"Done. Output written to {out_dir}")


if __name__ == "__main__":
    main()
