"""
Generates all files needed for the scan-rescan study, for every (non-excluded) scan/rescan
pair listed in the input JSON:

    1. Spinal cord segmentation of each run:
        SCT_USE_GPU=1 sct_deepseg spinalcord -i <image> -o <sc_seg>
    2. Intervertebral disc labels of each run:
        SCT_USE_GPU=1 sct_deepseg spine -i <image> -o <tmp>/out.nii.gz
       (only the out_totalspineseg_discs.nii.gz/.json outputs are kept and renamed)
    3. Disc labels common to both runs:
        sct_label_utils -i <run-01 discs> -remove-sym <run-02 discs> -o <run-01 common> <run-02 common>
    4. Registration of the rescan (run-02) onto the scan (run-01), using the common disc labels
       and the spinal cord segmentations:
        sct_register_multimodal -i <run-02> -d <run-01> -iseg ... -dseg ... -ilabel ... -dlabel ...
            -param step=0,type=label,dof=Tx_Ty_Tz:step=1,type=seg,algo=centermassrot

Files which already exist are not recomputed, so the script can be re-run to resume.

Output structure:
    <output>/
        <subject>/<session>/anat/
            <image>_label-SC_seg.nii.gz                      (both runs)
            <image>_label-discs_dlabel.nii.gz                (both runs)
            <image>_label-discs_desc-common_dlabel.nii.gz    (both runs)
            <run-02 image>_space-run01.nii.gz                (rescan registered to scan)
            <sub>_<ses>_acq-<acq>_from-run02_to-run01_warp.nii.gz
            <sub>_<ses>_acq-<acq>_from-run01_to-run02_warp.nii.gz
        qc/
        study_files.json   (input entries + paths of generated files, relative to <output>)

Arguments:
    -d / --dataset      Path to the BIDS dataset root
    -j / --json         Path to the JSON file listing the scan/rescan pairs
    -o / --output       Path to the output folder

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SC segs, disc labels and warping fields for the scan-rescan study.")
    parser.add_argument("-d", "--dataset", required=True, type=Path, help="Path to the BIDS dataset root")
    parser.add_argument("-j", "--json", required=True, type=Path, help="Path to the JSON file listing the scan/rescan pairs")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Path to the output folder")
    return parser.parse_args()


def run(cmd):
    """Runs a shell command and raises an error if it fails."""
    if os.system(cmd) != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def image_stem(image_path):
    return Path(image_path).name.replace(".nii.gz", "")


def segment_sc(image, out_dir):
    """Segments the spinal cord of an image with sct_deepseg spinalcord."""
    sc_seg = out_dir / f"{image_stem(image)}_label-SC_seg.nii.gz"
    if not sc_seg.exists():
        run(f"SCT_USE_GPU=1 sct_deepseg spinalcord -i {image} -o {sc_seg}")
    return sc_seg


def label_discs(image, out_dir, qc_dir):
    """Labels the intervertebral discs of an image with sct_deepseg spine (TotalSpineSeg)."""
    disc_labels = out_dir / f"{image_stem(image)}_label-discs_dlabel.nii.gz"
    if disc_labels.exists():
        return disc_labels

    # sct_deepseg spine writes several out_totalspineseg_* files next to the output: we run it in a
    # temporary folder so that runs don't overwrite each other, and only keep the disc labels
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        run(f"SCT_USE_GPU=1 sct_deepseg spine -i {image} -o {tmp_dir / 'out.nii.gz'} -qc {qc_dir}")
        shutil.move(tmp_dir / "out_totalspineseg_discs.json", str(disc_labels).replace(".nii.gz", ".json"))
        shutil.move(tmp_dir / "out_totalspineseg_discs.nii.gz", disc_labels)

    return disc_labels


def process_pair(entry, dataset, output, qc_dir):
    """Generates all files for one scan/rescan pair and returns their paths."""
    subject, session, acq = entry["subject"], entry["session"], entry["acquisition"]
    run1_img = dataset / entry["scan_run-01"]
    run2_img = dataset / entry["rescan_run-02"]
    for img in (run1_img, run2_img):
        if not img.exists():
            raise FileNotFoundError(f"Image not found: {img}")

    out_dir = output / subject / session / "anat"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Spinal cord segmentations
    run1_sc_seg = segment_sc(run1_img, out_dir)
    run2_sc_seg = segment_sc(run2_img, out_dir)

    # Disc labels
    run1_disc_seg = label_discs(run1_img, out_dir, qc_dir)
    run2_disc_seg = label_discs(run2_img, out_dir, qc_dir)

    # Disc labels common to both runs (label-based registration requires matching labels)
    run1_disc_seg_common = out_dir / f"{image_stem(run1_img)}_label-discs_desc-common_dlabel.nii.gz"
    run2_disc_seg_common = out_dir / f"{image_stem(run2_img)}_label-discs_desc-common_dlabel.nii.gz"
    if not run1_disc_seg_common.exists() or not run2_disc_seg_common.exists():
        run(f"sct_label_utils -i {run1_disc_seg} -remove-sym {run2_disc_seg} -o {run1_disc_seg_common} {run2_disc_seg_common}")

    # Registration of the rescan (run-02) onto the scan (run-01)
    prefix = f"{subject}_{session}_acq-{acq}"
    warp_path = out_dir / f"{prefix}_from-run02_to-run01_warp.nii.gz"
    warp_inv_path = out_dir / f"{prefix}_from-run01_to-run02_warp.nii.gz"
    run2_img_reg = out_dir / f"{image_stem(run2_img)}_space-run01.nii.gz"
    if not warp_path.exists() or not warp_inv_path.exists() or not run2_img_reg.exists():
        run(
            f"sct_register_multimodal -i {run2_img} -d {run1_img} -iseg {run2_sc_seg} -dseg {run1_sc_seg} "
            f"-ilabel {run2_disc_seg_common} -dlabel {run1_disc_seg_common} -o {run2_img_reg} "
            f"-owarp {warp_path} -owarpinv {warp_inv_path} "
            f"-param step=0,type=label,dof=Tx_Ty_Tz:step=1,type=seg,algo=centermassrot -qc {qc_dir}"
        )

    rel = lambda p: str(p.relative_to(output))
    return {
        "sc_seg_run-01": rel(run1_sc_seg),
        "sc_seg_run-02": rel(run2_sc_seg),
        "disc_labels_run-01": rel(run1_disc_seg),
        "disc_labels_run-02": rel(run2_disc_seg),
        "disc_labels_common_run-01": rel(run1_disc_seg_common),
        "disc_labels_common_run-02": rel(run2_disc_seg_common),
        "rescan_run-02_reg_to_run-01": rel(run2_img_reg),
        "warp_run-02_to_run-01": rel(warp_path),
        "warp_run-01_to_run-02": rel(warp_inv_path),
    }


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    qc_dir = output / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json) as f:
        entries = json.load(f)
    to_process = [e for e in entries if not e.get("excluded", False)]
    print(f"{len(to_process)} scan/rescan pairs to process ({len(entries) - len(to_process)} excluded)")

    results = []
    failed = []
    for entry in tqdm(to_process):
        name = f"{entry['subject']}/{entry['session']}/acq-{entry['acquisition']}"
        try:
            files = process_pair(entry, dataset, output, qc_dir)
            results.append({**entry, **files, "status": "success"})
        except Exception as e:
            print(f"Failed for {name}: {e}")
            failed.append(name)
            results.append({**entry, "status": "failed", "error": str(e)})

    # Save the index of generated files for the follow-up scripts
    with open(output / "study_files.json", "w") as f:
        json.dump({"dataset": str(dataset), "pairs": results}, f, indent=4)

    print(f"Done: {len(to_process) - len(failed)}/{len(to_process)} pairs processed successfully")
    if failed:
        print("Failed pairs:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
