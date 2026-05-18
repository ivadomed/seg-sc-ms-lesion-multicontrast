"""
This script adds WMn data (already registered in the conversion dict) to an existing
MSD-style dataset JSON. It reads image/label pairs from the conversion dict whose
nnUNet destination path contains 'WMn', infers the required metadata fields
(contrast, site, acquisition, resolution, dimension, nb_lesions, total_lesion_volume),
and appends the entries to the appropriate split (train / test) in the dataset JSON.

Arguments:
    --dataset       Path to the existing MSD dataset JSON
    --conv-dict     Path to the conversion dict JSON
    --path-wmn      Root path to the WMn data directory (used to load NIfTI files)
    --output        Path for the updated dataset JSON output

Example:
    python add_wmn_to_msd_dataset.py \\
        --dataset dataset_2025-04-15_seed42.json \\
        --conv-dict conversion_dict.json \\
        --path-wmn /home/plbenveniste/net/ms-lesion-agnostic/202604_train_WMn_model/nnunet_data/WMn_data \\
        --output dataset_2025-04-15_seed42_wmn.json

Pierre-Louis Benveniste
"""

import json
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import skimage.measure
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers (mirrors the logic in 1_create_msd_data.py)
# ---------------------------------------------------------------------------

def count_lesion(label_file):
    """Return (total_lesion_volume_mm3, nb_lesions) for a NIfTI label file."""
    label = nib.load(label_file)
    label_data = label.get_fdata()
    resolution = label.header.get_zooms()
    total_volume = float(np.sum(label_data) * np.prod(resolution))
    _, nb_lesions = skimage.measure.label(label_data, connectivity=2, return_num=True)
    return total_volume, int(nb_lesions)


def get_acquisition_resolution_and_dimension(image_path):
    """Return (acquisition_str, resolution_list, dimension_list) for a NIfTI image."""
    img = nib.load(str(image_path))
    header = img.header
    pixdim = np.array(header.get_zooms()[:3], dtype=float)
    shape = np.array(img.shape[:3], dtype=int)

    acquisition = '3D_sag'  # WMn images are always 3D sagittal acquisitions, so we hardcode this value.

    return acquisition, pixdim.tolist(), shape.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(
        description='Add WMn entries from a conversion dict to an MSD dataset JSON.'
    )
    parser.add_argument('--dataset',   required=True, type=str,
                        help='Path to the existing MSD dataset JSON')
    parser.add_argument('--conv-dict', required=True, type=str,
                        help='Path to the conversion dict JSON')
    parser.add_argument('--output',    required=True, type=str,
                        help='Output path for the updated dataset JSON')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset JSON from {args.dataset}")
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    logger.info(f"Loading conversion dict from {args.conv_dict}")
    with open(args.conv_dict, 'r') as f:
        conv_dict = json.load(f)

    # ------------------------------------------------------------------
    # Extract WMn image→label pairs from the conversion dict
    # The dict alternates: image_src → image_dst, label_src → label_dst
    # We identify image entries by their nnUNet destination containing
    # '_0000.nii.gz' (channel suffix), and labels by lacking it.
    # We also determine train vs test from imagesTr / imagesTs.
    # ------------------------------------------------------------------

    wmn_pairs = {}   # nnunet_id → {'image': src, 'label': src, 'split': 'train'|'test'}

    for src_path, dst_path in conv_dict.items():
        # Only process WMn entries
        if 'WMn' not in src_path:
            continue

        # Determine nnUNet case ID from destination filename
        dst_name = Path(dst_path).stem.replace('.nii', '')  # e.g. msLesionAgnostic_001_0000
        is_image = dst_name.endswith('_0000')
        case_id = dst_name.replace('_0000', '')             # e.g. msLesionAgnostic_001

        if case_id not in wmn_pairs:
            wmn_pairs[case_id] = {'image': None, 'label': None, 'split': None}

        if is_image:
            wmn_pairs[case_id]['image'] = src_path
            wmn_pairs[case_id]['split'] = 'train' if 'imagesTr' in dst_path else 'test'
        else:
            wmn_pairs[case_id]['label'] = src_path

    logger.info(f"Found {len(wmn_pairs)} WMn entries in the conversion dict")

    # ------------------------------------------------------------------
    # Build MSD-style records
    # ------------------------------------------------------------------
    train_entries = []
    test_entries  = []
    skipped       = 0

    for case_id, pair in tqdm(sorted(wmn_pairs.items()), desc="Processing WMn entries"):
        image_path = pair['image']
        label_path = pair['label']
        split      = pair['split']

        if image_path is None or label_path is None:
            logger.warning(f"Incomplete pair for {case_id}, skipping")
            skipped += 1
            continue

        if not os.path.exists(image_path):
            logger.warning(f"Image not found on disk: {image_path}")
            skipped += 1
            continue

        if not os.path.exists(label_path):
            logger.warning(f"Label not found on disk: {label_path}")
            skipped += 1
            continue

        try:
            total_lesion_volume, nb_lesions = count_lesion(label_path)
            acquisition, resolution, dimension = get_acquisition_resolution_and_dimension(image_path)
        except Exception as e:
            logger.warning(f"Failed to read NIfTI for {case_id}: {e}")
            skipped += 1
            continue

        entry = {
            "image":               image_path,
            "label":               label_path,
            "site":                "bordeaux",  # WMn data was acquired at Bordeaux, so we hardcode this value.
            "contrast":            "WMn",       # WMn data is always acquired with the WMn contrast, so we hardcode this value.
            "acquisition":         acquisition,
            "resolution":          resolution,
            "dimension":           dimension,
            "nb_lesions":          nb_lesions,
            "total_lesion_volume": total_lesion_volume,
        }

        if split == 'train':
            train_entries.append(entry)
        else:
            test_entries.append(entry)

    logger.info(f"WMn entries to add — train: {len(train_entries)}, test: {len(test_entries)}, skipped: {skipped}")

    # ------------------------------------------------------------------
    # Inject into dataset
    # ------------------------------------------------------------------
    dataset.setdefault('train', []).extend(train_entries)
    dataset.setdefault('test',  []).extend(test_entries)

    # Update counts
    dataset['numTraining'] = len(dataset['train'])
    dataset['numTest']     = len(dataset['test'])

    # Update contrast counts
    dataset.setdefault('contrasts', {})
    for entry in train_entries + test_entries:
        contrast = entry['contrast']
        dataset['contrasts'][contrast] = dataset['contrasts'].get(contrast, 0) + 1

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4, sort_keys=True)

    logger.info(f"Updated dataset saved to {output_path}")
    logger.info(f"  numTraining : {dataset['numTraining']}")
    logger.info(f"  numTest     : {dataset['numTest']}")
    if 'numValidation' in dataset:
        logger.info(f"  numValidation: {dataset['numValidation']}")


if __name__ == "__main__":
    main()