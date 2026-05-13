"""
This file creates the MSD-style JSON datalist to train a longitudinal model.
Creates pairs of consecutive labeled images of the same contrast for each subject.

Arguments:
    --path_data: Path to the data set directory
    --output: Path to the output directory where dataset json is saved
    --seed: Random seed for reproducibility

Pierre-Louis Benveniste
"""

import os
import json
from tqdm import tqdm
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from datetime import date
from pathlib import Path
from collections import defaultdict
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for longitudinal lesion segmentation')
    parser.add_argument('--data', type=str, required=True, help='Path to the data set directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory where dataset json is saved')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser


def get_session_date(derivative_path: Path) -> str:
    """
    Extracts the session date string (e.g. '20231121') from a BIDS path containing 'ses-YYYYMMDD'.
    Returns the raw string so it can be sorted lexicographically (ISO date format sorts correctly).
    """
    for part in derivative_path.parts:
        if part.startswith('ses-'):
            return part.replace('ses-', '')
    return ''


def get_contrast(derivative_path: Path) -> str:
    """Extracts the contrast identifier from the filename (last underscore-separated token before .nii.gz)."""
    return derivative_path.name.replace('_label-lesion_seg.nii.gz', '.nii.gz').split('_')[-1].replace('.nii.gz', '')


def get_subject(derivative_path: Path) -> str:
    """Extracts the subject ID from the filename."""
    return derivative_path.name.split('_')[0]


def build_longitudinal_pairs(derivatives: list, data_path: str) -> list:
    """
    Groups derivatives by (subject, contrast), sorts sessions chronologically,
    and builds consecutive pairs (session N, session N+1).

    Each pair is a dict with:
        image1, label1  -> earlier timepoint
        image2, label2  -> later timepoint
        subject, contrast, session1, session2, site

    Only pairs where all four files exist on disk are included.

    Input:
        derivatives : list[Path] : all label files found under data_path
        data_path   : str        : root of the dataset (for relative path computation)

    Returns:
        pairs : list[dict]
    """
    # Group by (subject, contrast)
    groups = defaultdict(list)
    for deriv in derivatives:
        subject  = get_subject(deriv)
        contrast = get_contrast(deriv)
        session  = get_session_date(deriv)
        groups[(subject, contrast)].append((session, deriv))

    pairs = []
    for (subject, contrast), entries in groups.items():
        # Sort by session date (lexicographic sort works for YYYYMMDD)
        entries_sorted = sorted(entries, key=lambda x: x[0])

        for i in range(len(entries_sorted) - 1):
            ses1, label1_path = entries_sorted[i]
            ses2, label2_path = entries_sorted[i + 1]

            image1_path = str(label1_path).replace('_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels/', '')
            image2_path = str(label2_path).replace('_label-lesion_seg.nii.gz', '.nii.gz').replace('derivatives/labels/', '')

            # Only keep pairs where all four files exist
            if not all(os.path.exists(p) for p in [str(label1_path), str(label2_path), image1_path, image2_path]):
                missing = [p for p in [str(label1_path), str(label2_path), image1_path, image2_path] if not os.path.exists(p)]
                logger.warning(f"Skipping pair ({subject}, {contrast}, {ses1}->{ses2}): missing files: {missing}")
                continue

            pairs.append({
                "image1":    image1_path,
                "label1":    str(label1_path),
                "image2":    image2_path,
                "label2":    str(label2_path),
                "subject":   subject,
                "contrast":  contrast,
                "session1":  ses1,
                "session2":  ses2,
                "site":      "ms-ucsf-2026",
            })

    return pairs


def split_pairs_by_subject(pairs: list, test_size: float = 0.1, random_state: int = 42):
    """
    Splits pairs into train / val / test by subject (no subject appears in two splits).

    Input:
        pairs        : list[dict] : output of build_longitudinal_pairs()
        test_size    : float      : fraction of subjects held out for test (and for val)
        random_state : int

    Returns:
        train, val, test : list[dict]
    """
    subjects = list({p["subject"] for p in pairs})

    subj_train, subj_test = train_test_split(subjects, test_size=test_size, random_state=random_state)
    subj_train, subj_val  = train_test_split(subj_train, test_size=test_size / (1 - test_size), random_state=random_state)

    subj_train = set(subj_train)
    subj_val   = set(subj_val)
    subj_test  = set(subj_test)

    train = [p for p in pairs if p["subject"] in subj_train]
    val   = [p for p in pairs if p["subject"] in subj_val]
    test  = [p for p in pairs if p["subject"] in subj_test]

    return train, val, test


def print_pairs_distribution(pairs: list, split_name: str):
    """Logs contrast distribution and subject count for a given split."""
    contrasts = [p["contrast"] for p in pairs]
    subjects  = {p["subject"] for p in pairs}
    logger.info(f"[{split_name}] {len(pairs)} pairs | {len(subjects)} subjects")
    for c in sorted(set(contrasts)):
        logger.info(f"  {c}: {contrasts.count(c)} pairs")


def main():
    parser = get_parser()
    args = parser.parse_args()
    data_path   = args.data
    output_path = args.output
    test_size   = 0.1

    # ------------------------------------------------------------------ #
    # 1. Discover all label files
    # ------------------------------------------------------------------ #
    derivatives = list(Path(data_path).rglob('*_label-lesion_seg.nii.gz'))
    logger.info(f"Found {len(derivatives)} label files under {data_path}")

    # ------------------------------------------------------------------ #
    # 2. Build consecutive longitudinal pairs per (subject, contrast)
    # ------------------------------------------------------------------ #
    all_pairs = build_longitudinal_pairs(derivatives, data_path)
    logger.info(f"Built {len(all_pairs)} valid consecutive pairs")

    # ------------------------------------------------------------------ #
    # 3. Train / val / test split (subject-level)
    # ------------------------------------------------------------------ #
    train_pairs, val_pairs, test_pairs = split_pairs_by_subject(
        all_pairs, test_size=test_size, random_state=args.seed
    )

    for split_name, split_pairs in [("train", train_pairs), ("validation", val_pairs), ("test", test_pairs)]:
        print_pairs_distribution(split_pairs, split_name)

    # ------------------------------------------------------------------ #
    # 4. Assemble the JSON
    # ------------------------------------------------------------------ #
    params = {
        "description":        "ms-lesion-longitudinal",
        "labels":             {"0": "background", "1": "ms-lesion-seg"},
        "license":            "plb",
        "modality":           {"0": "MRI"},
        "name":               "ms-lesion-longitudinal",
        "seed":               args.seed,
        "reference":          "NeuroPoly",
        "tensorImageSize":    "3D",
        "task":               "consecutive-pair segmentation",
        "train":              train_pairs,
        "validation":         val_pairs,
        "test":               test_pairs,
        "numTraining":        len(train_pairs),
        "numValidation":      len(val_pairs),
        "numTest":            len(test_pairs),
        "numSubjects":        len({p["subject"] for p in all_pairs}),
    }

    total = params["numTraining"] + params["numValidation"] + params["numTest"]
    logger.info(f"Total pairs in dataset: {total}")
    logger.info(f"Total unique subjects:  {params['numSubjects']}")

    # ------------------------------------------------------------------ #
    # 5. Write outputs
    # ------------------------------------------------------------------ #
    os.makedirs(output_path, exist_ok=True)
    today = str(date.today())

    json_path = os.path.join(output_path, f"dataset_{today}.json")
    with open(json_path, "w") as f:
        f.write(json.dumps(params, indent=4, sort_keys=True))
    logger.info(f"Dataset JSON saved to {json_path}")


if __name__ == "__main__":
    main()