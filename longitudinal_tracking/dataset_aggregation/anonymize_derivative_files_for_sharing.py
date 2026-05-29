"""
First this script adds the lesion mappings to the derivative files.
Lesion mappings are stored in folder:
    - lesion_mappings/
        sub-cal001/
            lesion_mapping.json
        ...
It copies the lesion mapping file for each subject and adds it to the derivative folder for that subject, so that it can be used in the next step of anonymization.
This script takes as input the folder with all derivative files and removes the site name and assigns a random value to the subject ID.
For example for subject sub-cal072, it would go from this:
    - sub-cal072_ses-M0_STIR_centerline.csv
    - sub-cal072_ses-M0_STIR_sc-seg.json
    - sub-cal072_ses-M12_STIR_inv_warp_to_sub-cal072_ses-M0_STIR.nii.gz
to this:
    - sub-001_ses-M0_STIR_centerline.csv
    - sub-001_ses-M0_STIR_sc-seg.json
    - sub-001_ses-M12_STIR_inv_warp_to_sub-001_ses-M0_STIR.nii.gz
And it would create a conversion dictionary that would look like this:
    {
        "sub-cal072": "sub-001",
        ...
    }

Input:
    -i: path to the folder with all derivative files
    -lesion-mapping: path to the folder containing the lesion mapping files for each subject
    -o: path to the output folder where the new derivative files will be saved

Author: Pierre-Louis Benveniste
"""

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(description="Anonymize derivative files by replacing subject IDs.")
    parser.add_argument("-i", required=True, help="Path to the input folder with all derivative files.")
    parser.add_argument("-lesion-mapping", required=True, help="Path to the folder containing the lesion mapping files for each subject (lesion_mappings/<subject>/lesion_mapping.json).")
    parser.add_argument("-o", required=True, help="Path to the output folder where anonymized files will be saved.")
    return parser


def find_all_subjects(input_dir):
    """Collect all unique subject IDs from filenames in the input directory."""
    subjects = set()
    pattern = re.compile(r"(sub-[a-zA-Z0-9]+)") #Looks for text with sub- followed by letters and/or numbers. The + signs tells it to stop at the first non-alphanumeric character
    for path in Path(input_dir).rglob("*"):
        if path.is_file():
            for match in pattern.findall(path.name):
                subjects.add(match)
    return sorted(subjects)


def build_conversion_dict(subjects):
    """Assign an anonymized ID (sub-001, sub-002, ...) to each subject in random order."""
    shuffled = list(subjects)
    random.shuffle(shuffled)
    return {subj: f"sub-{i+1:03d}" for i, subj in enumerate(shuffled)}


def anonymize_name(name, conversion_dict):
    """Replace all subject IDs in a filename or string using the conversion dict."""
    for original, anonymized in conversion_dict.items():
        name = name.replace(original, anonymized)
    return name


def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = Path(args.i)
    lesion_mapping_dir = Path(args.lesion_mapping)
    output_dir = Path(args.o)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy lesion mapping files into each subject's derivative folder
    print("Step 1: Copying lesion mapping files into derivative folders...")
    lesion_mapping_copied = 0
    for subject_dir in lesion_mapping_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name
        lesion_mapping_file = subject_dir / "lesion_mapping.json"
        if not lesion_mapping_file.exists():
            print(f"  Warning: no lesion_mapping.json found for {subject_id}, skipping.")
            continue
        dest_subject_dir = input_dir / subject_id
        if not dest_subject_dir.exists():
            print(f"  Warning: derivative folder for {subject_id} not found in input dir, skipping.")
            continue
        dest_file = dest_subject_dir / "lesion_mapping.json"
        shutil.copy2(lesion_mapping_file, dest_file)
        lesion_mapping_copied += 1
        print(f"  Copied lesion_mapping.json for {subject_id}")
    print(f"  Done. Copied {lesion_mapping_copied} lesion mapping file(s).\n")

    # Step 2: Discover subjects and build anonymization mapping
    subjects = find_all_subjects(input_dir)
    if not subjects:
        print("No subjects found in the input directory.")
        return

    conversion_dict = build_conversion_dict(subjects)
    print(f"Step 2: Found {len(subjects)} subjects. Conversion mapping:")
    for orig, anon in conversion_dict.items():
        print(f"  {orig} -> {anon}")

    # Save conversion dictionary
    conversion_path = output_dir / "subject_conversion_dict_for_sharing_derivative_files.json"
    with open(conversion_path, "w") as f:
        json.dump(conversion_dict, f, indent=4)
    print(f"\nConversion dictionary saved to: {conversion_path}")

    # Step 3: Copy and rename files with anonymized subject IDs
    print("\nStep 3: Anonymizing and copying derivative files...")
    for src_path in input_dir.rglob("*"):
        if src_path.is_file():
            rel_path = src_path.relative_to(input_dir)
            # Anonymize each part of the relative path
            new_parts = [anonymize_name(part, conversion_dict) for part in rel_path.parts]
            dst_path = output_dir / Path(*new_parts)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)

    print(f"\nAnonymized files saved to: {output_dir}")


if __name__ == "__main__":
    main()
