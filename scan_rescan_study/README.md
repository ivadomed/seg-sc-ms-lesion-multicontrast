# Scan-rescan study

This study uses the `data_description.json` with the following structure:

```json
{
    "subject": "sub-001",
    "session": "ses-M0",
    "acquisition": "lower",
    "scan_run-01": "sub-001/ses-M0/anat/sub-001_ses-M0_acq-lower_run-01_UNIT1.nii.gz",
    "rescan_run-02": "sub-001/ses-M0/anat/sub-001_ses-M0_acq-lower_run-02_UNIT1.nii.gz",
    "lesion_seg_run-01": "derivatives/labels/sub-001/ses-M0/anat/sub-001_ses-M0_acq-lower_run-01_UNIT1_label-lesion_seg.nii.gz",
    "lesion_seg_run-02": "derivatives/labels/sub-001/ses-M0/anat/sub-001_ses-M0_acq-lower_run-02_UNIT1_label-lesion_seg.nii.gz",
    "has_lesion": null,
    "excluded": false
}
```

## Scripts
 - `common.py`: functions shared by all the scripts below (study files, SCT commands, inference, Dice from `evaluation/utils.py`, volume differences, plots and lesion-wise matching)
 - `generate_study_files.py`: for each scan/rescan pair, computes the spinal cord segmentations, the disc labels and the warping fields between run-01 and run-02: those files will be used by other scripts
 - `evaluate_sc_seg_variability.py`: for each scan/rescan pair, compares the SC volumes over the common PAM50 slices (run-01 vs run-02, and run-01 vs run-02 registered to run-01) and saves one csv line per pair
 - `evaluate_manual_lesion_seg_variability.py`: for each manually segmented scan/rescan pair, compares the lesion segmentations (lesion volumes and counts, volume difference, Dice after registration) and saves a csv and variability plots (all pairs, and labeled by `has_lesion`)
 - `evaluate_model_lesion_seg_variability.py`: same as above, on the lesion segmentations predicted by `sct_deepseg lesion_ms` for all scan/rescan pairs (plus Dice relative to the manual segmentations), with an inference mode (`default`, `tta`, `single-fold`)
 - `evaluate_model_soft_lesion_seg_variability.py`: same as above, on the soft lesion segmentations predicted by `sct_deepseg lesion_ms -average-per-fold` (soft volumes and soft Dice), with or without test-time augmentation (`soft`, `soft-tta`)
 - `lesion_wise/evaluate_manual_lesion_wise_variability.py` and `lesion_wise/evaluate_model_lesion_wise_variability.py`: lesion-wise scan-rescan variability of the manual and model (default inference) lesion segmentations: each run-01 lesion is matched to a run-02 lesion (registered to run-01) to compare their volumes and Dice, and the lesions detected on only one run are reported in a separate figure
 - `lower_vs_upper/evaluate_manual_lower_vs_upper.py` and `lower_vs_upper/evaluate_model_lower_vs_upper.py`: compare the scan-rescan variability of the total lesion volume (native space, mm³ and %) between the lower and upper chunks, for the manual and model (default inference) lesion segmentations, with a paired Wilcoxon test on the subjects/sessions which have both chunks
 - `evaluate_model_vs_manual.py`: for each manually segmented scan, compares the lesion segmentation predicted by `sct_deepseg lesion_ms` to the manual one (Dice, lesion-wise PPV, F1 score and sensitivity) and saves a csv and a plot
 
