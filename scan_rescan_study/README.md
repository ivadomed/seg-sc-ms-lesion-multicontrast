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
 - `generate_study_files.py`: for each scan/rescan pair, computes the spinal cord segmentations, the disc labels and the warping fields between run-01 and run-02: those files will be used by other scripts
 
