## Longitudinal tracking of multiple sclerosis lesions in the spinal cord: A validation study

Here are the steps to reproduce the study:

1. Build the MSD file from the include.yml file of selected subject for this study (in this case we used this [include.yml](./dataset_aggregation/include.yml) file)

```console
    python build_msd_from_include_yml.py --include-yml /path/to/include.yml --dataset-path /path/to/dataset --output-path /path/to/output/folder
```

2. Generate the GT lesion mappings for:
    2.1. GT lesion segmentations
    2.2. predicted lesion segmentations
    For 