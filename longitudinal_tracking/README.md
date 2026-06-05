## Longitudinal tracking of multiple sclerosis lesions in the spinal cord: A validation study

Here are the steps to reproduce the study:

### Data preparation

1. Build the MSD file from the include.yml file of selected subject for this study (in this case we used this [include.yml](./dataset_aggregation/include.yml) file)

```console
python build_msd_from_include_yml.py --include-yml /path/to/include.yml --dataset-path /path/to/dataset --output-path /path/to/output/folder
```

2. Generate the GT lesion mappings for GT lesion segmentations
        
    To do so, I labeled the lesion segmentations (so that each connected-component has a unique instance value) using  [generate_gt_labeled_segs_with_mapping.py](./dataset_exploration/generate_gt_labeled_segs_with_mapping.py). Then I manually fill the created lesion_mapping.json file witout considering how they were initially filed.

    NB: for the creation of the lesion mapping for predicted lesion segmentations, I just use the initialized file from strategy #5 (reg+IoU) and correct the mappings manually.

3. Generate the segmentation files using SCT with [run_sct_methods.py](./tracking/run_sct_methods.py)

```console
python run_sct_methods.py -i /path/to/msd/dataset -o /output/path
```

4. Generate the csv file for ML lesion tracking with [generate_csv_for_ML.py](./tracking/generate_csv_for_ML.py)

```console
python generate_csv_for_ML.py  -i /path/to/msd/ -pred /path/to/pred/segs/ -gt_mappings /path/to/folder/GT/mappings -o /path/output/folder 
```

### Running lesion tracking

5. Run tracking for non-ML methods using [run_lesion_tracking.py](./tracking/run_lesion_tracking.py)

```console
 python run_lesion_tracking.py -i /path/to/msd -pred /path/to/pred/segs -o /path/output/folder -m registered_with_CoM OR registered_with_IoU OR unregistered
```

6. Run tracking with XGB using [track_lesion_xgb.py](./tracking/track_lesion_xgb.py)

```console
python track_lesion_xgb.py --dataset_csv /path/to/csv/dataset --output_folder /path/output/folder
```

7. Run tracking with a Siamese model using [track_lesion_siamese.py](./tracking/track_lesion_siamese.py)

```console
python track_lesion_siamese.py --dataset_csv /path/to/csv/dataset --output_folder /path/output/folder
```

### Lesion tracking evaluation

8. To evaluate each tracking methods we run [run_evaluation.py](./eval_tracking/run_evaluation.py)
