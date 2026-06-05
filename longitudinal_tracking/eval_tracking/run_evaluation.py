"""
This file runs the lesion mapping evaluation script and plots the results.

Input:
    - input_msd_dataset: Path to the MSD dataset folder.
    - pred_seg: Path to the folder containing predicted segmentations.
    - pred_mapping: Path to the folder containing lesion mapping results.
    - gt_mapping: Path to the folder containing ground truth lesion mapping results.
    - output_folder: Path to the folder where evaluation results will be saved.
    - sub-edss: Path to a csv file containing subject EDSS scores.
    - set: Dataset split to evaluate ('train', 'val', 'test' or None for all subjects)
Output:
    None

Author: Pierre-Louis Benveniste
"""
import json
import os
import matplotlib.pyplot as plt
from loguru import logger
import json
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from scipy import stats
from pathlib import Path
from datetime import date
from tqdm import tqdm
import nibabel as nib


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-msd', type=str, required=True, help='Path to the input MSD dataset')
    parser.add_argument('-p', '--pred-seg', type=str, required=True, help='Path to the folder containing predicted segmentations')
    parser.add_argument('-m', '--pred-mapping', type=str, required=True, help='Path to the folder containing lesion mapping results')
    parser.add_argument('-g', '--gt-mapping', type=str, required=True, help='Path to the folder containing ground truth lesion mapping results')
    parser.add_argument('-o', '--output-folder', type=str, required=True, help='Path to the output folder where evaluation results will be stored')
    parser.add_argument('-s', '--sub-edss', type=str, required=True, help='Path to a csv file containing subject EDSS scores')
    parser.add_argument('--set', type=str, choices=['train', 'val', 'test', None], default=None, help="Dataset split to evaluate ('train', 'val', 'test' or None for all subjects)")

    return parser.parse_args()


def compare_2_lesion_mappings(pred_mapping, gt_mapping):
    """
    Evaluate the predicted lesion mapping against the ground truth mapping.

    Args:
        pred_mapping (dict): Predicted lesion mapping.
        gt_mapping (dict): Ground truth lesion mapping.
    Returns:
        TP (list): list of True Positives for each lesion.
        FP (list): list of False Positives for each lesion.
        FN (list): list of False Negatives for each lesion.
    """
    TP = []
    FP = []
    FN = []
    print(pred_mapping)
    print(gt_mapping)

    # For each lesion in GT mapping, check if it is correctly mapped in predicted mapping
    for gt_lesion_id, gt_mapped_lesions in gt_mapping.items():
        tp, fp, fn = 0, 0, 0
        if str(gt_lesion_id) in pred_mapping:
            pred_mapped_lesions = pred_mapping[gt_lesion_id]
            # For each gt_mapped_lesion we check if is mapped in the prediction
            for lesion in gt_mapped_lesions:
                if lesion in pred_mapped_lesions:
                    tp += 1
                else:

                    fn += 1
            # For each predicted mapped lesion we check it they are potentially false positives
            for lesion in pred_mapped_lesions:
                if lesion not in gt_mapped_lesions:
                    fp += 1
        else:
            # All lesions in gt_mapped_lesions are false negatives
            fn += len(gt_mapped_lesions)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
    return TP, FP, FN


def lesion_volume(segmentation_path):
    """
    Compute the total lesion volume in mm^3 for a given segmentation.

    Args:
        segmentation_path (str): Path to the segmentation NIfTI file.
    Returns:
        volume_mm3 (float): Total lesion volume in mm^3.
    """
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()
    voxel_volume = np.prod(seg_img.header.get_zooms())  # in mm^3
    # Binarize the segmentation
    seg_data = (seg_data > 0).astype(np.uint8)
    lesion_voxels = np.sum(seg_data)
    volume_mm3 = lesion_voxels * voxel_volume
    return volume_mm3

def evaluate_lesion_mapping(input_msd_dataset, pred_seg, pred_mapping, gt_mapping, output_folder, set = None):
    """
    This file evaluates the lesion mapping results between two timepoints.
    It computes segmentation metrics of the predicted segmentations against the ground truth.
    It also computes lesion matching metrics based on the lesion mapping results and the ground truth mapping.

    Args:
        input_msd_dataset (str): Path to the input MSD dataset.
        predictions_folder (str): Path to the folder where predictions were stored.
        output_folder (str): Path to the output folder where evaluation results will be stored.

    Returns:
        results_dict (dict): Dictionary containing evaluation results for each subject.
    """
    # Initialize logger
    logger.add(os.path.join(output_folder, f'logger_evaluation.log'))
    logger.info(f"Input MSD dataset: {input_msd_dataset}")
    logger.info(f"Predicted segmentations folder: {pred_seg}")
    logger.info(f"Predicted lesion mapping folder: {pred_mapping}")
    logger.info(f"Ground truth lesion mapping folder: {gt_mapping}")
    logger.info(f"Output folder: {output_folder}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the msd dataset
    with open(input_msd_dataset, 'r') as f:
        msd_data = json.load(f)
    data = msd_data['data']
    if set == 'test':
        # Keep only test subjects
        data = {k: v for k, v in data.items() if 'tor' in k}
    elif set == 'train':
        data = {k: v for k, v in data.items() if 'edm' in k or 'van' in k}
    elif set == 'val':
        data = {k: v for k, v in data.items() if 'cal' in k or 'mon' in k}

    # Initilialize results dictionary
    results_dict = {}

    # We run evaluation now on all the data
    for subject in tqdm(data):
        subject_id = subject
        # Build output folder for the subject
        subject_output_folder = os.path.join(output_folder, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)
        # Initialize the timepoints and images
        timepoint1 = "ses-M0"
        timepoint2 = "ses-M12"
        input_image1 = data[subject][timepoint1][0]
        input_image2 = data[subject][timepoint2][0]
        # We build the path to the labeled predicted segmentations, lesion mapping file
        predicted_lesion_seg_1 = os.path.join(pred_seg, subject_id, Path(input_image1).name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        predicted_lesion_seg_2 = os.path.join(pred_seg, subject_id, Path(input_image2).name.replace('.nii.gz', '_lesion-seg-labeled.nii.gz'))
        pred_lesion_mapping_file = os.path.join(pred_mapping, subject_id, 'lesion_mapping.json')
        # Build path to the GT lesion mapping file
        gt_lesion_mapping_file = os.path.join(gt_mapping, subject_id, 'lesion_mapping.json')
        
        # Now we evaluate the converted predicted lesion mapping against the GT lesion mapping
        ## Load the GT lesion mapping
        with open(gt_lesion_mapping_file, 'r') as f:
            gt_lesion_mapping = json.load(f)
        ## Load the predicted lesion mapping
        with open(pred_lesion_mapping_file, 'r') as f:
            pred_lesion_mapping = json.load(f)
        TP, FP, FN = compare_2_lesion_mappings(pred_lesion_mapping, gt_lesion_mapping)
        logger.info(f"Lesion mapping evaluation - TP: {TP}, FP: {FP}, FN: {FN}")

        # Compute total lesion volume in mm^3 for both timepoints and both predicted and GT segmentations
        vol_pred1 = lesion_volume(predicted_lesion_seg_1)
        vol_pred2 = lesion_volume(predicted_lesion_seg_2)

        # Add all the results to a dictionnary
        results = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'vol_pred1_mm3': vol_pred1,
            'vol_pred2_mm3': vol_pred2
        }
        
        # Save the results to a JSON file
        results_file = os.path.join(subject_output_folder, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # We add the results to the overall results dictionary
        results_dict[subject_id] = results

    # Save the overall results to a JSON file
    overall_results_file = os.path.join(output_folder, 'overall_results.json')
    with open(overall_results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Remove the temp folder
    # os.system(f"rm -rf {temp_folder}")

    return results_dict


def plot_evaluation_results(results, output_folder, sub_edss):
    """
    This function plots the evaluation results.

    Input:
        - results: Dictionary containing evaluation results.
        - output_folder: Path to the folder where plots will be saved.
     Output:
        None
    """
    # Initialize a logger
    logger.add(os.path.join(output_folder, 'plot_evaluation.log'))
    # Build a pandas dataframe from the results
    df = pd.DataFrame(results).T

    # Load the subjects EDSS scores
    edss_df = pd.read_csv(sub_edss)
    # Remove some columns in edss_df
    edss_df = edss_df.drop(columns=['file', 'number', 'site', 'name', 'original_name' ])

    # Merge the two dataframes on subject ID
    df = df.merge(edss_df, left_index=True, right_on='sub')

    # For values in EDSS columns, we convert them to float
    for col in edss_df.columns:
        if col != 'sub' and col != 'phenotye':
            # Replace , by . in the column
            df[col] = df[col].str.replace(',', '.')
            # Convert the column to float
            df[col] = df[col].astype(float)

    # Finally for each subject with non-0 EDSS_diff, we look at the correlation between EDSS diff and volume change
    df_nonzero_edss = df[df['EDSS diff'] != 0]
    # We compute the difference in predicted lesion volume
    df_nonzero_edss['vol_pred_diff_mm3'] = df_nonzero_edss['vol_pred2_mm3'] - df_nonzero_edss['vol_pred1_mm3']
    # We want to compute the correlation coefficient on predicted volumes
    res = stats.spearmanr(list(df_nonzero_edss['vol_pred_diff_mm3']), list(df_nonzero_edss['EDSS diff']))
    logger.info(f"Spearman correlation coefficient (predicted volume): {res}")
    res = stats.pearsonr(list(df_nonzero_edss['vol_pred_diff_mm3']), list(df_nonzero_edss['EDSS diff']))
    logger.info(f"Pearson correlation coefficient (predicted volume): {res}")

    # Convert column lesion_mapping_TP, lesion_mapping_FP, lesion_mapping_FN to their average values and std and sum and count
    for columns in ['TP', 'FP', 'FN']:
        df[columns + '_mean'] = df[columns].apply(lambda x: np.mean(x))
        df[columns + '_std'] = df[columns].apply(lambda x: np.std(x))
        df[columns + '_sum'] = df[columns].apply(lambda x: np.sum(x))
        df[columns + '_count'] = df[columns].apply(lambda x: len(x))
    # Now we look at the lesion mapping metrics
    table_mapping = PrettyTable()
    table_mapping.field_names = ["Metric", "Sum over all subjects", "Mean per subject ± std", "Average per lesion ± std"]
    table_mapping.add_row([f"TP", f"{df['TP_sum'].sum()}", f"{df['TP_sum'].mean():.2f} ± {df['TP_sum'].std():.2f}", f"{df['TP_mean'].mean():.2f} ± {df['TP_mean'].std():.2f}"])
    table_mapping.add_row([f"FP", f"{df['FP_sum'].sum()}", f"{df['FP_sum'].mean():.2f} ± {df['FP_sum'].std():.2f}", f"{df['FP_mean'].mean():.2f} ± {df['FP_mean'].std():.2f}"])
    table_mapping.add_row([f"FN", f"{df['FN_sum'].sum()}", f"{df['FN_sum'].mean():.2f} ± {df['FN_sum'].std():.2f}", f"{df['FN_mean'].mean():.2f} ± {df['FN_mean'].std():.2f}"])
    table_mapping.add_row([f"Precision", f"{df['TP_sum'].sum() / (df['TP_sum'].sum() + df['FP_sum'].sum()):.2f}", f"{(df['TP_sum'] / (df['TP_sum'] + df['FP_sum'])).mean():.2f} ± {(df['TP_sum'] / (df['TP_sum'] + df['FP_sum'])).std():.2f}", f"{(df['TP_mean'] / (df['TP_mean'] + df['FP_mean'])).mean():.2f} ± {(df['TP_mean'] / (df['TP_mean'] + df['FP_mean'])).std():.2f}"])
    table_mapping.add_row([f"Recall", f"{df['TP_sum'].sum() / (df['TP_sum'].sum() + df['FN_sum'].sum()):.2f}", f"{(df['TP_sum'] / (df['TP_sum'] + df['FN_sum'])).mean():.2f} ± {(df['TP_sum'] / (df['TP_sum'] + df['FN_sum'])).std():.2f}", f"{(df['TP_mean'] / (df['TP_mean'] + df['FN_mean'])).mean():.2f} ± {(df['TP_mean'] / (df['TP_mean'] + df['FN_mean'])).std():.2f}"])
    table_mapping.add_row([f"F1-score", f"{2 * df['TP_sum'].sum() / (2 * df['TP_sum'].sum() + df['FP_sum'].sum() + df['FN_sum'].sum()):.2f}", f"{(2 * df['TP_sum'] / (2 * df['TP_sum'] + df['FP_sum'] + df['FN_sum'])).mean():.2f} ± {(2 * df['TP_sum'] / (2 * df['TP_sum'] + df['FP_sum'] + df['FN_sum'])).std():.2f}", f"{(2 * df['TP_mean'] / (2 * df['TP_mean'] + df['FP_mean'] + df['FN_mean'])).mean():.2f} ± {(2 * df['TP_mean'] / (2 * df['TP_mean'] + df['FP_mean'] + df['FN_mean'])).std():.2f}"])    
    logger.info("\n" + table_mapping.get_string())

    # Now we perform the same on the train, the validation and the test
    val_df = df[df['sub'].str.contains('cal') | df['sub'].str.contains('mon')]
    test_df = df[df['sub'].str.contains('tor')]
    ## train is all the other subjects
    train_df = df[~df['sub'].str.contains('cal') & ~df['sub'].str.contains('mon') & ~df['sub'].str.contains('tor')]

    for split_name, split_df in zip(['Train', 'Validation', 'Test'], [train_df, val_df, test_df]):
        table_mapping_split = PrettyTable()
        table_mapping_split.field_names = ["Metric", "Sum over all subjects", "Mean per subject ± std", "Average per lesion ± std"]
        table_mapping_split.add_row([f"TP", f"{split_df['TP_sum'].sum()}", f"{split_df['TP_sum'].mean():.2f} ± {split_df['TP_sum'].std():.2f}", f"{split_df['TP_mean'].mean():.2f} ± {split_df['TP_mean'].std():.2f}"])
        table_mapping_split.add_row([f"FP", f"{split_df['FP_sum'].sum()}", f"{split_df['FP_sum'].mean():.2f} ± {split_df['FP_sum'].std():.2f}", f"{split_df['FP_mean'].mean():.2f} ± {split_df['FP_mean'].std():.2f}"])
        table_mapping_split.add_row([f"FN", f"{split_df['FN_sum'].sum()}", f"{split_df['FN_sum'].mean():.2f} ± {split_df['FN_sum'].std():.2f}", f"{split_df['FN_mean'].mean():.2f} ± {split_df['FN_mean'].std():.2f}"])
        table_mapping_split.add_row([f"Precision", f"{split_df['TP_sum'].sum() / (split_df['TP_sum'].sum() + split_df['FP_sum'].sum()):.2f}", f"{(split_df['TP_sum'] / (split_df['TP_sum'] + split_df['FP_sum'])).mean():.2f} ± {(split_df['TP_sum'] / (split_df['TP_sum'] + split_df['FP_sum'])).std():.2f}", f"{(split_df['TP_mean'] / (split_df['TP_mean'] + split_df['FP_mean'])).mean():.2f} ± {(split_df['TP_mean'] / (split_df['TP_mean'] + split_df['FP_mean'])).std():.2f}"])
        table_mapping_split.add_row([f"Recall", f"{split_df['TP_sum'].sum() / (split_df['TP_sum'].sum() + split_df['FN_sum'].sum()):.2f}", f"{(split_df['TP_sum'] / (split_df['TP_sum'] + split_df['FN_sum'])).mean():.2f} ± {(split_df['TP_sum'] / (split_df['TP_sum'] + split_df['FN_sum'])).std():.2f}", f"{(split_df['TP_mean'] / (split_df['TP_mean'] + split_df['FN_mean'])).mean():.2f} ± {(split_df['TP_mean'] / (split_df['TP_mean'] + split_df['FN_mean'])).std():.2f}"])
        table_mapping_split.add_row([f"F1-score", f"{2 * split_df['TP_sum'].sum() / (2 * split_df['TP_sum'].sum() + split_df['FP_sum'].sum() + split_df['FN_sum'].sum()):.2f}", f"{(2 * split_df['TP_sum'] / (2 * split_df['TP_sum'] + split_df['FP_sum'] + split_df['FN_sum'])).mean():.2f} ± {(2 * split_df['TP_sum'] / (2 * split_df['TP_sum'] + split_df['FP_sum'] + split_df['FN_sum'])).std():.2f}", f"{(2 * split_df['TP_mean'] / (2 * split_df['TP_mean'] + split_df['FP_mean'] + split_df['FN_mean'])).mean():.2f} ± {(2 * split_df['TP_mean'] / (2 * split_df['TP_mean'] + split_df['FP_mean'] + split_df['FN_mean'])).std():.2f}"])    
        logger.info(f"\n{split_name} set results:\n" + table_mapping_split.get_string())

    # Finally for all subjects in train set, we plot the sum of TP and FN and add them to a list
    all_tp = []
    all_fn = []
    sum_tp_fn = []
    for subject, row in df.iterrows():
        # Keep only subjects in the train set
        if 'cal' in row['sub'] or 'mon' in row['sub'] or 'tor' in row['sub']:
            continue
        tp = row['TP_sum']
        fn = row['FN_sum']
        all_tp.append(tp)
        all_fn.append(fn)
        sum_tp_fn.append(tp + fn)
    print(list(sum_tp_fn))

    return None


if __name__ == "__main__":
    args = parse_args()
    input_msd_dataset = args.input_msd
    pred_seg = args.pred_seg
    pred_mapping = args.pred_mapping
    gt_mapping = args.gt_mapping
    output_folder = args.output_folder
    sub_edss = args.sub_edss

    # Run the evaluation
    results = evaluate_lesion_mapping(input_msd_dataset, pred_seg, pred_mapping, gt_mapping, output_folder, set=args.set)

    # results_path = "~/net/longitudinal_ms/20251121_lesion_matching_reg_com_results_debug/overall_results.json"
    # with open(os.path.expanduser(results_path), 'r') as f:
    #     results = json.load(f)

    plot_evaluation_results(results, output_folder, sub_edss)
