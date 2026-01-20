"""
This script analyses the results from the strat5 experiments.
It produces global performance, per-participant performance, and per-site performance.

Input:
    -i : path to the results JSON file
    -o : path to the output directory

Author: Pierre-Louis Benveniste
"""
import argparse
import json
import os
import pandas as pd
from loguru import logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyse strat5 experiment results.")
    parser.add_argument("-i", "--input", required=True, help="Path to the results JSON file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_folder = args.output

    # Initialize a logger
    logger.add(os.path.join(output_folder, 'results_strat5_analysis.log'))

    # Load results from JSON file
    with open(args.input, 'r') as f:
        results = json.load(f)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # intialize the dataframe to store everything
    metrics_df = pd.DataFrame(columns=[
        'Subject', 'Test', 'Site', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score'
    ])

    for subject, metrics in results.items():
        site = subject.split('-')[1][:3]

        sub_tp = sum(metrics.get('TP', []))
        sub_fp = sum(metrics.get('FP', []))
        sub_fn = sum(metrics.get('FN', []))

        precision = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 0
        recall = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0
        f1_score = (2*precision*recall) / (precision + recall) if (precision + recall) > 0 else 0

        if site=='tor':
            test = True
        else:
            test = False
        
        # We add each subject's metrics to the global metrics
        new_row = {
            'Subject': subject,
            'Test': test,
            'Site': site,
            'TP': sub_tp,
            'FP': sub_fp,
            'FN': sub_fn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([new_row])], ignore_index=True)

    # print global performance
    global_tp = metrics_df['TP'].sum()
    global_fp = metrics_df['FP'].sum()
    global_fn = metrics_df['FN'].sum()
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    global_f1_score = (2*global_precision*global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0

    logger.info("Global Performance: on train and test sets combined")
    logger.info(f"TP: {global_tp}")
    logger.info(f"FP: {global_fp}")
    logger.info(f"FN: {global_fn}")
    logger.info(f"Precision: {global_precision:.4f}")
    logger.info(f"Recall: {global_recall:.4f}")
    logger.info(f"F1-Score: {global_f1_score:.4f}")

    # print performance on train/val set
    logger.info("\nPerformance on Train/Val Set:")
    train_df = metrics_df[metrics_df['Test'] == False]
    train_tp = train_df['TP'].sum()
    train_fp = train_df['FP'].sum()
    train_fn = train_df['FN'].sum()
    train_precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0
    train_recall = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0
    train_f1_score = (2*train_precision*train_recall) / (train_precision + train_recall) if (train_precision + train_recall) > 0 else 0
    logger.info(f"TP: {train_tp}")
    logger.info(f"FP: {train_fp}")
    logger.info(f"FN: {train_fn}")
    logger.info(f"Precision: {train_precision:.4f}")
    logger.info(f"Recall: {train_recall:.4f}")
    logger.info(f"F1-Score: {train_f1_score:.4f}")

    # print performance on test set
    logger.info("\nPerformance on Test Set:")
    test_df = metrics_df[metrics_df['Test'] == True]
    test_tp = test_df['TP'].sum()
    test_fp = test_df['FP'].sum()
    test_fn = test_df['FN'].sum()
    test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
    test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_f1_score = (2*test_precision*test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    logger.info(f"TP: {test_tp}")
    logger.info(f"FP: {test_fp}")
    logger.info(f"FN: {test_fn}")
    logger.info(f"Precision: {test_precision:.4f}")
    logger.info(f"Recall: {test_recall:.4f}")
    logger.info(f"F1-Score: {test_f1_score:.4f}")

    # print performance per site
    logger.info("\nPerformance per Site:")
    for site in metrics_df['Site'].unique():
        site_df = metrics_df[metrics_df['Site'] == site]
        site_tp = site_df['TP'].sum()
        site_fp = site_df['FP'].sum()
        site_fn = site_df['FN'].sum()
        site_precision = site_tp / (site_tp + site_fp) if (site_tp + site_fp) > 0 else 0
        site_recall = site_tp / (site_tp + site_fn) if (site_tp + site_fn) > 0 else 0
        site_f1_score = (2*site_precision*site_recall) / (site_precision + site_recall) if (site_precision + site_recall) > 0 else 0
        logger.info(f"\nSite: {site}")
        # print the number of subjects for this site
        num_subjects = site_df.shape[0]
        logger.info(f"Number of Subjects: {num_subjects}")
        logger.info(f"TP: {site_tp}")
        logger.info(f"FP: {site_fp}")
        logger.info(f"FN: {site_fn}")
        logger.info(f"Precision: {site_precision:.4f}")
        logger.info(f"Recall: {site_recall:.4f}")
        logger.info(f"F1-Score: {site_f1_score:.4f}")

    # print avg precision, recall and f1 score
    logger.info("\nAverage Metrics across Participants:")
    avg_tp = metrics_df['TP'].mean()
    avg_fp = metrics_df['FP'].mean()
    avg_fn = metrics_df['FN'].mean()
    avg_precision = metrics_df['Precision'].mean()
    avg_recall = metrics_df['Recall'].mean()
    avg_f1_score = metrics_df['F1-Score'].mean()
    logger.info(f"Average TP: {avg_tp:.2f}")
    logger.info(f"Average FP: {avg_fp:.2f}")
    logger.info(f"Average FN: {avg_fn:.2f}")
    logger.info(f"Average Precision: {avg_precision:.4f}")
    logger.info(f"Average Recall: {avg_recall:.4f}")
    logger.info(f"Average F1-Score: {avg_f1_score:.4f}")

    # print avg precision, recall and f1 score on the train set
    logger.info("\nAverage Metrics on Train/val Set across Participants:")
    avg_train_tp = train_df['TP'].mean()
    avg_train_fp = train_df['FP'].mean()
    avg_train_fn = train_df['FN'].mean()
    train_df = metrics_df[metrics_df['Test'] == False]
    avg_train_precision = train_df['Precision'].mean()
    avg_train_recall = train_df['Recall'].mean()
    avg_train_f1_score = train_df['F1-Score'].mean()
    logger.info(f"Average TP (Train/val Set): {avg_train_tp:.2f}")
    logger.info(f"Average FP (Train/val Set): {avg_train_fp:.2f}")
    logger.info(f"Average FN (Train/val Set): {avg_train_fn:.2f}")
    logger.info(f"Average Precision (Train/val Set): {avg_train_precision:.2f}")
    logger.info(f"Average Recall (Train/val Set): {avg_train_recall:.2f}")
    logger.info(f"Average F1-Score (Train/val Set): {avg_train_f1_score:.2f}")

    # print avg precision, recall and f1 score on the test set
    logger.info("\nAverage Metrics on Test Set across Participants:")
    avg_test_tp = test_df['TP'].mean()
    avg_test_fp = test_df['FP'].mean()
    avg_test_fn = test_df['FN'].mean()
    avg_test_precision = test_df['Precision'].mean()
    avg_test_recall = test_df['Recall'].mean()
    avg_test_f1_score = test_df['F1-Score'].mean()
    logger.info(f"Average TP (Test Set): {avg_test_tp:.2f}")
    logger.info(f"Average FP (Test Set): {avg_test_fp:.2f}")
    logger.info(f"Average FN (Test Set): {avg_test_fn:.2f}")
    logger.info(f"Average Precision (Test Set): {avg_test_precision:.4f}")
    logger.info(f"Average Recall (Test Set): {avg_test_recall:.4f}")
    logger.info(f"Average F1-Score (Test Set): {avg_test_f1_score:.4f}")

    return None


if __name__ == "__main__":
    main()