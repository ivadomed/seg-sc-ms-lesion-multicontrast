"""
This script performs statistical tests to compare the performance of strat5 and other strategies.
Strat 3 was not included in the computation of the score for faster coding due to short deadline for rebuttal.

Input:
    -i1 : path to the results JSON file for strat1
    -i2 : path to the results JSON file for strat2
    -i4 : path to the results JSON file for strat4
    -i5 : path to the results JSON file for strat5
    -o : path to the output directory

Author: Pierre-Louis Benveniste
"""
import argparse
import json
import os
from loguru import logger
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests


def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform statistical tests on strat5 experiment results.")
    parser.add_argument("-i1", "--input1", required=True, help="Path to the results JSON file for strat1.")
    parser.add_argument("-i2", "--input2", required=True, help="Path to the results JSON file for strat2.")
    parser.add_argument("-i4", "--input4", required=True, help="Path to the results JSON file for strat4.")
    parser.add_argument("-i5", "--input5", required=True, help="Path to the results JSON file for strat5.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output directory.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_folder = args.output

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Initialize a logger
    logger.add(os.path.join(output_folder, 'stat_test_strat_perf.log'))

    # Load results from JSON files
    with open(args.input1, 'r') as f:
        results_strat1 = json.load(f)
    with open(args.input2, 'r') as f:
        results_strat2 = json.load(f)
    with open(args.input4, 'r') as f:
        results_strat4 = json.load(f)
    with open(args.input5, 'r') as f:
        results_strat5 = json.load(f)

    # Take care of srat 1,4 and 5 since the format of their results is the same
    strat_idx = [1, 4, 5]
    strategies_results = [results_strat1,results_strat4, results_strat5]

    # Initialize the results_dataframe
    tp_df = {}
    fp_df = {}
    fn_df = {}
    for idx, strat_result in zip(strat_idx, strategies_results):
        # Initialize the dataframe for each strategy
        tp_df[f'strat{idx}'] = {}
        fp_df[f'strat{idx}'] = {}
        fn_df[f'strat{idx}'] = {}
        for subject in strat_result.keys():
            tp_df[f'strat{idx}'][subject] = sum(strat_result[subject]['TP'])
            fp_df[f'strat{idx}'][subject] = sum(strat_result[subject]['FP'])
            fn_df[f'strat{idx}'][subject] = sum(strat_result[subject]['FN'])

    # Now we perform the same extraction for strat2 which has a different format
    tp_df['strat2'] = {}
    fp_df['strat2'] = {}
    fn_df['strat2'] = {}
    for subject in results_strat2.keys():
        tp_df['strat2'][subject] = results_strat2[subject]['TP']
        fp_df['strat2'][subject] = results_strat2[subject]['FP']
        fn_df['strat2'][subject] = results_strat2[subject]['FN']
     
    # Build dataframes where each column is a strategy and each row is a subject
    tp_df = pd.DataFrame(tp_df)
    fp_df = pd.DataFrame(fp_df)
    fn_df = pd.DataFrame(fn_df)

    # Compute the precision, recall, and F1-score for each strategy
    precision = tp_df / (tp_df + fp_df)
    recall = tp_df / (tp_df + fn_df)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # We only keep the test subjects (with 'tor' in the subject key)
    test_precision = precision[precision.index.str.contains('tor', na=False)]
    test_recall = recall[recall.index.str.contains('tor', na=False)]
    test_f1_score = f1_score[f1_score.index.str.contains('tor', na=False)]

    # We now perform statistical tests to compare strat5 with only strat2
    # Wilcoxon signed-rank test for pairwise comparisons
    strategies = ['strat2']
    metrics = {'Precision': test_precision, 'Recall': test_recall, 'F1-Score': test_f1_score}
    results = []
    for metric_name, metric_df in metrics.items():
        strat5_values = metric_df['strat5']
        for strat in strategies:
            stat, p_value = wilcoxon(strat5_values, metric_df[strat])
            results.append({
                'Metric': metric_name,
                'Comparison': f'strat5 vs {strat}',
                'Statistic': stat,
                'P-Value': p_value
            })

    results_df = pd.DataFrame(results)
    print("Statistical Test Results on the test set (n=9) (Wilcoxon signed-rank test):")
    print(results_df)

    # Now we perform the same statistical tests but on the entire set (train and test)
    strategies = ['strat2']
    metrics = {'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}
    results = []
    for metric_name, metric_df in metrics.items():
        strat5_values = metric_df['strat5']
        for strat in strategies:
            stat, p_value = wilcoxon(strat5_values, metric_df[strat])
            results.append({
                'Metric': metric_name,
                'Comparison': f'strat5 vs {strat}',
                'Statistic': stat,
                'P-Value': p_value
            })
    results_df_full = pd.DataFrame(results)
    print("Statistical Test Results on Full Set (n=34) (Wilcoxon signed-rank test):")
    print(results_df_full)



if __name__ == "__main__":
    main()