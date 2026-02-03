"""
Lesion Volume Precision Analysis Script
This script analyzes the precision of soft versus binary segmentation for lesion volumes.
It groups data by subject, calculates the Coefficient of Variation (CV) for repeated measures,
and performs statistical tests to compare the two methods.

Inputs:
    - input: Input CSV file containing lesion volume data.
    - output_folder: Path to the output folder to save plots and results. 

Outputs:
    - precision_boxplot.png: Visual comparison of Coefficient of Variation.
    - precision_scatter.png: Scatter plot comparing Soft vs Binary CV.

Author: Pierre-Louis Benveniste 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import argparse
import sys
from loguru import logger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze lesion volume precision from CSV data.")
    parser.add_argument("-i", "--input", dest="csv_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_folder", dest="output_folder", required=True, help="Path to the output folder.")
    return parser.parse_args()


def analyze_precision(csv_path, output_folder):
    # Build output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load the CSV file
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # We also want to save all results to log file
    log_path = os.path.join(output_folder, "precision_analysis.log")
    logger.add(log_path, format="{time} {level} {message}", level="INFO")
    logger.info(f"Loaded data from {csv_path} with {len(df)} entries.")
    
    # Group by subject and collect non-null values
    results = []
    
    subjects = df['subject'].unique()
    logger.info(f"Found {len(subjects)} subjects for analysis.")

    
    for sub in subjects:
        sub_df = df[df['subject'] == sub]
        
        soft_vols = sub_df['soft_volume'].dropna().values
        binary_vols = sub_df['binary_volume'].dropna().values
        
        # We expect ~24 values.
        n_soft = len(soft_vols)
        n_binary = len(binary_vols)
        
        if n_soft < 2 or n_binary < 2:
            logger.info(f"Skipping {sub}: Not enough data (Soft: {n_soft}, Binary: {n_binary})")
            continue
            
        # Calculate stats
        soft_mean = np.mean(soft_vols)
        soft_sd = np.std(soft_vols, ddof=1)
        soft_cv = (soft_sd / soft_mean) * 100 if soft_mean > 0 else 0
        
        bin_mean = np.mean(binary_vols)
        bin_sd = np.std(binary_vols, ddof=1)
        bin_cv = (bin_sd / bin_mean) * 100 if bin_mean > 0 else 0
        
        results.append({
            'Subject': sub,
            'Soft_Mean': soft_mean,
            'Soft_STD': soft_sd,
            'Soft_CV': soft_cv,
            'Binary_Mean': bin_mean,
            'Binary_STD': bin_sd,
            'Binary_CV': bin_cv,
            'N_Soft': n_soft,
            'N_Binary': n_binary
        })
        
    results_df = pd.DataFrame(results)
    
    # --- Statistical Comparison ---
    logger.info("\n--- Summary Statistics ---")
    logger.info(results_df[['Soft_CV', 'Binary_CV', 'Soft_STD', 'Binary_STD']].describe())
    
    # Paired T-test
    t_stat_cv, p_val_cv = stats.ttest_rel(results_df['Soft_CV'], results_df['Binary_CV'])
    t_stat_sd, p_val_sd = stats.ttest_rel(results_df['Soft_STD'], results_df['Binary_STD'])

    # Wilcoxon (non-parametric, robust to outliers)
    w_stat_cv, p_w_cv = stats.wilcoxon(results_df['Soft_CV'], results_df['Binary_CV'])

    summary_msg = (
        f"\nComparison (N={len(results_df)}):\n"
        f"Mean CV Soft: {results_df['Soft_CV'].mean():.2f}%\n"
        f"Mean CV Binary: {results_df['Binary_CV'].mean():.2f}%\n"
        f"CV Reduction with Soft: {results_df['Binary_CV'].mean() - results_df['Soft_CV'].mean():.2f}%\n"
        f"Paired t-test p-value (CV): {p_val_cv}\n"
        f"Wilcoxon p-value (CV): {p_w_cv}\n"
    )
    logger.info(summary_msg)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Boxplot of CVs
    plt.figure(figsize=(8, 6))
    plot_data = results_df[['Soft_CV', 'Binary_CV']].melt(var_name='Method', value_name='CV (%)')
    plot_data['Method'] = plot_data['Method'].replace({'Soft_CV': 'Soft Seg', 'Binary_CV': 'Binary Seg'})
    
    sns.boxplot(x='Method', y='CV (%)', data=plot_data, width=0.5, palette="Set2")
    sns.stripplot(x='Method', y='CV (%)', data=plot_data, color='black', alpha=0.3, jitter=0.1)
    
    plt.title('Lesion Volume Precision: Coefficient of Variation')
    plt.ylabel('Coefficient of Variation (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'precision_boxplot.png'))
    logger.info(f"Saved {os.path.join(output_folder, 'precision_boxplot.png')}")

    # 2. Scatter plot
    plt.figure(figsize=(8, 8))
    # Add diagonal line
    max_val = max(results_df['Soft_CV'].max(), results_df['Binary_CV'].max()) * 1.05
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Precision')
    
    sns.scatterplot(x='Binary_CV', y='Soft_CV', data=results_df, s=100, alpha=0.7)
    
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('Binary Segmentation CV (%)')
    plt.ylabel('Soft Segmentation CV (%)')
    plt.title('Precision Comparison per Subject')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'precision_scatter.png'))
    logger.info(f"Saved {os.path.join(output_folder, 'precision_scatter.png')}")

if __name__ == "__main__":
    # Parse the arguments
    args = parse_arguments()
    # Analyze the lesion volume precision
    analyze_precision(args.csv_file, args.output_folder)