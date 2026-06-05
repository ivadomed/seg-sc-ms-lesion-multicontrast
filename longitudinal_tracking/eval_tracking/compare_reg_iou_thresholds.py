"""
Compare registered-IoU lesion tracking performance across IoU threshold values.

Reads the summary JSON produced by sweep_reg_iou_threshold.py and generates:
  - A line plot of Precision / Recall / F1 vs IoU threshold (log-scale x-axis)
  - A PrettyTable summary in the log
  - A JSON with the full comparison results

Inputs:
    -i / --summary-json  : Path to summary_iou_sweep.json from sweep_reg_iou_threshold.py
    -o / --output-folder : Output folder for plots and results

Outputs:
    iou_threshold_sweep_plot.png
    iou_threshold_comparison.json
    compare_reg_iou_thresholds.log

Author: Pierre-Louis Benveniste
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from prettytable import PrettyTable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot and compare reg-IoU performance across IoU thresholds."
    )
    parser.add_argument('-i', '--summary-json', type=str, required=True,
                        help='Path to summary_iou_sweep.json produced by sweep_reg_iou_threshold.py')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Output folder for plots and results')
    return parser.parse_args()


def load_summary(path):
    """Load and sort summary by threshold value."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Sort entries by threshold value
    entries = sorted(data.values(), key=lambda x: x['threshold'])
    return entries


def print_summary_table(entries):
    """Print a formatted PrettyTable with all threshold results."""
    tbl = PrettyTable()
    tbl.field_names = ['Threshold', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN']
    tbl.align['Threshold'] = 'r'
    for m in ['Precision', 'Recall', 'F1']:
        tbl.align[m] = 'r'

    best_f1 = max(entries, key=lambda x: x['f1'])
    best_p  = max(entries, key=lambda x: x['precision'])
    best_r  = max(entries, key=lambda x: x['recall'])

    for e in entries:
        t = e['threshold']
        t_str = f"{t:.0e}" if t < 0.01 else f"{t:.2f}"
        tbl.add_row([
            t_str,
            f"{e['precision']:.4f}" + (' *' if e is best_p  else ''),
            f"{e['recall']:.4f}"    + (' *' if e is best_r  else ''),
            f"{e['f1']:.4f}"        + (' *' if e is best_f1 else ''),
            e['total_TP'],
            e['total_FP'],
            e['total_FN'],
        ])

    logger.info("\nPerformance across IoU thresholds (* = best for that metric):\n" + tbl.get_string())


def plot_metrics(entries, output_folder):
    """Plot Precision / Recall / F1 vs IoU threshold on a log-scale x-axis."""
    thresholds = [e['threshold'] for e in entries]
    precisions = [e['precision'] for e in entries]
    recalls    = [e['recall']    for e in entries]
    f1s        = [e['f1']        for e in entries]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, precisions, 'o-', label='Precision', color='steelblue',  linewidth=2, markersize=6)
    ax.plot(thresholds, recalls,    's-', label='Recall',    color='darkorange', linewidth=2, markersize=6)
    ax.plot(thresholds, f1s,        '^-', label='F1',        color='forestgreen', linewidth=2, markersize=6)

    # Mark best F1
    best_idx = int(np.argmax(f1s))
    ax.axvline(thresholds[best_idx], color='forestgreen', linestyle='--', alpha=0.5,
               label=f'Best F1 threshold = {thresholds[best_idx]:.2e}')

    ax.set_xscale('log')
    ax.set_xlabel('IoU Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Reg+IoU tracking performance vs IoU threshold', fontsize=13)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Annotate each point with its threshold value
    for t, p, r, f in zip(thresholds, precisions, recalls, f1s):
        t_str = f"{t:.0e}" if t < 0.01 else f"{t:.2f}"
        ax.annotate(t_str, xy=(t, f), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=7, color='forestgreen')

    plt.tight_layout()
    out_path = os.path.join(output_folder, 'iou_threshold_sweep_plot.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Plot saved to {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'compare_reg_iou_thresholds.log'))

    logger.info(f"Loading summary from: {args.summary_json}")
    entries = load_summary(args.summary_json)
    logger.info(f"Loaded {len(entries)} threshold entries")

    # Print table
    print_summary_table(entries)

    # Plot
    plot_metrics(entries, args.output_folder)

    # Identify best thresholds
    best_f1  = max(entries, key=lambda x: x['f1'])
    best_p   = max(entries, key=lambda x: x['precision'])
    best_r   = max(entries, key=lambda x: x['recall'])

    def fmt(t):
        return f"{t:.0e}" if t < 0.01 else f"{t:.2f}"

    logger.info(f"Best threshold by F1        : {fmt(best_f1['threshold'])}  "
                f"(F1={best_f1['f1']:.4f}  P={best_f1['precision']:.4f}  R={best_f1['recall']:.4f})")
    logger.info(f"Best threshold by Precision : {fmt(best_p['threshold'])}  "
                f"(P={best_p['precision']:.4f})")
    logger.info(f"Best threshold by Recall    : {fmt(best_r['threshold'])}  "
                f"(R={best_r['recall']:.4f})")

    # Save comparison JSON
    output = {
        'best_by_f1':        best_f1,
        'best_by_precision': best_p,
        'best_by_recall':    best_r,
        'all_thresholds':    entries,
    }
    out_json = os.path.join(args.output_folder, 'iou_threshold_comparison.json')
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=4)
    logger.info(f"Comparison results saved to {out_json}")


if __name__ == "__main__":
    main()
