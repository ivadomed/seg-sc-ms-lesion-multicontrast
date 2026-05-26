"""
Compare lesion-tracking method performance with statistical tests.

Loads per-subject (or per-fold) results from each method, aligns them on the
common set of observations, then runs:
  1. Friedman test — non-parametric K-method comparison for each metric.
  2. Post-hoc pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

Result formats handled automatically:
  ┌─────────────────────────────┬──────────────────────────────────────────────────┐
  │ Script                      │ Expected JSON structure                          │
  ├─────────────────────────────┼──────────────────────────────────────────────────┤
  │ loocv_xgb / loocv_siamese   │ {subjects:[..], tp_scores:[..], ...}             │
  │ loocv_unreg                 │ {per_subject: {sub: {total_TP, total_FP, ...}}}  │
  │ fivefold_cv_* (any method)  │ {folds: [{fold, test_subjects, total_TP, ...}]}  │
  │ run_evaluation (reg methods)│ {sub: {TP:[..], FP:[..], FN:[..]}}               │
  └─────────────────────────────┴──────────────────────────────────────────────────┘

Note: LOOCV and reg-method files yield one observation per subject (recommended).
      5-fold CV files yield one observation per fold (only 5 data points — weak
      statistical power). Do not mix LOOCV and 5-fold CV files in the same run.

Inputs (provide at least 2; all are optional):
    --unreg    : path to unreg CV results JSON
    --xgb      : path to XGB CV results JSON
    --siamese  : path to Siamese CV results JSON
    --reg-iou  : path to reg-IoU evaluation results JSON (overall_results.json)
    --reg-com  : path to reg-CoM evaluation results JSON (overall_results.json)
    -o         : output folder

Output:
    comparison_results.json        — full numerical results
    compare_methods.log            — human-readable summary + test tables
    boxplot_{precision,recall,f1}.png

Author: Pierre-Louis Benveniste
"""
import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from prettytable import PrettyTable


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unreg',   type=str, default=None,
                        help='Path to unreg CV results JSON')
    parser.add_argument('--xgb',     type=str, default=None,
                        help='Path to XGB CV results JSON')
    parser.add_argument('--siamese', type=str, default=None,
                        help='Path to Siamese CV results JSON')
    parser.add_argument('--reg-iou', type=str, default=None,
                        help='Path to reg-IoU overall_results.json')
    parser.add_argument('--reg-com', type=str, default=None,
                        help='Path to reg-CoM overall_results.json')
    parser.add_argument('-o', '--output-folder', type=str, required=True,
                        help='Output folder')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Result loading — normalise every format to {obs_key: {tp, fp, fn}}
# ---------------------------------------------------------------------------

def load_results(path):
    """
    Load a results JSON and return a dict mapping observation key → {tp, fp, fn}.

    Observation keys are subject IDs for LOOCV / reg-method files, or
    'fold_N' strings for 5-fold CV files.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    obs = {}

    # ---- Format A: loocv_xgb / loocv_siamese --------------------------------
    # {'subjects': [...], 'tp_scores': [...], 'fp_scores': [...], 'fn_scores': [...]}
    if 'subjects' in data:
        for sub, tp, fp, fn in zip(
            data['subjects'], data['tp_scores'], data['fp_scores'], data['fn_scores']
        ):
            obs[sub] = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}

    # ---- Format B: loocv_unreg -----------------------------------------------
    # {'per_subject': {sub: {'total_TP': .., 'total_FP': .., 'total_FN': ..}}}
    elif 'per_subject' in data:
        for sub, m in data['per_subject'].items():
            obs[sub] = {'tp': int(sum(m["TP"]) if isinstance(m['TP'], list) else m['TP']),
                        'fp': int(sum(m["FP"]) if isinstance(m['FP'], list) else m['FP']),
                        'fn': int(sum(m["FN"]) if isinstance(m['FN'], list) else m['FN'])}

    # ---- Format D/E: fivefold_cv_* -------------------------------------------
    # {'folds': [{'fold': 1, 'total_TP': .., 'total_FP': .., 'total_FN': ..}, ...]}
    elif 'folds' in data:
        for fold in data['folds']:
            key = f"fold_{fold['fold']}"
            obs[key] = {
                'tp': int(fold['total_TP']),
                'fp': int(fold['total_FP']),
                'fn': int(fold['total_FN']),
            }

    # ---- Format C: run_evaluation overall_results.json (reg methods) ---------
    # {sub: {'TP': [...], 'FP': [...], 'FN': [...]}}
    else:
        for sub, m in data.items():
            tp = sum(m['TP']) if isinstance(m['TP'], list) else int(m['TP'])
            fp = sum(m['FP']) if isinstance(m['FP'], list) else int(m['FP'])
            fn = sum(m['FN']) if isinstance(m['FN'], list) else int(m['FN'])
            obs[sub] = {'tp': tp, 'fp': fp, 'fn': fn}

    return obs


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def build_metric_dataframes(method_results):
    """
    Build one DataFrame per metric (Precision / Recall / F1), rows = common
    observations, columns = methods.

    method_results : {method_name: {obs_key: {tp, fp, fn}}}
    """
    common_keys = sorted(
        set.intersection(*[set(v.keys()) for v in method_results.values()])
    )
    logger.info(f"Common observations across all methods: {len(common_keys)}")
    if len(common_keys) == 0:
        raise ValueError(
            "No common observations found. Make sure all result files use the "
            "same evaluation strategy (all LOOCV or all 5-fold CV)."
        )

    p_df  = pd.DataFrame(index=common_keys, dtype=float)
    r_df  = pd.DataFrame(index=common_keys, dtype=float)
    f1_df = pd.DataFrame(index=common_keys, dtype=float)

    for method, results in method_results.items():
        ps, rs, f1s = [], [], []
        for key in common_keys:
            m = results[key]
            p, r, f1 = prf1(m['tp'], m['fp'], m['fn'])
            ps.append(p); rs.append(r); f1s.append(f1)
        p_df[method]  = ps
        r_df[method]  = rs
        f1_df[method] = f1s


    return {'Precision': p_df, 'Recall': r_df, 'F1': f1_df}


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_tests(metric_dfs):
    """
    For each metric:
      1. Friedman test (all methods).
      2. Pairwise Wilcoxon signed-rank + Holm-Bonferroni correction.

    Returns a nested dict of results.
    """
    test_results = {}

    for metric, df in metric_dfs.items():
        methods = list(df.columns)

        # Friedman test
        friedman_stat, friedman_p = friedmanchisquare(*[df[m].values for m in methods])
        logger.info(
            f"\n{metric} — Friedman: χ²={friedman_stat:.4f}, p={friedman_p:.4f}"
        )

        # Pairwise Wilcoxon
        pairs  = list(itertools.combinations(methods, 2))
        raw_ps, w_stats = [], []
        for m1, m2 in pairs:
            try:
                stat, p = wilcoxon(df[m1].values, df[m2].values)
            except ValueError:
                stat, p = 0.0, 1.0   # all differences are zero
            w_stats.append(float(stat))
            raw_ps.append(float(p))

        # Holm-Bonferroni correction
        reject, p_corr, _, _ = multipletests(raw_ps, method='holm')

        pairwise = []
        for (m1, m2), stat, raw_p, corr_p, rej in zip(pairs, w_stats, raw_ps, p_corr, reject):
            pairwise.append({
                'method_1':    m1,
                'method_2':    m2,
                'statistic':   round(stat, 4),
                'p_raw':       round(float(raw_p), 4),
                'p_corrected': round(float(corr_p), 4),
                'significant': bool(rej),
            })

        test_results[metric] = {
            'friedman_statistic': round(float(friedman_stat), 4),
            'friedman_p':         round(float(friedman_p), 4),
            'pairwise':           pairwise,
        }

    return test_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(metric_dfs, test_results):
    methods = list(list(metric_dfs.values())[0].columns)

    # Performance table
    perf = PrettyTable()
    perf.field_names = ['Method', 'Precision (mean±std)', 'Recall (mean±std)', 'F1 (mean±std)']
    for m in methods:
        perf.add_row([
            m,
            f"{metric_dfs['Precision'][m].mean():.3f} ± {metric_dfs['Precision'][m].std():.3f}",
            f"{metric_dfs['Recall'][m].mean():.3f} ± {metric_dfs['Recall'][m].std():.3f}",
            f"{metric_dfs['F1'][m].mean():.3f} ± {metric_dfs['F1'][m].std():.3f}",
        ])
    logger.info("\nPerformance summary (per-observation mean ± std):\n" + perf.get_string())

    # Per-metric statistical test tables
    for metric, res in test_results.items():
        logger.info(
            f"\n{metric} — Friedman test: "
            f"χ²={res['friedman_statistic']:.4f},  p={res['friedman_p']:.4f}"
        )
        tbl = PrettyTable()
        tbl.field_names = ['Method A', 'Method B', 'W statistic', 'p (raw)', 'p (Holm)', 'Sig.']
        for row in res['pairwise']:
            tbl.add_row([
                row['method_1'], row['method_2'],
                f"{row['statistic']:.2f}",
                f"{row['p_raw']:.4f}",
                f"{row['p_corrected']:.4f}",
                '✓' if row['significant'] else '',
            ])
        logger.info(
            f"Pairwise Wilcoxon signed-rank (Holm-Bonferroni correction):\n"
            + tbl.get_string()
        )

    # Best method per metric
    logger.info("\nBest method per metric (highest mean):")
    for metric, df in metric_dfs.items():
        best = df.mean().idxmax()
        logger.info(f"  {metric}: {best}  ({df[best].mean():.3f} ± {df[best].std():.3f})")


def plot_boxplots(metric_dfs, output_folder):
    for metric, df in metric_dfs.items():
        fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 1.8), 5))
        bp = ax.boxplot(
            [df[m].values for m in df.columns],
            labels=df.columns,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
        )
        colours = plt.cm.Set2.colors
        for patch, colour in zip(bp['boxes'], colours):
            patch.set_facecolor(colour)
        ax.set_title(f'{metric} distribution across methods', fontsize=13)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylim([-0.05, 1.05])
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        path = os.path.join(output_folder, f'boxplot_{metric.lower()}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    logger.add(os.path.join(args.output_folder, 'compare_methods.log'))

    # Collect provided method paths
    method_paths = {
        'unreg':   args.unreg,
        'xgb':     args.xgb,
        'siamese': args.siamese,
        'reg_iou': args.reg_iou,
        'reg_com': args.reg_com,
    }
    method_paths = {k: v for k, v in method_paths.items() if v is not None}

    if len(method_paths) < 2:
        raise ValueError("At least 2 method result files must be provided.")

    logger.info(f"Methods to compare: {list(method_paths.keys())}")

    # Load and normalise results
    method_results = {}
    for method, path in method_paths.items():
        logger.info(f"Loading '{method}' from {path}")
        method_results[method] = load_results(path)
        logger.info(f"  → {len(method_results[method])} observations loaded")

    # Build metric DataFrames (aligned on common observations)
    metric_dfs = build_metric_dataframes(method_results)
    n_obs = len(list(metric_dfs.values())[0])
    logger.info(f"\nStatistical tests will use {n_obs} observations per method.")
    if n_obs < 10:
        logger.warning(
            f"Only {n_obs} observations — statistical power will be low. "
            "Consider using LOOCV results for more reliable tests."
        )

    # Statistical tests
    test_results = run_tests(metric_dfs)

    # Human-readable summary
    print_summary(metric_dfs, test_results)

    # Box plots
    plot_boxplots(metric_dfs, args.output_folder)

    # Save full results to JSON
    output = {
        'n_observations': n_obs,
        'methods':        list(method_paths.keys()),
        'performance': {
            metric: {
                method: {
                    'mean':   round(float(df[method].mean()), 4),
                    'std':    round(float(df[method].std()),  4),
                    'median': round(float(df[method].median()), 4),
                    'min':    round(float(df[method].min()),  4),
                    'max':    round(float(df[method].max()),  4),
                }
                for method in df.columns
            }
            for metric, df in metric_dfs.items()
        },
        'statistical_tests': test_results,
    }
    out_path = os.path.join(args.output_folder, 'comparison_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=4)
    logger.info(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
