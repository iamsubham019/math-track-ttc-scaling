"""
Aggregate all logs/*.jsonl into one results table and an
Accuracy-vs-Estimated-FLOPs plot, ready for the paper's math-track results
section.

Plots estimated_flops (from shared/flop_utils.py's standard 2*N_params*
N_tokens inference approximation, computed per-problem inside every
strategy — see src/flop_helper.py) on the x-axis, NOT avg_output_tokens.
Token count alone was a misleading compute proxy for two reasons: (1) it
can't distinguish a 1B-model token from a 3B-model token, which cost very
different amounts of real compute, and (2) for Tree Search specifically it
entirely missed the extra forward passes spent on verifier/judge scoring
calls during beam pruning — real compute that no amount of counting
*generation* tokens would capture. estimated_flops accounts for both.

Example:
    python scripts/evaluate_all.py
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.model_utils import load_config

try:
    from adjustText import adjust_text
    _HAVE_ADJUST_TEXT = True
except ImportError:
    _HAVE_ADJUST_TEXT = False


def load_log(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    fname = os.path.basename(path).replace(".jsonl", "")
    df["log_file"] = fname
    return df


def summarize(df: pd.DataFrame) -> dict:
    accuracy = df["correct"].mean()
    mean_flops = df["estimated_flops"].mean() if "estimated_flops" in df else None
    # kept for reference/debugging, no longer the primary compute axis
    avg_output_tokens = df["output_tokens_total"].mean() if "output_tokens_total" in df else None
    return {
        "n_examples": len(df),
        "accuracy": round(float(accuracy), 4),
        "mean_estimated_flops": float(mean_flops) if mean_flops is not None else None,
        "avg_output_tokens": round(float(avg_output_tokens), 1) if avg_output_tokens is not None else None,
    }


def plot_accuracy_vs_flops(summary_df: pd.DataFrame, path: str):
    """
    Labeled scatter, NOT connected lines — the strategies aren't points along
    a single path, so connecting them with a line falsely implies a
    trajectory between unrelated methods. Color = model/dataset combo,
    marker shape = strategy, matching the corrected plot style adopted after
    the earlier misleading connected-line version.
    """
    combos = summary_df[["model", "dataset"]].drop_duplicates().values.tolist()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    colors = {f"{m}/{d}": palette[i % len(palette)] for i, (m, d) in enumerate(combos)}

    strategies = sorted(summary_df["strategy"].unique())
    marker_cycle = ["o", "s", "^", "D", "P", "X", "*", "v", "<", ">"]
    markers = {s: marker_cycle[i % len(marker_cycle)] for i, s in enumerate(strategies)}

    fig, ax = plt.subplots(figsize=(12, 7.5))
    texts = []
    for model, dataset in combos:
        key = f"{model}/{dataset}"
        sub = summary_df[(summary_df.model == model) & (summary_df.dataset == dataset)]
        for _, row in sub.iterrows():
            if pd.isna(row["mean_estimated_flops"]):
                continue
            ax.scatter(row["mean_estimated_flops"], row["accuracy"],
                       color=colors[key], marker=markers[row["strategy"]],
                       s=140, edgecolors="black", linewidths=0.6, zorder=3)
            texts.append(ax.text(row["mean_estimated_flops"], row["accuracy"], row["strategy"], fontsize=7.5))

    ax.set_xscale("log")  # FLOPs span orders of magnitude across 1B vs 3B models
    if _HAVE_ADJUST_TEXT and texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5), expand_points=(1.4, 1.6))

    color_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=k)
                      for k, c in colors.items()]
    marker_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="None", markersize=9, label=s)
                       for s, m in markers.items()]
    leg1 = ax.legend(handles=color_handles, title="Model / Dataset", loc="upper left", fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, title="Strategy", loc="lower right", fontsize=8, title_fontsize=9)

    ax.set_xlabel("Estimated FLOPs per problem (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Math Track — Accuracy vs. Test-Time Compute (estimated FLOPs)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    logs_dir = args.logs_dir or cfg["paths"]["logs_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    log_paths = sorted(glob.glob(f"{logs_dir}/*.jsonl"))
    if not log_paths:
        print(f"No logs found in {logs_dir}. Run the strategy scripts first.")
        return

    summary_rows = []
    for path in log_paths:
        df = load_log(path)
        stem = os.path.basename(path).replace(".jsonl", "")
        parts = stem.split("_")
        dataset = parts[-1]
        model = parts[-2]
        strategy = "_".join(parts[:-2])
        stats = summarize(df)
        summary_rows.append({"strategy": strategy, "model": model, "dataset": dataset, **stats})

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "model", "strategy"])
    csv_path = f"{results_dir}/summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSummary table written to {csv_path}")

    if summary_df["mean_estimated_flops"].notna().any():
        plot_path = f"{results_dir}/performance_vs_flops.png"
        plot_accuracy_vs_flops(summary_df, plot_path)
        print(f"Plot written to {plot_path}")
    else:
        print("No logs have estimated_flops yet — re-run strategy scripts with the updated code to populate it.")


if __name__ == "__main__":
    main()
