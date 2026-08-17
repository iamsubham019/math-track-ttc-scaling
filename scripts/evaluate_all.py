"""
Aggregate all logs/*.jsonl into one results table and a performance-vs-FLOPs
plot, ready for the paper's math-track results section.

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

from src.model_utils import load_config


def load_log(path: str) -> pd.DataFrame:
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    fname = os.path.basename(path).replace(".jsonl", "")
    # filename convention: <strategy>_<model>_<dataset>.jsonl
    parts = fname.split("_")
    df["log_file"] = fname
    return df


def summarize(df: pd.DataFrame) -> dict:
    accuracy = df["correct"].mean()
    # avg output tokens as compute proxy; swap in a param-count-based FLOPs calc if needed
    avg_output_tokens = df["output_tokens_total"].mean() if "output_tokens_total" in df else None
    return {
        "n_examples": len(df),
        "accuracy": round(float(accuracy), 4),
        "avg_output_tokens": round(float(avg_output_tokens), 1) if avg_output_tokens else None,
    }


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
        # strategy is everything before the last two underscore-separated fields (model, dataset)
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

    # Performance-vs-FLOPs (proxy: avg output tokens) plot, one line per (model, dataset)
    if "avg_output_tokens" in summary_df and summary_df["avg_output_tokens"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 5))
        for (model, dataset), group in summary_df.groupby(["model", "dataset"]):
            group = group.sort_values("avg_output_tokens")
            ax.plot(group["avg_output_tokens"], group["accuracy"], marker="o", label=f"{model}/{dataset}")
            for _, row in group.iterrows():
                ax.annotate(row["strategy"], (row["avg_output_tokens"], row["accuracy"]), fontsize=7)
        ax.set_xlabel("Avg. output tokens per problem (compute proxy)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Math Track — Performance vs. Test-Time Compute")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = f"{results_dir}/performance_vs_flops.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Plot written to {plot_path}")


if __name__ == "__main__":
    main()
