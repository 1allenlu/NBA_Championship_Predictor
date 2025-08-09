# src/plot_model_results.py
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def flatten_results(data: dict) -> pd.DataFrame:
    """
    Flattens your results JSON into a DataFrame with columns:
    model_key, accuracy, f1_score, precision_pos, recall_pos, timestamp
    Handles dict entries AND list entries (e.g., pytorch_roster runs).
    """
    rows = []
    for key, val in data.items():
        if isinstance(val, dict):
            rows.append({
                "model_key": key,
                "accuracy": float(val.get("accuracy", 0) or 0),
                "f1_score": float(val.get("f1_score", 0) or 0),
                "precision_pos": float(val.get("precision_pos", 0) or 0),
                "recall_pos": float(val.get("recall_pos", 0) or 0),
                "timestamp": val.get("timestamp", "")
            })
        elif isinstance(val, list):
            # e.g., key="pytorch_roster", val=[{run1}, {run2}, ...]
            for i, run in enumerate(val, start=1):
                rows.append({
                    "model_key": f"{key}#{i}",
                    "accuracy": float(run.get("accuracy", 0) or 0),
                    "f1_score": float(run.get("f1_score", 0) or 0),
                    "precision_pos": float(run.get("precision_pos", 0) or 0),
                    "recall_pos": float(run.get("recall_pos", 0) or 0),
                    "timestamp": run.get("timestamp", "")
                })
        else:
            # unknown structure; skip
            continue
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Plot model results from JSON.")
    parser.add_argument(
        "--file",
        default="results/result_part2.json",  # change to your actual filename if different
        help="Path to the results JSON (e.g., results/model_results.json or results/result_part2.json)"
    )
    parser.add_argument(
        "--metric",
        default="f1_score",
        choices=["f1_score", "accuracy", "precision_pos", "recall_pos"],
        help="Metric to sort and plot (default: f1_score)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="How many top rows to plot (default: 50)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, saves the plot to results/plots/"
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Could not find {args.file}")

    with open(args.file, "r") as f:
        data = json.load(f)

    df = flatten_results(data)
    if df.empty:
        print("No results found in the JSON.")
        return

    # Sort and trim
    metric = args.metric
    df_sorted = df.sort_values(by=metric, ascending=False).head(args.top).reset_index(drop=True)

    # Nice labels
    title = f"Model Results — sorted by {metric}"
    ylabel = "Models"
    xlabel = metric

    # Plot (horizontal bars so long names fit; highest at top)
    plt.figure(figsize=(10, max(4, 0.35 * len(df_sorted))))
    plt.barh(df_sorted["model_key"], df_sorted[metric])
    plt.gca().invert_yaxis()  # so the highest is first (top)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    # Save if requested
    if args.save:
        outdir = Path("results/plots")
        outdir.mkdir(parents=True, exist_ok=True)
        outname = f"plot_{Path(args.file).stem}_{metric}.png"
        outpath = outdir / outname
        plt.savefig(outpath, dpi=200)
        print(f"Saved plot → {outpath}")

    plt.show()

if __name__ == "__main__":
    main()