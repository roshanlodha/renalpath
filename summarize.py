import argparse
import json
import os
from datetime import datetime

import pandas as pd


def _safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_summaries(analysis_dir: str):
    rows = []
    summaries = {}

    if not os.path.isdir(analysis_dir):
        raise FileNotFoundError(f"analysis_dir not found: {analysis_dir}")

    for entry in sorted(os.listdir(analysis_dir)):
        model_dir = os.path.join(analysis_dir, entry)
        if not os.path.isdir(model_dir):
            continue

        summary_path = os.path.join(model_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        summaries[entry] = summary
        metrics = summary.get("metrics", {})

        rows.append(
            {
                "model": entry,
                "num_classes": summary.get("num_classes"),
                "accuracy": _safe_float(metrics.get("accuracy")),
                "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")),
                "f1_macro": _safe_float(metrics.get("f1_macro")),
                "mcc": _safe_float(metrics.get("mcc")),
                "auprc_macro": _safe_float(metrics.get("auprc_macro")),
                "auroc_macro": _safe_float(metrics.get("auroc_macro")),
                "binary_roc_auc": _safe_float(metrics.get("binary_roc_auc")),
                "summary_path": os.path.relpath(summary_path, start=analysis_dir),
            }
        )

    return summaries, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate model evaluation summaries")
    parser.add_argument("--analysis_dir", type=str, default="analysis", help="Directory containing per-model analysis folders")
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join("analysis", "summaries.json"),
        help="Where to write aggregated JSON",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join("analysis", "summaries.csv"),
        help="Where to write aggregated CSV",
    )
    args = parser.parse_args()

    summaries, df = load_summaries(args.analysis_dir)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    aggregated = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dir": os.path.abspath(args.analysis_dir),
        "models_found": sorted(list(summaries.keys())),
        "summaries": summaries,
    }

    with open(args.output_json, "w") as f:
        json.dump(aggregated, f, indent=2, sort_keys=True)

    if not df.empty:
        # Sort by balanced accuracy primarily (works for both binary/multiclass)
        df_sorted = df.sort_values(by=["balanced_accuracy", "f1_macro"], ascending=[False, False])
        df_sorted.to_csv(args.output_csv, index=False)

        # Console-friendly view
        cols = [
            "model",
            "num_classes",
            "balanced_accuracy",
            "accuracy",
            "f1_macro",
            "mcc",
            "auroc_macro",
            "auprc_macro",
            "binary_roc_auc",
        ]
        print("\nAggregated model comparison (sorted by balanced_accuracy):")
        print(df_sorted[cols].to_string(index=False))
    else:
        print(f"No summaries found under {args.analysis_dir}. Expected files like analysis/<model>/summary.json")

    print(f"\nWrote aggregated JSON: {args.output_json}")
    print(f"Wrote aggregated CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
