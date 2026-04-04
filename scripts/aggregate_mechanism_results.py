#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("results/main_results/mechanism")
DEFAULT_REPORT_PATH = Path("docs/generated/stage3_mechanism_analysis.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mechanism analysis outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def summarize(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for model_kind, group in df.groupby("model_kind"):
        row = {"model_kind": model_kind}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_kind").reset_index(drop=True)


def delta_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    pivot = df.pivot(index="seed", columns="model_kind", values=metrics)
    rows = []
    for metric in metrics:
        delta = pivot[(metric, "connectome")] - pivot[(metric, "random")]
        rows.append(
            {
                "metric": metric,
                "connectome_minus_random_mean": float(delta.mean()),
                "connectome_minus_random_std": float(delta.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def mechanism_label(node_summary: pd.DataFrame, edge_summary: pd.DataFrame, temporal_summary: pd.DataFrame) -> str:
    node_c = node_summary.loc[node_summary["model_kind"].eq("connectome")].iloc[0]
    node_r = node_summary.loc[node_summary["model_kind"].eq("random")].iloc[0]
    edge_c = edge_summary.loc[edge_summary["model_kind"].eq("connectome")].iloc[0]
    edge_r = edge_summary.loc[edge_summary["model_kind"].eq("random")].iloc[0]
    temp_c = temporal_summary.loc[temporal_summary["model_kind"].eq("connectome")].iloc[0]
    temp_r = temporal_summary.loc[temporal_summary["model_kind"].eq("random")].iloc[0]

    lower_node_concentration = (
        node_c["node_activity_gini_mean"] < node_r["node_activity_gini_mean"]
        and node_c["node_top10_frac_mean"] < node_r["node_top10_frac_mean"]
    )
    lower_edge_concentration = (
        edge_c["edge_usage_gini_mean"] < edge_r["edge_usage_gini_mean"]
        and edge_c["edge_top10_frac_mean"] < edge_r["edge_top10_frac_mean"]
    )
    lower_temporal_variation = (
        temp_c["mean_temporal_variation_mean"] < temp_r["mean_temporal_variation_mean"]
        and temp_c["mean_total_activity_over_time_mean"] < temp_r["mean_total_activity_over_time_mean"]
    )
    higher_concentration = (
        node_c["node_activity_gini_mean"] > node_r["node_activity_gini_mean"]
        and edge_c["edge_usage_gini_mean"] > edge_r["edge_usage_gini_mean"]
    )

    if lower_node_concentration and lower_edge_concentration and lower_temporal_variation:
        return "distribution advantage"
    if higher_concentration and temp_c["mean_total_activity_over_time_mean"] < temp_r["mean_total_activity_over_time_mean"]:
        return "concentration advantage"
    if lower_temporal_variation:
        return "stability advantage"
    if lower_node_concentration or lower_edge_concentration or lower_temporal_variation:
        return "mixed mechanism"
    return "unclear"


def main() -> None:
    args = parse_args()
    if args.report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.report_path}")

    node_metrics_path = args.output_root / "node_activity_metrics.csv"
    edge_metrics_path = args.output_root / "edge_usage_metrics.csv"
    temporal_metrics_path = args.output_root / "temporal_stability_metrics.csv"
    for path in (node_metrics_path, edge_metrics_path, temporal_metrics_path):
        require_file(path)

    node_df = pd.read_csv(node_metrics_path)
    edge_df = pd.read_csv(edge_metrics_path)
    temporal_df = pd.read_csv(temporal_metrics_path)

    node_metrics = [
        "total_activity",
        "mean_abs_activity",
        "std_abs_activity",
        "node_activity_gini",
        "node_top1_frac",
        "node_top5_frac",
        "node_top10_frac",
    ]
    edge_metrics = [
        "total_usage",
        "mean_usage",
        "std_usage",
        "edge_usage_gini",
        "edge_top1_frac",
        "edge_top5_frac",
        "edge_top10_frac",
    ]
    temporal_metrics = [
        "mean_total_activity_over_time",
        "std_total_activity_over_time",
        "mean_node_variance_over_time",
        "mean_temporal_variation",
    ]

    node_summary = summarize(node_df, node_metrics)
    edge_summary = summarize(edge_df, edge_metrics)
    temporal_summary = summarize(temporal_df, temporal_metrics)
    node_summary.to_csv(args.output_root / "node_activity_summary.csv", index=False)
    edge_summary.to_csv(args.output_root / "edge_usage_summary.csv", index=False)
    temporal_summary.to_csv(args.output_root / "temporal_stability_summary.csv", index=False)

    delta_frames = [
        delta_table(node_df, ["node_activity_gini", "node_top1_frac", "node_top5_frac", "node_top10_frac"]),
        delta_table(edge_df, ["edge_usage_gini", "edge_top1_frac", "edge_top5_frac", "edge_top10_frac"]),
        delta_table(temporal_df, ["mean_total_activity_over_time", "mean_temporal_variation"]),
    ]
    delta_df = pd.concat(delta_frames, ignore_index=True)
    delta_df.to_csv(args.output_root / "mechanism_delta_summary.csv", index=False)

    label = mechanism_label(node_summary, edge_summary, temporal_summary)

    def fmt(summary: pd.DataFrame, metric: str, model_kind: str) -> str:
        row = summary.loc[summary["model_kind"].eq(model_kind)].iloc[0]
        return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"

    lines = [
        "# Stage 3 Mechanism Analysis",
        "",
        "Scope:",
        "- No retraining.",
        "- Uses saved 5-step matched-step artifacts only.",
        "- Seeds: 0, 1, 2.",
        "- Connectome vs random only.",
        "",
        "Node activity results:",
        "",
        "| Model | Gini | Top 1% frac | Top 5% frac | Top 10% frac | Mean abs activity |",
        "|---|---:|---:|---:|---:|---:|",
        f"| connectome | {fmt(node_summary, 'node_activity_gini', 'connectome')} | {fmt(node_summary, 'node_top1_frac', 'connectome')} | {fmt(node_summary, 'node_top5_frac', 'connectome')} | {fmt(node_summary, 'node_top10_frac', 'connectome')} | {fmt(node_summary, 'mean_abs_activity', 'connectome')} |",
        f"| random | {fmt(node_summary, 'node_activity_gini', 'random')} | {fmt(node_summary, 'node_top1_frac', 'random')} | {fmt(node_summary, 'node_top5_frac', 'random')} | {fmt(node_summary, 'node_top10_frac', 'random')} | {fmt(node_summary, 'mean_abs_activity', 'random')} |",
        "",
        "Edge usage results:",
        "",
        "| Model | Gini | Top 1% frac | Top 5% frac | Top 10% frac | Mean usage |",
        "|---|---:|---:|---:|---:|---:|",
        f"| connectome | {fmt(edge_summary, 'edge_usage_gini', 'connectome')} | {fmt(edge_summary, 'edge_top1_frac', 'connectome')} | {fmt(edge_summary, 'edge_top5_frac', 'connectome')} | {fmt(edge_summary, 'edge_top10_frac', 'connectome')} | {fmt(edge_summary, 'mean_usage', 'connectome')} |",
        f"| random | {fmt(edge_summary, 'edge_usage_gini', 'random')} | {fmt(edge_summary, 'edge_top1_frac', 'random')} | {fmt(edge_summary, 'edge_top5_frac', 'random')} | {fmt(edge_summary, 'edge_top10_frac', 'random')} | {fmt(edge_summary, 'mean_usage', 'random')} |",
        "",
        "Temporal stability results:",
        "",
        "| Model | Mean total activity over time | Mean node variance over time | Mean temporal variation |",
        "|---|---:|---:|---:|",
        f"| connectome | {fmt(temporal_summary, 'mean_total_activity_over_time', 'connectome')} | {fmt(temporal_summary, 'mean_node_variance_over_time', 'connectome')} | {fmt(temporal_summary, 'mean_temporal_variation', 'connectome')} |",
        f"| random | {fmt(temporal_summary, 'mean_total_activity_over_time', 'random')} | {fmt(temporal_summary, 'mean_node_variance_over_time', 'random')} | {fmt(temporal_summary, 'mean_temporal_variation', 'random')} |",
        "",
        "Candidate explanation:",
    ]

    if label == "distribution advantage":
        lines.extend(
            [
                "- Connectome activity is less concentrated across both nodes and edges while also maintaining lower overall activity and lower temporal variation.",
                "- This supports a more efficient distributed-computation explanation than a sparse critical-path explanation.",
            ]
        )
    elif label == "concentration advantage":
        lines.extend(
            [
                "- Connectome activity/usage is more concentrated while total activity remains lower.",
                "- This is consistent with more selective routing or compressed computation.",
            ]
        )
    elif label == "stability advantage":
        lines.extend(
            [
                "- The clearest difference is lower temporal variation and lower total activity over time.",
                "- This supports a smoother-dynamics explanation more than a concentration explanation.",
            ]
        )
    elif label == "mixed mechanism":
        lines.extend(
            [
                "- The connectome differs from the random graph in more than one way, but the metrics do not cleanly reduce to a single concentration or stability story.",
                "- The safest reading is that the advantage is mechanistically mixed.",
            ]
        )
    else:
        lines.extend(
            [
                "- The saved-state metrics do not separate cleanly enough to support a single mechanism explanation.",
            ]
        )

    lines.extend(["", f"Final one-line mechanism label: {label}."])
    args.report_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
