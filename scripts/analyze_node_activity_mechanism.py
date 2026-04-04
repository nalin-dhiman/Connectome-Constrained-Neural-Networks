#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


DEFAULT_INPUT_ROOT = Path("results/main_results/stage3_connectome_vs_random_matched_steps_saved")
DEFAULT_OUTPUT_ROOT = Path("results/main_results/mechanism")
SEEDS = (0, 1, 2)
MODELS = ("connectome", "random")
TOP_FRACS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze node activity concentration.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    if np.any(values < 0):
        raise ValueError("Gini expects non-negative values.")
    total = values.sum()
    if total <= 0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - (n + 1) / n)


def top_fraction(values: np.ndarray, frac: float) -> float:
    n = values.size
    k = max(1, int(math.ceil(frac * n)))
    sorted_values = np.sort(values)
    top_sum = float(sorted_values[-k:].sum())
    total = float(sorted_values.sum())
    return 0.0 if total <= 0 else top_sum / total


def cumulative_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_desc = np.sort(values)[::-1]
    total = sorted_desc.sum()
    x = np.arange(1, len(sorted_desc) + 1, dtype=np.float64) / len(sorted_desc)
    if total <= 0:
        y = np.zeros_like(x)
    else:
        y = np.cumsum(sorted_desc) / total
    return x, y


def plot_histogram(metrics_df: pd.DataFrame, plot_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for model_kind, color in (("connectome", "tab:blue"), ("random", "tab:orange")):
        subset = metrics_df.loc[metrics_df["model_kind"].eq(model_kind), "mean_abs_activity"]
        plt.hist(
            subset.to_numpy(),
            bins=50,
            density=True,
            alpha=0.45,
            label=model_kind,
            color=color,
        )
    plt.xlabel("Per-node mean absolute activity")
    plt.ylabel("Density")
    plt.title("Node Activity Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def plot_cumulative_curves(per_node_frames: list[pd.DataFrame], plot_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model_kind, color in zip(
        axes,
        ("connectome", "random"),
        ("tab:blue", "tab:orange"),
    ):
        for seed in SEEDS:
            frame = next(
                item for item in per_node_frames if int(item["seed"].iloc[0]) == seed and item["model_kind"].iloc[0] == model_kind
            )
            x, y = cumulative_curve(frame["mean_abs_activity"].to_numpy())
            ax.plot(x, y, alpha=0.8, label=f"seed {seed}")
        ax.set_title(model_kind)
        ax.set_xlabel("Top fraction of nodes")
        ax.legend()
    axes[0].set_ylabel("Cumulative activity contribution")
    fig.suptitle("Node Activity Concentration")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_root}")

    args.output_root.mkdir(parents=True, exist_ok=False)

    metric_rows: list[dict[str, float | int | str]] = []
    per_node_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        seed_root = args.input_root / f"seed_{seed}"
        for model_kind in MODELS:
            activity_path = seed_root / f"{model_kind}_activity.pt"
            require_file(activity_path)
            activity = torch.load(activity_path, map_location="cpu", weights_only=False)
            if activity.ndim != 3:
                raise RuntimeError(
                    f"Unexpected activity tensor rank for {activity_path}: {activity.ndim}"
                )
            mean_abs = activity.abs().mean(dim=(0, 1)).cpu().numpy().astype(np.float64)
            node_indices = np.arange(mean_abs.shape[0], dtype=np.int64)
            frame = pd.DataFrame(
                {
                    "seed": seed,
                    "model_kind": model_kind,
                    "node_index": node_indices,
                    "mean_abs_activity": mean_abs,
                }
            )
            frame.to_csv(
                args.output_root / f"node_activity_seed{seed}_{model_kind}.csv",
                index=False,
            )
            per_node_frames.append(frame)

            row: dict[str, float | int | str] = {
                "seed": seed,
                "model_kind": model_kind,
                "n_nodes": int(mean_abs.shape[0]),
                "total_activity": float(mean_abs.sum()),
                "mean_abs_activity": float(mean_abs.mean()),
                "std_abs_activity": float(mean_abs.std()),
                "node_activity_gini": gini(mean_abs),
            }
            for frac in TOP_FRACS:
                row[f"node_top{int(frac * 100)}_frac"] = top_fraction(mean_abs, frac)
            metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["seed", "model_kind"]).reset_index(drop=True)
    metrics_df.to_csv(args.output_root / "node_activity_metrics.csv", index=False)

    all_nodes_df = pd.concat(per_node_frames, ignore_index=True)
    plot_histogram(all_nodes_df, args.output_root / "node_activity_histogram.png")
    plot_cumulative_curves(per_node_frames, args.output_root / "node_activity_cumulative.png")


if __name__ == "__main__":
    main()
