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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage3_train_connectome_vs_random_matched_steps import init_model
from stage34_movingedge_utils import BASELINE_CHECKPOINT


DEFAULT_INPUT_ROOT = Path("results/main_results/stage3_connectome_vs_random_matched_steps_saved")
DEFAULT_OUTPUT_ROOT = Path("results/main_results/mechanism")
DEFAULT_RANDOM_MASK_PATH = Path("results/main_results/random_mask_selfloop.pt")
SEEDS = (0, 1, 2)
MODELS = ("connectome", "random")
TOP_FRACS = (0.01, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze edge usage concentration.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--random-mask-path", type=Path, default=DEFAULT_RANDOM_MASK_PATH)
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


def effective_edge_weight(network) -> torch.Tensor:
    sign = network.edges_sign.detach().cpu()[network.edge_params.sign.indices.cpu()]
    syn_count = network.edges_syn_count.detach().cpu()[network.edge_params.syn_count.indices.cpu()]
    syn_strength = network.edges_syn_strength.detach().cpu()[
        network.edge_params.syn_strength.indices.cpu()
    ]
    return sign * syn_count * syn_strength


def plot_usage_curves(per_edge_frames: list[pd.DataFrame], output_root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model_kind, color in zip(
        axes,
        ("connectome", "random"),
        ("tab:blue", "tab:orange"),
    ):
        for frame in per_edge_frames:
            if frame["model_kind"].iloc[0] != model_kind:
                continue
            values = frame["edge_usage"].to_numpy()
            sorted_desc = np.sort(values)[::-1]
            ax.plot(sorted_desc, alpha=0.75, linewidth=1.0)
        ax.set_title(model_kind)
        ax.set_xlabel("Edges sorted by usage")
        ax.set_yscale("log")
    axes[0].set_ylabel("Edge usage proxy")
    fig.suptitle("Sorted Edge Usage Curves")
    fig.tight_layout()
    fig.savefig(output_root / "edge_usage_sorted.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model_kind in zip(axes, ("connectome", "random")):
        for frame in per_edge_frames:
            if frame["model_kind"].iloc[0] != model_kind:
                continue
            x, y = cumulative_curve(frame["edge_usage"].to_numpy())
            ax.plot(x, y, alpha=0.75, linewidth=1.0)
        ax.set_title(model_kind)
        ax.set_xlabel("Top fraction of edges")
    axes[0].set_ylabel("Cumulative usage contribution")
    fig.suptitle("Edge Usage Concentration")
    fig.tight_layout()
    fig.savefig(output_root / "edge_usage_cumulative.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if not output_root.exists():
        raise FileNotFoundError(
            f"Mechanism output directory missing. Run node analysis first: {output_root}"
        )
    metrics_path = output_root / "edge_usage_metrics.csv"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {metrics_path}")

    metric_rows: list[dict[str, float | int | str]] = []
    per_edge_frames: list[pd.DataFrame] = []

    for seed in SEEDS:
        seed_root = args.input_root / f"seed_{seed}"
        for model_kind in MODELS:
            activity_path = seed_root / f"{model_kind}_activity.pt"
            network_path = seed_root / f"{model_kind}_network.pt"
            require_file(activity_path)
            require_file(network_path)

            network, _decoder, _optimizer = init_model(
                model_kind=model_kind,
                baseline_checkpoint=args.baseline_checkpoint,
                random_mask_path=args.random_mask_path,
                learning_rate=1e-3,
            )
            state_dict = torch.load(network_path, map_location="cpu", weights_only=False)
            network.load_state_dict(state_dict)

            activity = torch.load(activity_path, map_location="cpu", weights_only=False)
            source_mean_abs = activity.abs().mean(dim=(0, 1)).cpu()
            source_index = torch.as_tensor(network.connectome.edges.source_index[:], dtype=torch.long)
            edge_weights = effective_edge_weight(network)
            edge_usage = (edge_weights.abs() * source_mean_abs[source_index]).numpy().astype(np.float64)

            edge_frame = pd.DataFrame(
                {
                    "seed": seed,
                    "model_kind": model_kind,
                    "edge_index": np.arange(edge_usage.shape[0], dtype=np.int64),
                    "edge_usage": edge_usage,
                }
            )
            per_edge_frames.append(edge_frame)

            row: dict[str, float | int | str] = {
                "seed": seed,
                "model_kind": model_kind,
                "n_edges": int(edge_usage.shape[0]),
                "total_usage": float(edge_usage.sum()),
                "mean_usage": float(edge_usage.mean()),
                "std_usage": float(edge_usage.std()),
                "edge_usage_gini": gini(edge_usage),
            }
            for frac in TOP_FRACS:
                row[f"edge_top{int(frac * 100)}_frac"] = top_fraction(edge_usage, frac)
            metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["seed", "model_kind"]).reset_index(drop=True)
    metrics_df.to_csv(metrics_path, index=False)
    plot_usage_curves(per_edge_frames, output_root)


if __name__ == "__main__":
    main()
