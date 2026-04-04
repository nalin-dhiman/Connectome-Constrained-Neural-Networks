#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


DEFAULT_INPUT_ROOT = Path("results/main_results/stage3_connectome_vs_random_matched_steps_saved")
DEFAULT_OUTPUT_ROOT = Path("results/main_results/mechanism")
SEEDS = (0, 1, 2)
MODELS = ("connectome", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze temporal activity stability.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if not output_root.exists():
        raise FileNotFoundError(
            f"Mechanism output directory missing. Run node analysis first: {output_root}"
        )
    metrics_path = output_root / "temporal_stability_metrics.csv"
    trajectory_path = output_root / "temporal_stability_trajectories.csv"
    if metrics_path.exists() or trajectory_path.exists():
        raise FileExistsError("Refusing to overwrite existing temporal outputs.")

    metric_rows: list[dict[str, float | int | str]] = []
    trajectory_rows: list[dict[str, float | int | str]] = []

    for seed in SEEDS:
        seed_root = args.input_root / f"seed_{seed}"
        for model_kind in MODELS:
            activity_path = seed_root / f"{model_kind}_activity.pt"
            require_file(activity_path)
            activity = torch.load(activity_path, map_location="cpu", weights_only=False)

            mean_batch = activity.mean(dim=0)
            abs_mean_batch = activity.abs().mean(dim=0)

            total_abs_t = abs_mean_batch.sum(dim=1).cpu().numpy().astype(np.float64)
            node_var_t = mean_batch.var(dim=1, unbiased=False).cpu().numpy().astype(np.float64)
            delta_t = (
                (activity[:, 1:] - activity[:, :-1]).abs().mean(dim=(0, 2)).cpu().numpy().astype(np.float64)
            )

            for timestep, (a_t, v_t) in enumerate(zip(total_abs_t, node_var_t)):
                trajectory_rows.append(
                    {
                        "seed": seed,
                        "model_kind": model_kind,
                        "timestep": timestep,
                        "total_abs_activity_t": float(a_t),
                        "node_variance_t": float(v_t),
                        "mean_abs_delta_t": float(delta_t[timestep - 1]) if timestep > 0 else np.nan,
                    }
                )

            metric_rows.append(
                {
                    "seed": seed,
                    "model_kind": model_kind,
                    "mean_total_activity_over_time": float(total_abs_t.mean()),
                    "std_total_activity_over_time": float(total_abs_t.std()),
                    "mean_node_variance_over_time": float(node_var_t.mean()),
                    "mean_temporal_variation": float(delta_t.mean()),
                    "max_total_activity_t": float(total_abs_t.max()),
                    "min_total_activity_t": float(total_abs_t.min()),
                }
            )

    metrics_df = pd.DataFrame(metric_rows).sort_values(["seed", "model_kind"]).reset_index(drop=True)
    trajectories_df = pd.DataFrame(trajectory_rows).sort_values(["seed", "model_kind", "timestep"]).reset_index(drop=True)
    metrics_df.to_csv(metrics_path, index=False)
    trajectories_df.to_csv(trajectory_path, index=False)

    avg_traj = trajectories_df.groupby(["model_kind", "timestep"], as_index=False).agg(
        total_abs_activity_t=("total_abs_activity_t", "mean"),
        node_variance_t=("node_variance_t", "mean"),
        mean_abs_delta_t=("mean_abs_delta_t", "mean"),
    )

    plt.figure(figsize=(8, 4.5))
    for model_kind, color in (("connectome", "tab:blue"), ("random", "tab:orange")):
        subset = avg_traj.loc[avg_traj["model_kind"].eq(model_kind)]
        plt.plot(subset["timestep"], subset["total_abs_activity_t"], label=model_kind, color=color)
    plt.xlabel("Timestep")
    plt.ylabel("Total absolute activity")
    plt.title("Temporal Activity Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "temporal_total_activity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for model_kind, color in (("connectome", "tab:blue"), ("random", "tab:orange")):
        subset = avg_traj.loc[avg_traj["model_kind"].eq(model_kind)]
        plt.plot(subset["timestep"], subset["node_variance_t"], label=model_kind, color=color)
    plt.xlabel("Timestep")
    plt.ylabel("Variance across nodes")
    plt.title("Temporal Node Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "temporal_node_variance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for model_kind, color in (("connectome", "tab:blue"), ("random", "tab:orange")):
        subset = avg_traj.loc[avg_traj["model_kind"].eq(model_kind) & avg_traj["timestep"].gt(0)]
        plt.plot(subset["timestep"], subset["mean_abs_delta_t"], label=model_kind, color=color)
    plt.xlabel("Timestep")
    plt.ylabel("Mean absolute activity change")
    plt.title("Temporal Activity Change")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_root / "temporal_activity_change.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
