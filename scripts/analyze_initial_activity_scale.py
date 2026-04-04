#!/usr/bin/env python

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

import random_mask_utils  # Registers RandomMaskConnectome.
from degree_preserving_mask_utils import (
    DEFAULT_DEGREE_PRESERVING_MASK_PATH,
    init_degree_preserving_random_network,
)
from revision_control_utils import (
    build_stage3_train_payload,
    gradient_norm,
    init_connectome_network_from_scratch,
    init_decoder_with_seed,
    model_summary,
    sync_if_cuda,
    write_markdown,
)


DEFAULT_RANDOM_MASK_PATH = Path("results/main_results/random_mask_selfloop.pt")
DEFAULT_OUTPUT_ROOT = Path("results/revision_results/revision_initial_activity")
DEFAULT_REPORT_PATH = Path("docs/generated/revision_initial_activity.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure initial activity scale under the fair from-scratch initialization route."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--random-mask-path", type=Path, default=DEFAULT_RANDOM_MASK_PATH)
    parser.add_argument(
        "--degree-mask-path",
        type=Path,
        default=DEFAULT_DEGREE_PRESERVING_MASK_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--train-speeds", nargs="+", type=float, default=[2.4])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[19.0])
    return parser.parse_args()


def init_random_factory(mask_path: Path):
    from revision_control_utils import base_network_config
    from flyvis.network.network import Network

    def factory(seed: int):
        config = base_network_config(seed)
        config.network.connectome.type = "RandomMaskConnectome"
        config.network.connectome.mask_path = str(mask_path)
        for key in ["file", "extent", "n_syn_fill"]:
            if key in config.network.connectome:
                del config.network.connectome[key]
        return Network(**config.network)

    return factory


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {args.output_root}")
    if args.report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.report_path}")
    if not args.random_mask_path.exists():
        raise FileNotFoundError(f"Random self-loop-matched mask not found: {args.random_mask_path}")
    if not args.degree_mask_path.exists():
        raise FileNotFoundError(f"Degree-preserving mask not found: {args.degree_mask_path}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    payload = build_stage3_train_payload(
        train_speeds=args.train_speeds,
        test_speeds=args.test_speeds,
    )
    stimuli = payload["stimuli"]
    targets = payload["targets"]
    feature_start = payload["feature_start"]
    feature_stop = payload["feature_stop"]
    generalization_config = payload["generalization_config"]

    from stage34_movingedge_utils import pooled_decoder_features, run_network_batch, set_global_seed

    factories = {
        "connectome": init_connectome_network_from_scratch,
        "random": init_random_factory(args.random_mask_path),
        "degreepres": lambda seed: init_degree_preserving_random_network(
            mask_path=args.degree_mask_path,
            init_seed=seed,
        ),
    }

    rows = []
    for seed in args.seeds:
        for model_kind, network_factory in factories.items():
            set_global_seed(seed)
            network = network_factory(seed)
            decoder = init_decoder_with_seed(network, seed)
            network.train()
            decoder.train()

            for parameter in list(network.parameters()) + list(decoder.parameters()):
                if parameter.grad is not None:
                    parameter.grad = None

            sync_if_cuda()
            activity = run_network_batch(
                network=network,
                stimuli=stimuli,
                dt=generalization_config.dt,
                steady_state_seconds=generalization_config.steady_state_seconds,
            )
            pooled, _ = pooled_decoder_features(
                activity=activity,
                network=network,
                start=feature_start,
                stop=feature_stop,
            )
            prediction = decoder(pooled)
            task_loss = F.mse_loss(prediction, targets)
            task_loss.backward()
            sync_if_cuda()

            node_mean = activity.mean(dim=(0, 1))
            grad_norm_value = gradient_norm(list(network.parameters()) + list(decoder.parameters()))
            summary = model_summary(model_kind, network)
            rows.append(
                {
                    "seed": seed,
                    "model_kind": model_kind,
                    "mask_type": summary["mask_type"],
                    "n_nodes": summary["n_nodes"],
                    "n_edges": summary["n_edges"],
                    "self_loops": summary["self_loops"],
                    "free_parameters": summary["free_parameters"],
                    "fixed_parameters": summary["fixed_parameters"],
                    "task_loss": float(task_loss.detach().cpu()),
                    "mean_abs_activity": float(activity.abs().mean().detach().cpu()),
                    "total_abs_activity": float(activity.abs().sum().detach().cpu()),
                    "node_variance": float(node_mean.var(unbiased=False).detach().cpu()),
                    "gradient_norm": safe_float(grad_norm_value) if grad_norm_value is not None else None,
                    "finite": bool(
                        math.isfinite(float(task_loss.detach().cpu()))
                        and math.isfinite(float(activity.abs().mean().detach().cpu()))
                        and (grad_norm_value is None or math.isfinite(float(grad_norm_value)))
                    ),
                }
            )
            print(
                f"seed={seed} model={model_kind} "
                f"mean_abs_activity={rows[-1]['mean_abs_activity']:.6f} "
                f"total_abs_activity={rows[-1]['total_abs_activity']:.2f} "
                f"node_variance={rows[-1]['node_variance']:.6f} "
                f"grad_norm={rows[-1]['gradient_norm']}"
            )

    df = pd.DataFrame(rows).sort_values(["model_kind", "seed"])
    metrics_path = args.output_root / "initial_activity_metrics.csv"
    df.to_csv(metrics_path, index=False)

    summary_rows = (
        df.groupby("model_kind", as_index=False)[
            ["task_loss", "mean_abs_activity", "total_abs_activity", "node_variance", "gradient_norm"]
        ]
        .agg(["mean", "std"])
    )
    summary_rows.columns = ["_".join(col).strip("_") for col in summary_rows.columns]
    summary_rows = summary_rows.rename(columns={"model_kind_": "model_kind"})

    def metric_mean(model: str, column: str) -> float:
        value = summary_rows.loc[summary_rows["model_kind"].eq(model), column]
        return float(value.iloc[0])

    conn_mean = metric_mean("connectome", "mean_abs_activity_mean")
    deg_mean = metric_mean("degreepres", "mean_abs_activity_mean")
    rel_diff = abs(deg_mean - conn_mean) / max(conn_mean, 1e-12)
    grad_conn = metric_mean("connectome", "gradient_norm_mean")
    grad_deg = metric_mean("degreepres", "gradient_norm_mean")
    grad_ratio = max(grad_conn, grad_deg) / max(min(grad_conn, grad_deg), 1e-12)

    if rel_diff > 0.25 or grad_ratio > 2.0:
        severity = "large"
        recommendation = (
            "Initial activity or gradient scale mismatch looks large enough that a minimal "
            "calibration diagnostic would be justified before making stronger claims."
        )
    elif rel_diff > 0.10 or grad_ratio > 1.5:
        severity = "moderate"
        recommendation = (
            "Initial activity differences are present but not extreme. A calibration test "
            "does not appear mandatory from this diagnostic alone."
        )
    else:
        severity = "small"
        recommendation = (
            "Connectome and degree-preserving random start in broadly comparable dynamical regimes "
            "under the fair initialization route. A calibration follow-up is not currently justified."
        )

    lines = [
        "# Initial Activity / Dynamical-Scale Check",
        "",
        "This report measures pre-training forward-pass activity under the same fair from-scratch "
        "initialization route used in the corrected experiments. The goal is to test whether the "
        "revised conclusion could still be explained by a trivial initial-scale mismatch.",
        "",
        f"- Seeds: {args.seeds}",
        f"- Canonical batch shape: {tuple(stimuli.shape)}",
        f"- Random mask path: {args.random_mask_path}",
        f"- Degree-preserving mask path: {args.degree_mask_path}",
        "",
        "| Model | Mean abs activity | Total abs activity | Node variance | Gradient norm |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in summary_rows.iterrows():
        lines.append(
            f"| {row['model_kind']} | "
            f"{row['mean_abs_activity_mean']:.6f} +/- {row['mean_abs_activity_std']:.6f} | "
            f"{row['total_abs_activity_mean']:.2f} +/- {row['total_abs_activity_std']:.2f} | "
            f"{row['node_variance_mean']:.6f} +/- {row['node_variance_std']:.6f} | "
            f"{row['gradient_norm_mean']:.6f} +/- {row['gradient_norm_std']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Relative mean-activity difference (connectome vs degree-preserving): {rel_diff:.2%}",
            f"- Gradient-norm ratio (larger/smaller, connectome vs degree-preserving): {grad_ratio:.3f}",
            f"- Qualitative severity: {severity}",
            f"- Read: {recommendation}",
        ]
    )
    write_markdown(args.report_path, lines)
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
