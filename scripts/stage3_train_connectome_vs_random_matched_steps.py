#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import pandas as pd
import torch
import torch.nn.functional as F

from random_mask_utils import init_random_mask_network
from stage34_movingedge_utils import (
    BASELINE_CHECKPOINT,
    MovingEdgeGeneralizationConfig,
    batch_config_dict,
    build_movingedge_train_test_split,
    direction_targets,
    ensure_dir,
    init_linear_decoder,
    init_network_from_baseline,
    pooled_decoder_features,
    run_network_batch,
    set_global_seed,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("results/main_results/stage3_connectome_vs_random_matched_steps")
DEFAULT_REPORT_PATH = Path("docs/generated/stage3_connectome_vs_random_matched_steps.md")
DEFAULT_RANDOM_MASK_PATH = Path("results/main_results/random_mask_selfloop.pt")
MODEL_KINDS = ("connectome", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-step Stage 3 connectome vs random-mask baseline comparison."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=BASELINE_CHECKPOINT,
    )
    parser.add_argument(
        "--random-mask-path",
        type=Path,
        default=DEFAULT_RANDOM_MASK_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument("--train-speeds", nargs="+", type=float, default=[2.4])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[19.0])
    return parser.parse_args()


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def finite_or_none(value: float) -> float | None:
    if math.isfinite(value):
        return value
    return None


def gradient_norm(parameters) -> float | None:
    total_sq = 0.0
    found = False
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        grad_norm = float(grad.detach().norm().cpu())
        if not math.isfinite(grad_norm):
            return None
        total_sq += grad_norm**2
        found = True
    if not found:
        return None
    return math.sqrt(total_sq)


def model_summary(model_kind: str, network) -> dict[str, Any]:
    source = network.connectome.edges.source_index[:]
    target = network.connectome.edges.target_index[:]
    self_loops = int((source == target).sum())
    return {
        "model_kind": model_kind,
        "mask_type": type(network.connectome).__name__,
        "n_nodes": int(network.n_nodes),
        "n_edges": int(network.n_edges),
        "self_loops": self_loops,
        "free_parameters": int(network.num_parameters.free),
        "fixed_parameters": int(network.num_parameters.fixed),
    }


def init_model(
    *,
    model_kind: str,
    baseline_checkpoint: Path,
    random_mask_path: Path,
    learning_rate: float,
):
    if model_kind == "connectome":
        network = init_network_from_baseline(baseline_checkpoint)
    elif model_kind == "random":
        network = init_random_mask_network(
            mask_path=random_mask_path,
            checkpoint_path=baseline_checkpoint,
        )
    else:
        raise KeyError(f"Unknown model kind: {model_kind}")

    decoder = init_linear_decoder(network)
    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
    )
    return network, decoder, optimizer


def run_fixed_steps(
    *,
    model_kind: str,
    seed: int,
    max_iters: int,
    baseline_checkpoint: Path,
    random_mask_path: Path,
    learning_rate: float,
    stimuli: torch.Tensor,
    targets: torch.Tensor,
    feature_start: int,
    feature_stop: int,
    dt: float,
    steady_state_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_global_seed(seed)
    network, decoder, optimizer = init_model(
        model_kind=model_kind,
        baseline_checkpoint=baseline_checkpoint,
        random_mask_path=random_mask_path,
        learning_rate=learning_rate,
    )
    network.train()
    decoder.train()
    summary = model_summary(model_kind, network)

    print(
        f"seed={seed} {model_kind}: nodes={summary['n_nodes']}, "
        f"edges={summary['n_edges']}, self_loops={summary['self_loops']}, "
        f"mask_type={summary['mask_type']}"
    )

    rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    remained_finite = True

    for iteration in range(1, max_iters + 1):
        optimizer.zero_grad()

        sync_if_cuda()
        iter_start = time.perf_counter()
        activity = run_network_batch(
            network=network,
            stimuli=stimuli,
            dt=dt,
            steady_state_seconds=steady_state_seconds,
        )
        pooled, _ = pooled_decoder_features(
            activity=activity,
            network=network,
            start=feature_start,
            stop=feature_stop,
        )
        prediction = decoder(pooled)
        task_loss = F.mse_loss(prediction, targets)
        activity_penalty = activity.abs().mean()
        sync_if_cuda()
        forward_end = time.perf_counter()

        task_loss.backward()
        sync_if_cuda()
        backward_end = time.perf_counter()

        grad_norm_value = gradient_norm(
            list(network.parameters()) + list(decoder.parameters())
        )

        optimizer.step()
        network.clamp()
        sync_if_cuda()
        iter_end = time.perf_counter()

        task_loss_value = float(task_loss.detach().cpu())
        activity_value = float(activity_penalty.detach().cpu())
        finite = math.isfinite(task_loss_value) and math.isfinite(activity_value)
        if grad_norm_value is not None:
            finite = finite and math.isfinite(grad_norm_value)
        remained_finite = remained_finite and finite

        rows.append(
            {
                "model_kind": model_kind,
                "seed": seed,
                "iteration": iteration,
                "elapsed_sec": iter_end - run_start,
                "task_loss": task_loss_value,
                "activity_abs_mean": activity_value,
                "gradient_norm": finite_or_none(grad_norm_value)
                if grad_norm_value is not None
                else None,
                "forward_sec": forward_end - iter_start,
                "backward_plus_step_sec": iter_end - forward_end,
                "iter_total_sec": iter_end - iter_start,
                "finite": finite,
            }
        )

        if not finite:
            break

    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        raise RuntimeError(f"{model_kind} completed zero iterations for seed {seed}.")

    return curve_df, {
        **summary,
        "seed": seed,
        "iterations_requested": int(max_iters),
        "iterations_completed": int(len(curve_df)),
        "total_elapsed_sec": float(curve_df["elapsed_sec"].iloc[-1]),
        "mean_iter_total_sec": float(curve_df["iter_total_sec"].mean()),
        "remained_finite": bool(remained_finite),
        "final_task_loss": float(curve_df["task_loss"].iloc[-1]),
        "final_activity_abs_mean": float(curve_df["activity_abs_mean"].iloc[-1]),
    }


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> str:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return "n/a"
    return f"{float(row.iloc[0][column]):.6f}"


def conclusion_from_seed_summaries(seed_summaries: list[dict[str, Any]]) -> str:
    if any(
        (not item["connectome"]["remained_finite"]) or (not item["random"]["remained_finite"])
        for item in seed_summaries
    ):
        return "inconclusive"

    connectome_better_loss = 0
    connectome_better_activity = 0
    for item in seed_summaries:
        if item["connectome"]["final_task_loss"] < item["random"]["final_task_loss"]:
            connectome_better_loss += 1
        if (
            item["connectome"]["final_activity_abs_mean"]
            < item["random"]["final_activity_abs_mean"]
        ):
            connectome_better_activity += 1

    if connectome_better_loss == len(seed_summaries) and connectome_better_activity == len(
        seed_summaries
    ):
        return "optimization advantage"
    if connectome_better_loss < len(seed_summaries) and connectome_better_activity == len(
        seed_summaries
    ):
        return "activation advantage only"
    if connectome_better_loss == 0 and connectome_better_activity == 0:
        return "inconclusive"
    return "unstable"


def write_report(
    *,
    report_path: Path,
    seed_summaries: list[dict[str, Any]],
    conclusion: str,
) -> None:
    if not seed_summaries:
        raise RuntimeError("No seed summaries to report.")

    first = seed_summaries[0]
    lines = [
        "# Stage 3 Connectome vs Random Matched Steps",
        "",
        "Scope:",
        "- Matched-step comparison, not a benchmark.",
        "- Same canonical Stage 3 setup.",
        "- Seeds: 0, 1, 2.",
        "- Fixed 5 iterations for both models.",
        "- No wall-clock stopping.",
        "",
        "Model table:",
        "",
        "| Model | Nodes | Edges | Self-loops | Free params | Fixed params |",
        "|---|---:|---:|---:|---:|---:|",
        f"| connectome | {first['connectome']['n_nodes']} | {first['connectome']['n_edges']} | {first['connectome']['self_loops']} | {first['connectome']['free_parameters']} | {first['connectome']['fixed_parameters']} |",
        f"| random | {first['random']['n_nodes']} | {first['random']['n_edges']} | {first['random']['self_loops']} | {first['random']['free_parameters']} | {first['random']['fixed_parameters']} |",
        "",
        "Per-seed completion:",
    ]
    for item in seed_summaries:
        lines.extend(
            [
                f"- seed {item['seed']}: connectome iters={item['connectome']['iterations_completed']}, random iters={item['random']['iterations_completed']}, connectome finite={item['connectome']['remained_finite']}, random finite={item['random']['remained_finite']}",
            ]
        )
    lines.extend(
        [
            "",
            "This report contains only per-seed summaries. Aggregated tables are written by the separate aggregation script.",
            "",
            f"Conclusion label: {conclusion}.",
        ]
    )
    report_path.write_text("\n".join(lines))


def run_all_seeds(args: argparse.Namespace) -> None:
    if args.output_root.exists() or args.report_path.exists():
        raise FileExistsError("Refusing to overwrite existing matched-step outputs.")
    if not args.random_mask_path.exists():
        raise FileNotFoundError(f"Random mask not found: {args.random_mask_path}")
    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {args.baseline_checkpoint}"
        )

    output_root = ensure_dir(args.output_root)
    generalization_config = MovingEdgeGeneralizationConfig(
        train_speed_values=tuple(args.train_speeds),
        test_speed_values=tuple(args.test_speeds),
    )
    _dataset, stimuli_splits, metadata_splits, feature_slices = build_movingedge_train_test_split(
        generalization_config
    )
    train_stimuli = stimuli_splits["train"]
    train_targets = direction_targets(metadata_splits["train"])
    feature_start, feature_stop = feature_slices["train"]

    batch_info = {
        "train_input_shape": list(train_stimuli.shape),
        "train_batch_size": int(train_stimuli.shape[0]),
        "train_frames": int(train_stimuli.shape[1]),
        "train_angles": sorted(metadata_splits["train"]["angle"].unique().tolist()),
        "train_speeds": sorted(metadata_splits["train"]["speed"].unique().tolist()),
        "task_definition": (
            "Canonical Stage 3 task: linear decoder predicts 2D edge direction "
            "(cos(theta), sin(theta)) from mean central-cell activity during the stimulus window."
        ),
        "generalization_config": batch_config_dict(generalization_config),
        "max_iters": int(args.max_iters),
    }
    print(f"Canonical train input shape: {tuple(train_stimuli.shape)}")
    print(f"Canonical train batch size: {batch_info['train_batch_size']}")

    seed_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_root = ensure_dir(output_root / f"seed_{seed}")
        connectome_curve, connectome_summary = run_fixed_steps(
            model_kind="connectome",
            seed=seed,
            max_iters=args.max_iters,
            baseline_checkpoint=args.baseline_checkpoint,
            random_mask_path=args.random_mask_path,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
        )
        random_curve, random_summary = run_fixed_steps(
            model_kind="random",
            seed=seed,
            max_iters=args.max_iters,
            baseline_checkpoint=args.baseline_checkpoint,
            random_mask_path=args.random_mask_path,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
        )

        connectome_curve.to_csv(seed_root / "connectome_curve.csv", index=False)
        random_curve.to_csv(seed_root / "random_curve.csv", index=False)

        seed_summary = {
            "seed": seed,
            "batch_info": batch_info,
            "connectome": connectome_summary,
            "random": random_summary,
        }
        write_json(seed_root / "summary.json", seed_summary)
        seed_summaries.append(seed_summary)

        print(
            f"seed={seed} connectome loss@5={value_at_iteration(connectome_curve, 5, 'task_loss')} "
            f"activity@5={value_at_iteration(connectome_curve, 5, 'activity_abs_mean')}"
        )
        print(
            f"seed={seed} random loss@5={value_at_iteration(random_curve, 5, 'task_loss')} "
            f"activity@5={value_at_iteration(random_curve, 5, 'activity_abs_mean')}"
        )

    conclusion = conclusion_from_seed_summaries(seed_summaries)
    report_payload = {
        "seeds": args.seeds,
        "max_iters": args.max_iters,
        "conclusion": conclusion,
        "seed_summaries": seed_summaries,
    }
    write_json(output_root / "summary.json", report_payload)
    write_report(report_path=args.report_path, seed_summaries=seed_summaries, conclusion=conclusion)
    print(f"Conclusion: {conclusion}")


def main() -> None:
    args = parse_args()
    run_all_seeds(args)


if __name__ == "__main__":
    main()
