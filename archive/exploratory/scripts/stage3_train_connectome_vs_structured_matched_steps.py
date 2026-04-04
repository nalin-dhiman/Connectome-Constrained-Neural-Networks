#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import pandas as pd
import torch
import torch.nn.functional as F

from random_mask_utils import init_random_mask_network
from smallworld_mask_utils import (
    DEFAULT_SMALLWORLD_MASK_PATH,
    init_smallworld_mask_network,
)
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


DEFAULT_OUTPUT_ROOT = Path("results/stage3_connectome_vs_structured_matched_steps")
DEFAULT_STEPS = (5, 10)
DEFAULT_EXISTING_RESULTS = {
    5: Path("results/stage3_connectome_vs_random_matched_steps"),
    10: Path("results/stage3_connectome_vs_random_matched_steps_10"),
}
MODEL_KINDS = ("connectome", "random", "smallworld")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured matched-step Stage 3 comparison with connectome, random, and small-world graphs."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=BASELINE_CHECKPOINT,
    )
    parser.add_argument(
        "--random-mask-path",
        type=Path,
        default=Path("results/random_mask_selfloop.pt"),
    )
    parser.add_argument(
        "--smallworld-mask-path",
        type=Path,
        default=DEFAULT_SMALLWORLD_MASK_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--train-speeds", nargs="+", type=float, default=[2.4])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[19.0])
    return parser.parse_args()


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


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    smallworld_mask_path: Path,
    learning_rate: float,
):
    if model_kind == "connectome":
        network = init_network_from_baseline(baseline_checkpoint)
    elif model_kind == "random":
        network = init_random_mask_network(
            mask_path=random_mask_path,
            checkpoint_path=baseline_checkpoint,
        )
    elif model_kind == "smallworld":
        network = init_smallworld_mask_network(
            mask_path=smallworld_mask_path,
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
    smallworld_mask_path: Path,
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
        smallworld_mask_path=smallworld_mask_path,
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
        task_loss.backward()
        grad_norm_value = gradient_norm(
            list(network.parameters()) + list(decoder.parameters())
        )
        optimizer.step()
        network.clamp()
        sync_if_cuda()
        iter_end = time.perf_counter()

        task_loss_value = float(task_loss.detach().cpu())
        activity_value = float(activity.abs().mean().detach().cpu())
        finite = math.isfinite(task_loss_value) and math.isfinite(activity_value)
        if grad_norm_value is not None:
            finite = finite and math.isfinite(grad_norm_value)
        remained_finite = remained_finite and finite
        if not finite and iteration < 5:
            raise RuntimeError(f"{model_kind} seed {seed} became non-finite before iter 5.")

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
                "iter_total_sec": iter_end - iter_start,
                "finite": finite,
            }
        )
        if not finite:
            break

    curve_df = pd.DataFrame(rows)
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


def copy_existing_baseline(
    *,
    existing_root: Path,
    output_seed_root: Path,
    seed: int,
) -> dict[str, Any]:
    existing_seed_root = existing_root / f"seed_{seed}"
    if not existing_seed_root.exists():
        raise FileNotFoundError(f"Missing existing seed root: {existing_seed_root}")

    for model_kind in ("connectome", "random"):
        source_curve = existing_seed_root / f"{model_kind}_curve.csv"
        if not source_curve.exists():
            raise FileNotFoundError(f"Missing existing curve: {source_curve}")
        shutil.copy2(source_curve, output_seed_root / f"{model_kind}_curve.csv")

    summary = json.loads((existing_seed_root / "summary.json").read_text())
    return {
        "connectome": summary["connectome"],
        "random": summary["random"],
    }


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> str:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return "n/a"
    return f"{float(row.iloc[0][column]):.6f}"


def run_all(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite existing structured matched-step outputs.")
    if not args.random_mask_path.exists():
        raise FileNotFoundError(f"Random mask not found: {args.random_mask_path}")
    if not args.smallworld_mask_path.exists():
        raise FileNotFoundError(f"Small-world mask not found: {args.smallworld_mask_path}")
    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")

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
    }

    print(f"Canonical train input shape: {tuple(train_stimuli.shape)}")
    print(f"Canonical train batch size: {batch_info['train_batch_size']}")

    max_step = max(args.steps)
    smallworld_curves_by_seed: dict[int, pd.DataFrame] = {}
    smallworld_summaries_by_seed: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        curve_df, summary = run_fixed_steps(
            model_kind="smallworld",
            seed=seed,
            max_iters=max_step,
            baseline_checkpoint=args.baseline_checkpoint,
            random_mask_path=args.random_mask_path,
            smallworld_mask_path=args.smallworld_mask_path,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
        )
        smallworld_curves_by_seed[seed] = curve_df
        smallworld_summaries_by_seed[seed] = summary

    for step in args.steps:
        if step not in DEFAULT_EXISTING_RESULTS:
            raise KeyError(f"No existing baseline results configured for step horizon {step}")
        steps_root = ensure_dir(output_root / f"steps_{step}")
        run_summary = {
            "steps": int(step),
            "batch_info": batch_info,
            "seed_summaries": [],
        }

        for seed in args.seeds:
            seed_root = ensure_dir(steps_root / f"seed_{seed}")
            existing_summaries = copy_existing_baseline(
                existing_root=DEFAULT_EXISTING_RESULTS[step],
                output_seed_root=seed_root,
                seed=seed,
            )
            smallworld_curve = smallworld_curves_by_seed[seed].loc[
                smallworld_curves_by_seed[seed]["iteration"].le(step)
            ].copy()
            smallworld_curve.to_csv(seed_root / "smallworld_curve.csv", index=False)
            smallworld_summary = {
                **smallworld_summaries_by_seed[seed],
                "iterations_requested": int(step),
                "iterations_completed": int(len(smallworld_curve)),
                "total_elapsed_sec": float(smallworld_curve["elapsed_sec"].iloc[-1]),
                "mean_iter_total_sec": float(smallworld_curve["iter_total_sec"].mean()),
                "remained_finite": bool(smallworld_curve["finite"].all()),
                "final_task_loss": float(smallworld_curve["task_loss"].iloc[-1]),
                "final_activity_abs_mean": float(
                    smallworld_curve["activity_abs_mean"].iloc[-1]
                ),
            }
            seed_summary = {
                "seed": seed,
                "connectome": existing_summaries["connectome"],
                "random": existing_summaries["random"],
                "smallworld": smallworld_summary,
            }
            write_json(seed_root / "summary.json", seed_summary)
            run_summary["seed_summaries"].append(seed_summary)
            print(
                f"steps={step} seed={seed} smallworld loss@{step}="
                f"{value_at_iteration(smallworld_curve, step, 'task_loss')} "
                f"activity@{step}={value_at_iteration(smallworld_curve, step, 'activity_abs_mean')}"
            )

        write_json(steps_root / "summary.json", run_summary)


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
