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

from ringlattice_mask_utils import (
    DEFAULT_RINGLATTICE_MASK_PATH,
    init_ringlattice_mask_network,
)
from stage34_movingedge_utils import (
    BASELINE_CHECKPOINT,
    MovingEdgeGeneralizationConfig,
    batch_config_dict,
    build_movingedge_train_test_split,
    direction_targets,
    ensure_dir,
    init_linear_decoder,
    pooled_decoder_features,
    run_network_batch,
    set_global_seed,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("results/stage3_connectome_vs_lattice_matched_steps")
DEFAULT_EXISTING_ROOT = Path("results/stage3_connectome_vs_random_matched_steps")
TARGET_STEP = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-step Stage 3 comparison with connectome, random, and ring-lattice graphs."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=TARGET_STEP)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=BASELINE_CHECKPOINT,
    )
    parser.add_argument(
        "--ringlattice-mask-path",
        type=Path,
        default=DEFAULT_RINGLATTICE_MASK_PATH,
    )
    parser.add_argument(
        "--existing-root",
        type=Path,
        default=DEFAULT_EXISTING_ROOT,
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


def run_ringlattice_fixed_steps(
    *,
    seed: int,
    max_iters: int,
    baseline_checkpoint: Path,
    ringlattice_mask_path: Path,
    learning_rate: float,
    stimuli: torch.Tensor,
    targets: torch.Tensor,
    feature_start: int,
    feature_stop: int,
    dt: float,
    steady_state_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_global_seed(seed)
    network = init_ringlattice_mask_network(
        mask_path=ringlattice_mask_path,
        checkpoint_path=baseline_checkpoint,
    )
    decoder = init_linear_decoder(network)
    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
    )
    network.train()
    decoder.train()
    summary = model_summary("ringlattice", network)

    print(
        f"seed={seed} ringlattice: nodes={summary['n_nodes']}, "
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
        if not finite and iteration < 5:
            raise RuntimeError(f"ringlattice seed {seed} became non-finite before iter 5.")
        remained_finite = remained_finite and finite

        rows.append(
            {
                "model_kind": "ringlattice",
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


def copy_existing_baselines(existing_root: Path, seed_root: Path, seed: int) -> dict[str, Any]:
    existing_seed_root = existing_root / f"seed_{seed}"
    if not existing_seed_root.exists():
        raise FileNotFoundError(f"Missing existing seed root: {existing_seed_root}")
    for model_kind in ("connectome", "random"):
        shutil.copy2(
            existing_seed_root / f"{model_kind}_curve.csv",
            seed_root / f"{model_kind}_curve.csv",
        )
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
        raise FileExistsError("Refusing to overwrite existing lattice matched-step outputs.")
    if not args.ringlattice_mask_path.exists():
        raise FileNotFoundError(f"Ring-lattice mask not found: {args.ringlattice_mask_path}")
    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {args.baseline_checkpoint}")
    if not args.existing_root.exists():
        raise FileNotFoundError(f"Existing connectome/random root not found: {args.existing_root}")

    output_root = ensure_dir(args.output_root)
    steps_root = ensure_dir(output_root / "steps_5")

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

    run_summary = {
        "steps": int(args.max_iters),
        "batch_info": batch_info,
        "seed_summaries": [],
    }

    for seed in args.seeds:
        seed_root = ensure_dir(steps_root / f"seed_{seed}")
        existing = copy_existing_baselines(args.existing_root, seed_root, seed)
        ring_curve, ring_summary = run_ringlattice_fixed_steps(
            seed=seed,
            max_iters=args.max_iters,
            baseline_checkpoint=args.baseline_checkpoint,
            ringlattice_mask_path=args.ringlattice_mask_path,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
        )
        ring_curve.to_csv(seed_root / "ringlattice_curve.csv", index=False)
        seed_summary = {
            "seed": seed,
            "connectome": existing["connectome"],
            "random": existing["random"],
            "ringlattice": ring_summary,
        }
        write_json(seed_root / "summary.json", seed_summary)
        run_summary["seed_summaries"].append(seed_summary)
        print(
            f"seed={seed} ringlattice loss@5={value_at_iteration(ring_curve, 5, 'task_loss')} "
            f"activity@5={value_at_iteration(ring_curve, 5, 'activity_abs_mean')}"
        )

    write_json(steps_root / "summary.json", run_summary)


def main() -> None:
    args = parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
