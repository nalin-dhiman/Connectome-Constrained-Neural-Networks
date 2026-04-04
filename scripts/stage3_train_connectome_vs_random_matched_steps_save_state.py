#!/usr/bin/env python

from __future__ import annotations

import argparse
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

from stage3_train_connectome_vs_random_matched_steps import (
    DEFAULT_RANDOM_MASK_PATH,
    finite_or_none,
    gradient_norm,
    init_model,
    model_summary,
    sync_if_cuda,
)
from stage34_movingedge_utils import (
    BASELINE_CHECKPOINT,
    MovingEdgeGeneralizationConfig,
    batch_config_dict,
    build_movingedge_train_test_split,
    direction_targets,
    ensure_dir,
    pooled_decoder_features,
    run_network_batch,
    set_global_seed,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("results/main_results/stage3_connectome_vs_random_matched_steps_saved")
DEFAULT_REPORT_PATH = Path("docs/generated/stage3_connectome_vs_random_matched_steps_saved.md")
MODEL_KINDS = ("connectome", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-step Stage 3 connectome vs random rerun with saved checkpoints "
            "and one post-training activity snapshot."
        )
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
    parser.add_argument(
        "--max-activity-mb",
        type=float,
        default=700.0,
        help="Stop instead of saving if a single activity tensor exceeds this size.",
    )
    return parser.parse_args()


def activity_size_mb(activity: torch.Tensor) -> float:
    return float(activity.numel() * activity.element_size() / (1024**2))


def save_model_artifacts(
    *,
    seed_root: Path,
    model_kind: str,
    network,
    decoder,
    stimuli: torch.Tensor,
    dt: float,
    steady_state_seconds: float,
    max_activity_mb: float,
) -> dict[str, Any]:
    network_path = seed_root / f"{model_kind}_network.pt"
    decoder_path = seed_root / f"{model_kind}_decoder.pt"
    activity_path = seed_root / f"{model_kind}_activity.pt"

    torch.save(network.state_dict(), network_path)
    torch.save(decoder.state_dict(), decoder_path)

    network.eval()
    decoder.eval()
    with torch.no_grad():
        activity = run_network_batch(
            network=network,
            stimuli=stimuli,
            dt=dt,
            steady_state_seconds=steady_state_seconds,
        ).detach()

    activity_shape = list(activity.shape)
    activity_mb = activity_size_mb(activity)
    print(
        f"{model_kind} post-training activity shape={tuple(activity_shape)} "
        f"estimated_size_mb={activity_mb:.2f}"
    )

    if not torch.isfinite(activity).all():
        raise RuntimeError(
            f"{model_kind} produced non-finite post-training activity for saving."
        )
    if activity_mb > max_activity_mb:
        raise RuntimeError(
            f"{model_kind} activity tensor is too large to save safely: "
            f"{activity_mb:.2f} MB > {max_activity_mb:.2f} MB."
        )

    torch.save(activity.cpu(), activity_path)
    return {
        "network_checkpoint": str(network_path),
        "decoder_checkpoint": str(decoder_path),
        "activity_tensor": str(activity_path),
        "activity_shape": activity_shape,
        "activity_size_mb": activity_mb,
        "activity_dtype": str(activity.dtype).replace("torch.", ""),
    }


def run_fixed_steps_and_save(
    *,
    model_kind: str,
    seed: int,
    seed_root: Path,
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
    max_activity_mb: float,
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
    if not remained_finite:
        raise RuntimeError(f"{model_kind} seed {seed} became non-finite during rerun.")

    artifact_info = save_model_artifacts(
        seed_root=seed_root,
        model_kind=model_kind,
        network=network,
        decoder=decoder,
        stimuli=stimuli,
        dt=dt,
        steady_state_seconds=steady_state_seconds,
        max_activity_mb=max_activity_mb,
    )

    return curve_df, {
        **summary,
        **artifact_info,
        "seed": seed,
        "iterations_requested": int(max_iters),
        "iterations_completed": int(len(curve_df)),
        "total_elapsed_sec": float(curve_df["elapsed_sec"].iloc[-1]),
        "mean_iter_total_sec": float(curve_df["iter_total_sec"].mean()),
        "remained_finite": bool(remained_finite),
        "final_task_loss": float(curve_df["task_loss"].iloc[-1]),
        "final_activity_abs_mean": float(curve_df["activity_abs_mean"].iloc[-1]),
    }


def write_report(
    *,
    report_path: Path,
    seed_summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Stage 3 Connectome vs Random Matched Steps Saved",
        "",
        "Scope:",
        "- Identical 5-step matched-step experiment.",
        "- No scientific changes to architecture, loss, dataset, optimizer, or seeds.",
        "- Additional artifacts only: trained checkpoints and one post-training activity snapshot per run.",
        "",
    ]
    for item in seed_summaries:
        lines.extend(
            [
                f"## Seed {item['seed']}",
                "",
            ]
        )
        for model_kind in MODEL_KINDS:
            summary = item[model_kind]
            lines.extend(
                [
                    f"- {model_kind}: iterations={summary['iterations_completed']}/{summary['iterations_requested']}, finite={summary['remained_finite']}, activity_shape={tuple(summary['activity_shape'])}",
                    f"- {model_kind} checkpoints: network={summary['network_checkpoint']}, decoder={summary['decoder_checkpoint']}",
                    f"- {model_kind} activity tensor: path={summary['activity_tensor']}, size_mb={summary['activity_size_mb']:.2f}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Confirmation:",
            "- This rerun preserves trained state and one post-training activity tensor per run, so it is suitable for later mechanism analysis.",
        ]
    )
    report_path.write_text("\n".join(lines))


def run_all_seeds(args: argparse.Namespace) -> None:
    if args.output_root.exists() or args.report_path.exists():
        raise FileExistsError("Refusing to overwrite existing saved-state outputs.")
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
        "max_activity_mb": float(args.max_activity_mb),
    }
    print(f"Canonical train input shape: {tuple(train_stimuli.shape)}")
    print(f"Canonical train batch size: {batch_info['train_batch_size']}")

    seed_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_root = ensure_dir(output_root / f"seed_{seed}")
        connectome_curve, connectome_summary = run_fixed_steps_and_save(
            model_kind="connectome",
            seed=seed,
            seed_root=seed_root,
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
            max_activity_mb=args.max_activity_mb,
        )
        random_curve, random_summary = run_fixed_steps_and_save(
            model_kind="random",
            seed=seed,
            seed_root=seed_root,
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
            max_activity_mb=args.max_activity_mb,
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

    write_json(output_root / "summary.json", {"seeds": seed_summaries, "batch_info": batch_info})
    write_report(report_path=args.report_path, seed_summaries=seed_summaries)


def main() -> None:
    args = parse_args()
    run_all_seeds(args)


if __name__ == "__main__":
    main()
