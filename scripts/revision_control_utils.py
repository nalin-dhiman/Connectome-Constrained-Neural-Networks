from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from path_helpers import configure_flyvis_path

REPO_ROOT = Path(__file__).resolve().parents[1]
configure_flyvis_path(REPO_ROOT)

from flyvis.network.network import Network
from flyvis.utils.config_utils import get_default_config

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


DEFAULT_CONFIG_OVERRIDES = ["task_name=flow", "ensemble_and_network_id=0000/000"]
TRAIN_ITERATION_POINTS = (1, 3, 5, 10)


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


def base_network_config(init_seed: int):
    config = get_default_config(overrides=DEFAULT_CONFIG_OVERRIDES)
    if "seed" in config.network.node_config.bias:
        config.network.node_config.bias.seed = int(init_seed)
    return config


def init_connectome_network_from_scratch(init_seed: int) -> Network:
    config = base_network_config(init_seed)
    return Network(**config.network)


def init_decoder_with_seed(network: Network, init_seed: int) -> torch.nn.Linear:
    torch.manual_seed(init_seed)
    return init_linear_decoder(network)


def model_summary(model_kind: str, network: Network) -> dict[str, Any]:
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


def build_stage3_train_payload(
    *,
    train_speeds: list[float] | tuple[float, ...],
    test_speeds: list[float] | tuple[float, ...],
) -> dict[str, Any]:
    generalization_config = MovingEdgeGeneralizationConfig(
        train_speed_values=tuple(train_speeds),
        test_speed_values=tuple(test_speeds),
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
    return {
        "generalization_config": generalization_config,
        "stimuli": train_stimuli,
        "targets": train_targets,
        "feature_start": feature_start,
        "feature_stop": feature_stop,
        "batch_info": batch_info,
    }


def run_fixed_steps(
    *,
    model_kind: str,
    seed: int,
    max_iters: int,
    learning_rate: float,
    stimuli: torch.Tensor,
    targets: torch.Tensor,
    feature_start: int,
    feature_stop: int,
    dt: float,
    steady_state_seconds: float,
    network_factory: Callable[[int], Network],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    set_global_seed(seed)
    network = network_factory(seed)
    decoder = init_decoder_with_seed(network, seed)
    optimizer = torch.optim.Adam(
        list(network.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
    )
    network.train()
    decoder.train()
    summary = model_summary(model_kind, network)

    print(
        f"seed={seed} {model_kind}: nodes={summary['n_nodes']}, "
        f"edges={summary['n_edges']}, self_loops={summary['self_loops']}, "
        f"mask_type={summary['mask_type']}, free={summary['free_parameters']}, "
        f"fixed={summary['fixed_parameters']}"
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
        "init_mode": "from_scratch",
    }


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> str:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return "n/a"
    return f"{float(row.iloc[0][column]):.6f}"


def load_curve(seed_dir: Path, filename: str) -> pd.DataFrame:
    path = seed_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing curve file: {path}")
    return pd.read_csv(path)


def stats_row(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")
    return mean, std, int(len(valid))


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(valid, size=len(valid), replace=True)
        boots.append(np.mean(sample))
    boots = np.asarray(boots, dtype=float)
    return float(np.mean(valid)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def format_mean_std(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "n/a"
    if n == 1 or np.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} +/- {std:.4f}"


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
