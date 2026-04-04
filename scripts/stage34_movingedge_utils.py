from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from path_helpers import configure_flyvis_path, default_checkpoint_path

REPO_ROOT = Path(__file__).resolve().parents[1]
configure_flyvis_path(REPO_ROOT)

from flyvis.datasets.moving_bar import MovingEdge
from flyvis.network.network import Network
from flyvis.utils.chkpt_utils import recover_network
from flyvis.utils.config_utils import get_default_config


BASELINE_CHECKPOINT = default_checkpoint_path()


@dataclass(frozen=True)
class MovingEdgeBatchConfig:
    dt: float = 0.02
    speed: float = 2.4
    angles: tuple[int, ...] = (0, 90, 180, 270)
    intensities: tuple[int, ...] = (0, 1)
    dataset_t_pre: float = 1.0
    dataset_t_post: float = 1.0
    steady_state_seconds: float = 0.25


@dataclass(frozen=True)
class MovingEdgeGeneralizationConfig:
    dt: float = 0.02
    train_speed_values: tuple[float, ...] = (2.4,)
    test_speed_values: tuple[float, ...] = (19.0,)
    intensities: tuple[int, ...] = (0, 1)
    dataset_t_pre: float = 1.0
    dataset_t_post: float = 1.0
    steady_state_seconds: float = 0.25


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (torch.Tensor,)):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True))


def build_fixed_movingedge_batch(
    config: MovingEdgeBatchConfig,
) -> tuple[MovingEdge, torch.Tensor, pd.DataFrame]:
    dataset = MovingEdge(
        dt=config.dt,
        t_pre=config.dataset_t_pre,
        t_post=config.dataset_t_post,
    )

    selector = (
        dataset.arg_df["speed"].eq(config.speed)
        & dataset.arg_df["angle"].isin(config.angles)
        & dataset.arg_df["intensity"].isin(config.intensities)
    )
    metadata = dataset.arg_df.loc[selector].copy()
    metadata["dataset_index"] = metadata.index
    metadata = metadata.sort_values(["angle", "intensity"]).reset_index(drop=True)

    stimuli = torch.stack(
        [
            torch.nan_to_num(dataset[int(idx)], nan=0.5)
            for idx in metadata["dataset_index"].tolist()
        ],
        dim=0,
    ).unsqueeze(2)

    return dataset, stimuli, metadata


def angle_parity_split(dataset: MovingEdge) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique_angles = tuple(sorted(int(a) for a in dataset.arg_df["angle"].unique()))
    return unique_angles[::2], unique_angles[1::2]


def build_movingedge_subset(
    dataset: MovingEdge,
    *,
    angles: Sequence[int],
    speeds: Sequence[float],
    intensities: Sequence[int],
) -> tuple[torch.Tensor, pd.DataFrame]:
    selector = (
        dataset.arg_df["angle"].isin(list(angles))
        & dataset.arg_df["speed"].isin(list(speeds))
        & dataset.arg_df["intensity"].isin(list(intensities))
    )
    metadata = dataset.arg_df.loc[selector].copy()
    metadata["dataset_index"] = metadata.index
    metadata = metadata.sort_values(["angle", "speed", "intensity"]).reset_index(drop=True)
    stimuli = torch.stack(
        [
            torch.nan_to_num(dataset[int(idx)], nan=0.5)
            for idx in metadata["dataset_index"].tolist()
        ],
        dim=0,
    ).unsqueeze(2)
    return stimuli, metadata


def build_movingedge_train_test_split(
    config: MovingEdgeGeneralizationConfig,
) -> tuple[MovingEdge, dict[str, torch.Tensor], dict[str, pd.DataFrame], dict[str, tuple[int, int]]]:
    dataset = MovingEdge(
        dt=config.dt,
        t_pre=config.dataset_t_pre,
        t_post=config.dataset_t_post,
    )
    train_angles, test_angles = angle_parity_split(dataset)

    train_stimuli, train_metadata = build_movingedge_subset(
        dataset,
        angles=train_angles,
        speeds=config.train_speed_values,
        intensities=config.intensities,
    )
    test_stimuli, test_metadata = build_movingedge_subset(
        dataset,
        angles=test_angles,
        speeds=config.test_speed_values,
        intensities=config.intensities,
    )

    feature_slices = {
        "train": feature_time_slice(dataset, train_metadata),
        "test": feature_time_slice(dataset, test_metadata),
    }
    stimuli = {"train": train_stimuli, "test": test_stimuli}
    metadata = {"train": train_metadata, "test": test_metadata}
    return dataset, stimuli, metadata, feature_slices


def feature_time_slice(
    dataset: MovingEdge,
    metadata: pd.DataFrame,
) -> tuple[int, int]:
    start = int(round(dataset.t_pre / dataset.dt))
    stim_frames = int(round(float(metadata["t_stim"].iloc[0]) / dataset.dt))
    stop = start + stim_frames
    return start, stop


def direction_targets(metadata: pd.DataFrame) -> torch.Tensor:
    angles = torch.tensor(
        metadata["angle"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    radians = torch.deg2rad(angles)
    return torch.stack((torch.cos(radians), torch.sin(radians)), dim=1)


def init_network_from_baseline(
    checkpoint_path: Path | str = BASELINE_CHECKPOINT,
) -> Network:
    config = get_default_config(
        overrides=["task_name=flow", "ensemble_and_network_id=0000/000"]
    )
    network = Network(**config.network)
    recover_network(network, checkpoint_path)
    return network


def init_linear_decoder(network: Network) -> torch.nn.Linear:
    n_central = len(network.connectome.central_cells_index[:])
    return torch.nn.Linear(n_central, 2)


def run_network_batch(
    network: Network,
    stimuli: torch.Tensor,
    dt: float,
    steady_state_seconds: float,
) -> torch.Tensor:
    batch_size, n_frames = stimuli.shape[:2]
    state = network.steady_state(
        steady_state_seconds,
        dt,
        batch_size=batch_size,
        value=0.5,
    )
    network.stimulus.zero(batch_size, n_frames)
    network.stimulus.add_input(stimuli)
    return network(network.stimulus(), dt, state=state)


def extract_central_activity(
    activity: torch.Tensor,
    network: Network,
) -> torch.Tensor:
    central_idx = torch.tensor(
        network.connectome.central_cells_index[:],
        dtype=torch.long,
        device=activity.device,
    )
    return activity.index_select(2, central_idx)


def pooled_decoder_features(
    activity: torch.Tensor,
    network: Network,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    central = extract_central_activity(activity, network)
    pooled = central[:, start:stop].mean(dim=1)
    final = central[:, -1]
    return pooled, final


def normalized_activity_metrics(
    activity: torch.Tensor,
    network: Network,
) -> Dict[str, float]:
    central = extract_central_activity(activity, network)
    return {
        "activity_abs_mean_all_nodes": float(activity.abs().mean().detach().cpu()),
        "activity_abs_mean_central": float(central.abs().mean().detach().cpu()),
        "activity_rms_all_nodes": float(
            torch.sqrt((activity**2).mean()).detach().cpu()
        ),
        "activity_rms_central": float(
            torch.sqrt((central**2).mean()).detach().cpu()
        ),
    }


def direction_eval_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    metadata: pd.DataFrame,
) -> Dict[str, float]:
    mse = F.mse_loss(prediction, target).detach().cpu().item()
    cosine = F.cosine_similarity(prediction, target, dim=1).mean().detach().cpu().item()

    pred_angles = (
        torch.rad2deg(torch.atan2(prediction[:, 1], prediction[:, 0])).detach().cpu()
        % 360.0
    )
    target_angles = torch.tensor(
        metadata["angle"].to_numpy(dtype=np.float32), dtype=torch.float32
    )
    angle_error = ((pred_angles - target_angles + 180.0) % 360.0 - 180.0).abs()

    unique_angles = torch.tensor(
        sorted(metadata["angle"].unique().tolist()), dtype=torch.float32
    )
    nearest = (
        (pred_angles[:, None] - unique_angles[None, :] + 180.0) % 360.0 - 180.0
    ).abs()
    pred_classes = unique_angles[nearest.argmin(dim=1)]
    nearest_accuracy = pred_classes.eq(target_angles).float().mean().item()

    return {
        "direction_mse": float(mse),
        "direction_cosine_mean": float(cosine),
        "direction_angle_error_deg": float(angle_error.mean().item()),
        "direction_nearest_angle_accuracy": float(nearest_accuracy),
    }


def evaluate_direction_split(
    network: Network,
    decoder: torch.nn.Module,
    stimuli: torch.Tensor,
    targets: torch.Tensor,
    metadata: pd.DataFrame,
    dt: float,
    steady_state_seconds: float,
    feature_start: int,
    feature_stop: int,
) -> Dict[str, Any]:
    with torch.no_grad():
        activity = run_network_batch(
            network=network,
            stimuli=stimuli,
            dt=dt,
            steady_state_seconds=steady_state_seconds,
        )
        pooled, final = pooled_decoder_features(
            activity=activity,
            network=network,
            start=feature_start,
            stop=feature_stop,
        )
        prediction = decoder(pooled)
        task_loss = float(F.mse_loss(prediction, targets).detach().cpu())

    return {
        "task_loss": task_loss,
        "pooled_state": pooled.detach().cpu(),
        "final_state": final.detach().cpu(),
        "prediction": prediction.detach().cpu(),
        **normalized_activity_metrics(activity, network),
        **direction_eval_metrics(prediction, targets, metadata),
    }


def effective_edge_weight(network: Network) -> torch.Tensor:
    params = network._param_api()
    return params.edges.weight.detach().clone()


def syn_strength_parameter_indices(network: Network, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        network.edge_params.syn_strength.indices,
        dtype=torch.long,
        device=device,
    )


def aggregate_edge_scores_to_synapse_params(
    network: Network,
    edge_scores: torch.Tensor,
) -> torch.Tensor:
    edge_to_param = syn_strength_parameter_indices(network, edge_scores.device)
    n_params = network.edge_params.syn_strength.raw_values.numel()
    param_scores = torch.zeros(n_params, dtype=edge_scores.dtype, device=edge_scores.device)
    param_scores.scatter_add_(0, edge_to_param, edge_scores)
    return param_scores


def usage_ranking_scores(
    network: Network,
    activity: torch.Tensor,
) -> torch.Tensor:
    mean_abs_activity = activity.abs().mean(dim=(0, 1))
    source_idx = network._source_indices.to(mean_abs_activity.device)
    weights = effective_edge_weight(network).to(mean_abs_activity.device)
    edge_scores = weights.abs() * mean_abs_activity.index_select(0, source_idx)
    return aggregate_edge_scores_to_synapse_params(network, edge_scores)


def weight_only_ranking_scores(network: Network) -> torch.Tensor:
    return aggregate_edge_scores_to_synapse_params(
        network,
        effective_edge_weight(network).abs(),
    )


def mask_synapses(
    network: Network,
    edge_indices: Sequence[int],
) -> torch.Tensor:
    syn_strength = network.edge_params.syn_strength.raw_values
    original = syn_strength.detach().clone()
    if edge_indices:
        syn_strength.data[list(edge_indices)] = 0.0
    return original


def restore_synapses(network: Network, original: torch.Tensor) -> None:
    network.edge_params.syn_strength.raw_values.data.copy_(original)


def flatten_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(tensor.reshape(-1))


def normalized_state_divergence(
    masked_state: torch.Tensor,
    base_state: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    numerator = flatten_l2_norm(masked_state - base_state)
    denominator = flatten_l2_norm(base_state) + eps
    return float((numerator / denominator).detach().cpu())


def cosine_dissimilarity(masked_state: torch.Tensor, base_state: torch.Tensor) -> float:
    masked_flat = masked_state.reshape(1, -1)
    base_flat = base_state.reshape(1, -1)
    cosine = F.cosine_similarity(masked_flat, base_flat, dim=1)
    return float((1.0 - cosine).detach().cpu().item())


def save_metadata_csv(path: Path | str, metadata: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(path, index=False)


def variant_table(variant_lambdas: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"variant": name, "lambda_act": value} for name, value in variant_lambdas.items()]
    )


def batch_config_dict(config: MovingEdgeBatchConfig) -> Dict[str, Any]:
    return asdict(config)


def summarize_instability(loss_history: Sequence[float]) -> Dict[str, float | bool]:
    loss = np.asarray(loss_history, dtype=np.float64)
    if loss.size == 0:
        return {"is_unstable": False, "final_over_best_ratio": float("nan")}
    best = float(loss.min())
    final = float(loss[-1])
    ratio = final / max(best, 1e-8)
    return {
        "is_unstable": bool(np.isnan(loss).any() or ratio > 1.25),
        "final_over_best_ratio": ratio,
    }
