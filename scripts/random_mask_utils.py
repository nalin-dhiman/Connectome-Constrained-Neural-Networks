from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import numpy as np
import pandas as pd
import torch
from datamate import Namespace

from path_helpers import configure_flyvis_path

REPO_ROOT = Path(__file__).resolve().parents[1]
configure_flyvis_path(REPO_ROOT)

from flyvis.connectome.connectome import init_connectome, register_connectome
from flyvis.network.network import Network
from flyvis.utils.chkpt_utils import recover_network
from flyvis.utils.config_utils import get_default_config

from stage34_movingedge_utils import BASELINE_CHECKPOINT


DEFAULT_RANDOM_MASK_PATH = Path("results/main_results/random_mask_selfloop.pt")
DEFAULT_CONFIG_OVERRIDES = ["task_name=flow", "ensemble_and_network_id=0000/000"]


class ArrayNamespace(Namespace):
    """Namespace wrapper for numpy-backed table data."""


@dataclass(frozen=True)
class RandomMaskSummary:
    n_nodes: int
    n_edges: int
    density: float
    original_self_loops: int
    random_self_loops: int
    duplicate_edges: int
    unchanged_edge_fraction: float
    free_parameters: int
    fixed_parameters: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "original_self_loops": self.original_self_loops,
            "random_self_loops": self.random_self_loops,
            "duplicate_edges": self.duplicate_edges,
            "unchanged_edge_fraction": self.unchanged_edge_fraction,
            "free_parameters": self.free_parameters,
            "fixed_parameters": self.fixed_parameters,
        }


def _base_network_config():
    return get_default_config(overrides=DEFAULT_CONFIG_OVERRIDES)


def load_base_connectome():
    config = _base_network_config()
    return init_connectome(**config.network.connectome)


def _to_numpy_table(table: Any, fields: list[str]) -> dict[str, np.ndarray]:
    return {field: np.array(table[field][:], copy=True) for field in fields}


def _layer_index_to_numpy(layer_index: dict[str, Any]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for key, value in layer_index.items():
        if hasattr(value, "__getitem__") and not isinstance(value, np.ndarray):
            converted[key] = np.array(value[:], copy=True)
        else:
            converted[key] = np.array(value, copy=True)
    return converted


def _sample_pairs_for_type_pair(
    *,
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    n_edges: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_source = int(len(source_nodes))
    n_target = int(len(target_nodes))
    capacity = n_source * n_target
    if n_edges > capacity:
        raise RuntimeError(
            "Cannot sample unique random edges without replacement: "
            f"requested {n_edges}, capacity {capacity}."
        )
    picks = rng.choice(capacity, size=n_edges, replace=False)
    return source_nodes[picks // n_target], target_nodes[picks % n_target]


def build_random_mask_payload(seed: int) -> dict[str, Any]:
    connectome = load_base_connectome()
    nodes = pd.DataFrame(
        {
            "index": connectome.nodes.index[:],
            "type": connectome.nodes.type[:].astype(str),
            "u": connectome.nodes.u[:],
            "v": connectome.nodes.v[:],
        }
    )
    edges = pd.DataFrame(
        {
            "source_index": connectome.edges.source_index[:],
            "target_index": connectome.edges.target_index[:],
            "sign": connectome.edges.sign[:],
            "n_syn": connectome.edges.n_syn[:],
            "source_type": connectome.edges.source_type[:].astype(str),
            "target_type": connectome.edges.target_type[:].astype(str),
            "source_u": connectome.edges.source_u[:],
            "target_u": connectome.edges.target_u[:],
            "source_v": connectome.edges.source_v[:],
            "target_v": connectome.edges.target_v[:],
            "du": connectome.edges.du[:],
            "dv": connectome.edges.dv[:],
            "n_syn_certainty": connectome.edges.n_syn_certainty[:],
        }
    )

    node_ids_by_type = {
        cell_type: nodes.loc[nodes["type"].eq(cell_type), "index"].to_numpy(dtype=np.int64)
        for cell_type in sorted(nodes["type"].unique().tolist())
    }
    node_u = nodes.set_index("index")["u"].to_dict()
    node_v = nodes.set_index("index")["v"].to_dict()
    rng = np.random.default_rng(seed)

    random_source = np.empty(len(edges), dtype=np.int64)
    random_target = np.empty(len(edges), dtype=np.int64)
    for (source_type, target_type), index_array in edges.groupby(
        ["source_type", "target_type"]
    ).indices.items():
        sampled_source, sampled_target = _sample_pairs_for_type_pair(
            source_nodes=node_ids_by_type[source_type],
            target_nodes=node_ids_by_type[target_type],
            n_edges=len(index_array),
            rng=rng,
        )
        index_array = np.asarray(index_array)
        random_source[index_array] = sampled_source
        random_target[index_array] = sampled_target

    original_source = connectome.edges.source_index[:]
    original_target = connectome.edges.target_index[:]
    random_pairs = np.stack([random_source, random_target], axis=1)
    duplicate_edges = int(len(random_pairs) - len(np.unique(random_pairs, axis=0)))
    original_self_loops = int(np.sum(original_source == original_target))
    random_self_loops = int(np.sum(random_source == random_target))
    n_nodes = int(len(connectome.nodes.index))
    n_edges = int(len(connectome.edges.source_index))

    return {
        "seed": seed,
        "connectome_type": "RandomMaskConnectome",
        "nodes": _to_numpy_table(connectome.nodes, ["index", "type", "u", "v", "role"]),
        "edges": {
            "source_index": random_source.astype(np.int64),
            "target_index": random_target.astype(np.int64),
            "sign": np.array(connectome.edges.sign[:], copy=True),
            "n_syn": np.array(connectome.edges.n_syn[:], copy=True),
            "source_type": np.array(connectome.edges.source_type[:], copy=True),
            "target_type": np.array(connectome.edges.target_type[:], copy=True),
            "source_u": np.array([node_u[int(idx)] for idx in random_source], dtype=np.int32),
            "target_u": np.array([node_u[int(idx)] for idx in random_target], dtype=np.int32),
            "source_v": np.array([node_v[int(idx)] for idx in random_source], dtype=np.int32),
            "target_v": np.array([node_v[int(idx)] for idx in random_target], dtype=np.int32),
            "du": np.array(connectome.edges.du[:], copy=True),
            "dv": np.array(connectome.edges.dv[:], copy=True),
            "n_syn_certainty": np.array(connectome.edges.n_syn_certainty[:], copy=True),
        },
        "layer_index": _layer_index_to_numpy(connectome.nodes.layer_index),
        "unique_cell_types": np.array(connectome.unique_cell_types[:], copy=True),
        "input_cell_types": np.array(connectome.input_cell_types[:], copy=True),
        "output_cell_types": np.array(connectome.output_cell_types[:], copy=True),
        "intermediate_cell_types": np.array(
            connectome.intermediate_cell_types[:], copy=True
        ),
        "layout": np.array(connectome.layout[:], copy=True),
        "central_cells_index": np.array(connectome.central_cells_index[:], copy=True),
        "stats": {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": float(n_edges / (n_nodes * n_nodes)),
            "original_self_loops": original_self_loops,
            "random_self_loops": random_self_loops,
            "duplicate_edges": duplicate_edges,
            "unchanged_edge_fraction": float(
                np.mean(
                    (random_source == original_source) & (random_target == original_target)
                )
            ),
        },
    }


def save_random_mask(payload: dict[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_random_mask_payload(mask_path: Path | str) -> dict[str, Any]:
    return torch.load(Path(mask_path), map_location="cpu", weights_only=False)


@register_connectome
class RandomMaskConnectome:
    """Synthetic connectome with random edge incidence and matched edge-table labels."""

    def __init__(self, mask_path: str | Path) -> None:
        payload = load_random_mask_payload(mask_path)
        self.nodes = ArrayNamespace(**payload["nodes"])
        self.nodes.layer_index = payload["layer_index"]
        self.edges = ArrayNamespace(**payload["edges"])
        self.unique_cell_types = payload["unique_cell_types"]
        self.input_cell_types = payload["input_cell_types"]
        self.output_cell_types = payload["output_cell_types"]
        self.intermediate_cell_types = payload["intermediate_cell_types"]
        self.layout = payload["layout"]
        self.central_cells_index = payload["central_cells_index"]


def init_random_mask_network(
    mask_path: Path | str = DEFAULT_RANDOM_MASK_PATH,
    checkpoint_path: Path | str | None = BASELINE_CHECKPOINT,
) -> Network:
    config = _base_network_config()
    config.network.connectome.type = "RandomMaskConnectome"
    config.network.connectome.mask_path = str(mask_path)
    for key in ["file", "extent", "n_syn_fill"]:
        if key in config.network.connectome:
            del config.network.connectome[key]
    network = Network(**config.network)
    if checkpoint_path is not None:
        recover_network(network, checkpoint_path)
    return network


def summarize_random_mask(mask_path: Path | str) -> RandomMaskSummary:
    payload = load_random_mask_payload(mask_path)
    checkpoint_path = BASELINE_CHECKPOINT if Path(BASELINE_CHECKPOINT).exists() else None
    network = init_random_mask_network(mask_path=mask_path, checkpoint_path=checkpoint_path)
    return RandomMaskSummary(
        n_nodes=int(payload["stats"]["n_nodes"]),
        n_edges=int(payload["stats"]["n_edges"]),
        density=float(payload["stats"]["density"]),
        original_self_loops=int(payload["stats"]["original_self_loops"]),
        random_self_loops=int(payload["stats"]["random_self_loops"]),
        duplicate_edges=int(payload["stats"]["duplicate_edges"]),
        unchanged_edge_fraction=float(payload["stats"]["unchanged_edge_fraction"]),
        free_parameters=int(network.num_parameters.free),
        fixed_parameters=int(network.num_parameters.fixed),
    )
