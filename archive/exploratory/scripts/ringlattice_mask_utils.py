from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from random_mask_utils import (
    BASELINE_CHECKPOINT,
    DEFAULT_CONFIG_OVERRIDES,
    load_random_mask_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLYVIS_REPO_ROOT = REPO_ROOT / "flyvis"
if str(FLYVIS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(FLYVIS_REPO_ROOT))

from datamate import Namespace
from flyvis.connectome.connectome import register_connectome
from flyvis.network.network import Network
from flyvis.utils.chkpt_utils import recover_network
from flyvis.utils.config_utils import get_default_config


DEFAULT_RINGLATTICE_MASK_PATH = Path("results/ringlattice_mask_selfloop.pt")


class ArrayNamespace(Namespace):
    """Namespace wrapper for numpy-backed table data."""


@dataclass(frozen=True)
class RingLatticeMaskSummary:
    n_nodes: int
    n_edges: int
    density: float
    original_self_loops: int
    ringlattice_self_loops: int
    duplicate_edges: int
    in_degree_mean: float
    in_degree_std: float
    out_degree_mean: float
    out_degree_std: float
    free_parameters: int
    fixed_parameters: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density,
            "original_self_loops": self.original_self_loops,
            "ringlattice_self_loops": self.ringlattice_self_loops,
            "duplicate_edges": self.duplicate_edges,
            "in_degree_mean": self.in_degree_mean,
            "in_degree_std": self.in_degree_std,
            "out_degree_mean": self.out_degree_mean,
            "out_degree_std": self.out_degree_std,
            "free_parameters": self.free_parameters,
            "fixed_parameters": self.fixed_parameters,
        }


def _base_network_config():
    return get_default_config(overrides=DEFAULT_CONFIG_OVERRIDES)


@register_connectome
class RingLatticeMaskConnectome:
    """Synthetic connectome with ring-lattice edge incidence."""

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


def init_ringlattice_mask_network(
    mask_path: Path | str = DEFAULT_RINGLATTICE_MASK_PATH,
    checkpoint_path: Path | str | None = BASELINE_CHECKPOINT,
) -> Network:
    config = _base_network_config()
    config.network.connectome.type = "RingLatticeMaskConnectome"
    config.network.connectome.mask_path = str(mask_path)
    for key in ["file", "extent", "n_syn_fill"]:
        if key in config.network.connectome:
            del config.network.connectome[key]
    network = Network(**config.network)
    if checkpoint_path is not None:
        recover_network(network, checkpoint_path)
    return network


def summarize_ringlattice_mask(mask_path: Path | str) -> RingLatticeMaskSummary:
    payload = load_random_mask_payload(mask_path)
    network = init_ringlattice_mask_network(
        mask_path=mask_path,
        checkpoint_path=BASELINE_CHECKPOINT,
    )
    stats = payload["stats"]
    return RingLatticeMaskSummary(
        n_nodes=int(stats["n_nodes"]),
        n_edges=int(stats["n_edges"]),
        density=float(stats["density"]),
        original_self_loops=int(stats["original_self_loops"]),
        ringlattice_self_loops=int(stats["ringlattice_self_loops"]),
        duplicate_edges=int(stats["duplicate_edges"]),
        in_degree_mean=float(stats["in_degree_mean"]),
        in_degree_std=float(stats["in_degree_std"]),
        out_degree_mean=float(stats["out_degree_mean"]),
        out_degree_std=float(stats["out_degree_std"]),
        free_parameters=int(network.num_parameters.free),
        fixed_parameters=int(network.num_parameters.fixed),
    )
