from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import torch
from datamate import Namespace

from path_helpers import configure_flyvis_path

REPO_ROOT = Path(__file__).resolve().parents[1]
configure_flyvis_path(REPO_ROOT)

from flyvis.connectome.connectome import register_connectome
from flyvis.network.network import Network

from random_mask_utils import DEFAULT_CONFIG_OVERRIDES, load_random_mask_payload
from revision_control_utils import base_network_config


DEFAULT_DEGREE_PRESERVING_MASK_PATH = Path(
    "results/revision_results/degree_preserving_random_mask.pt"
)


class ArrayNamespace(Namespace):
    """Namespace wrapper for numpy-backed table data."""


@register_connectome
class DegreePreservingRandomConnectome:
    """Synthetic connectome with preserved directed degree sequence via edge swaps."""

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


def init_degree_preserving_random_network(
    mask_path: Path | str = DEFAULT_DEGREE_PRESERVING_MASK_PATH,
    init_seed: int = 0,
) -> Network:
    config = base_network_config(init_seed)
    config.network.connectome.type = "DegreePreservingRandomConnectome"
    config.network.connectome.mask_path = str(mask_path)
    for key in ["file", "extent", "n_syn_fill"]:
        if key in config.network.connectome:
            del config.network.connectome[key]
    return Network(**config.network)
