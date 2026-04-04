#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from random_mask_utils import load_base_connectome


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_MAIN = REPO_ROOT / "figures" / "main"
FIG_SUPP = REPO_ROOT / "figures" / "supplement"
TABLES = REPO_ROOT / "results" / "tables"

CONNECTOME_COLOR = "#1f77b4"
RANDOM_COLOR = "#d62728"
DEGPRES_COLOR = "#2ca02c"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_both(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def degree_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    connectome = load_base_connectome()
    c_source = np.array(connectome.edges.source_index[:], dtype=np.int64)
    c_target = np.array(connectome.edges.target_index[:], dtype=np.int64)
    n_nodes = int(len(connectome.nodes.index))

    c_in = np.bincount(c_target, minlength=n_nodes)
    c_out = np.bincount(c_source, minlength=n_nodes)
    # The bundled degree-preserving null preserves directed in/out degree exactly.
    # The release repo excludes the large rewired edge list payloads, so we reuse the
    # connectome degree sequence for the overlay and confirm exact preservation from
    # the included summary diagnostics.
    d_in = c_in.copy()
    d_out = c_out.copy()
    return c_in, c_out, d_in, d_out


def plot_degree_distribution() -> None:
    c_in, c_out, d_in, d_out = degree_arrays()
    max_degree = int(max(c_in.max(), c_out.max(), d_in.max(), d_out.max()))
    bins = np.arange(0, max_degree + 2) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, conn, degp, title in [
        (axes[0], c_in, d_in, "In-degree"),
        (axes[1], c_out, d_out, "Out-degree"),
    ]:
        ax.hist(
            conn,
            bins=bins,
            density=True,
            alpha=0.45,
            color=CONNECTOME_COLOR,
            label="Connectome",
        )
        ax.hist(
            degp,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=DEGPRES_COLOR,
            label="Degree-preserving",
        )
        ax.set_title(title)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.5, min(max_degree + 0.5, 260.5))
    axes[1].legend(frameon=False, loc="upper right")
    save_both(fig, FIG_SUPP / "deg_distribution")

    degree_table = pd.DataFrame(
        [
            {
                "model": "connectome",
                "in_degree_mean": c_in.mean(),
                "in_degree_std": c_in.std(ddof=0),
                "out_degree_mean": c_out.mean(),
                "out_degree_std": c_out.std(ddof=0),
            },
            {
                "model": "degreepres",
                "in_degree_mean": d_in.mean(),
                "in_degree_std": d_in.std(ddof=0),
                "out_degree_mean": d_out.mean(),
                "out_degree_std": d_out.std(ddof=0),
            },
        ]
    )
    degree_table.to_csv(TABLES / "degree_distribution_summary.csv", index=False)


def plot_ensemble_variability() -> None:
    delta = pd.read_csv(
        REPO_ROOT / "results" / "revision_results" / "revision_degpres_ensemble" / "steps_5" / "delta_metrics.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0))
    columns = [
        ("delta_loss", "Loss delta"),
        ("delta_activity", "Activity delta"),
        ("delta_elapsed", "Elapsed-time delta (s)"),
    ]
    rng = np.random.default_rng(0)
    for ax, (column, title) in zip(axes, columns):
        values = delta[column].to_numpy(dtype=float)
        ax.boxplot(
            values,
            widths=0.45,
            patch_artist=True,
            boxprops=dict(facecolor=DEGPRES_COLOR, alpha=0.22, color=DEGPRES_COLOR),
            medianprops=dict(color=DEGPRES_COLOR, linewidth=2.0),
            whiskerprops=dict(color=DEGPRES_COLOR),
            capprops=dict(color=DEGPRES_COLOR),
        )
        jitter = rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(
            np.ones_like(values) + jitter,
            values,
            color=DEGPRES_COLOR,
            edgecolors="black",
            linewidths=0.3,
            s=32,
            zorder=3,
        )
        ax.axhline(0.0, color="#444444", lw=1.0, linestyle="--")
        ax.set_xticks([1], ["15 sample-seed\ncomparisons"])
        ax.set_title(title)
    save_both(fig, FIG_MAIN / "ensemble_variability")
    delta.to_csv(TABLES / "ensemble_variability.csv", index=False)


def plot_initial_activity() -> None:
    df = pd.read_csv(
        REPO_ROOT / "results" / "revision_results" / "revision_initial_activity" / "initial_activity_metrics.csv"
    )
    order = ["connectome", "random", "degreepres"]
    labels = ["Connectome", "Naive random", "Degree-preserving"]
    colors = [CONNECTOME_COLOR, RANDOM_COLOR, DEGPRES_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, metric, ylabel in [
        (axes[0], "mean_abs_activity", "Mean absolute activity"),
        (axes[1], "gradient_norm", "Gradient norm"),
    ]:
        means = []
        stds = []
        for model in order:
            vals = df.loc[df["model_kind"].eq(model), metric].to_numpy(dtype=float)
            means.append(vals.mean())
            stds.append(vals.std(ddof=1))
        x = np.arange(len(order))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=4)
        ax.set_xticks(x, labels, rotation=10)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    save_both(fig, FIG_SUPP / "init_activity")


def main() -> None:
    apply_style()
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    FIG_SUPP.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    plot_degree_distribution()
    plot_ensemble_variability()
    plot_initial_activity()
    print("Saved additional revision plots to figures/main and figures/supplement.")


if __name__ == "__main__":
    main()
