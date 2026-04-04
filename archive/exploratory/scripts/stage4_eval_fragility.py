#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from stage34_movingedge_utils import (
    BASELINE_CHECKPOINT,
    cosine_dissimilarity,
    ensure_dir,
    init_linear_decoder,
    init_network_from_baseline,
    mask_synapses,
    normalized_state_divergence,
    pooled_decoder_features,
    restore_synapses,
    run_network_batch,
    set_global_seed,
    usage_ranking_scores,
    weight_only_ranking_scores,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical Stage 4 MovingEdge fragility evaluation."
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--random-repeats", type=int, default=4)
    parser.add_argument(
        "--stage3-root",
        type=Path,
        default=Path("results/stage3/movingedge_energy"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/stage4/movingedge_fragility"),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=BASELINE_CHECKPOINT,
    )
    return parser.parse_args()


def load_stage3_payload(stage3_root: Path) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, int, int, dict]:
    stimuli_payload = torch.load(stage3_root / "stimulus_batch.pt", weights_only=False)
    stimuli = stimuli_payload["stimuli"]
    targets = stimuli_payload["targets"]
    metadata = pd.DataFrame(stimuli_payload["metadata"])
    feature_start = int(stimuli_payload["feature_start"])
    feature_stop = int(stimuli_payload["feature_stop"])
    batch_config = stimuli_payload["batch_config"]
    return stimuli, targets, metadata, feature_start, feature_stop, batch_config


def load_variant_model(
    checkpoint_path: Path,
    baseline_checkpoint: Path,
) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network = init_network_from_baseline(baseline_checkpoint)
    network.load_state_dict(payload["network"])
    decoder = init_linear_decoder(network)
    decoder.load_state_dict(payload["decoder"])
    network.eval()
    decoder.eval()
    return network, decoder, payload


def evaluate_condition(
    network: torch.nn.Module,
    decoder: torch.nn.Module,
    stimuli: torch.Tensor,
    targets: torch.Tensor,
    feature_start: int,
    feature_stop: int,
    steady_state_seconds: float,
    dt: float,
    edge_indices: list[int],
    base_pooled_state: torch.Tensor,
    base_final_state: torch.Tensor,
    base_task_loss: float,
) -> dict[str, float]:
    original = mask_synapses(network, edge_indices)
    try:
        with torch.no_grad():
            activity = run_network_batch(
                network=network,
                stimuli=stimuli,
                dt=dt,
                steady_state_seconds=steady_state_seconds,
            )
            pooled_state, final_state = pooled_decoder_features(
                activity=activity,
                network=network,
                start=feature_start,
                stop=feature_stop,
            )
            prediction = decoder(pooled_state)
            task_loss = float(F.mse_loss(prediction, targets).detach().cpu())

        return {
            "normalized_state_divergence": normalized_state_divergence(
                pooled_state, base_pooled_state
            ),
            "task_degradation": task_loss - base_task_loss,
            "task_loss_masked": task_loss,
            "cosine_dissimilarity_final_state": cosine_dissimilarity(
                final_state, base_final_state
            ),
        }
    finally:
        restore_synapses(network, original)


def summarize_random_and_targeted(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw_df.groupby(["variant", "ranking", "condition", "k"], as_index=False)
        .agg(
            normalized_state_divergence_mean=("normalized_state_divergence", "mean"),
            normalized_state_divergence_std=("normalized_state_divergence", "std"),
            task_degradation_mean=("task_degradation", "mean"),
            task_degradation_std=("task_degradation", "std"),
            cosine_dissimilarity_mean=("cosine_dissimilarity_final_state", "mean"),
            cosine_dissimilarity_std=("cosine_dissimilarity_final_state", "std"),
        )
        .fillna(0.0)
    )
    return summary


def plot_model_results(summary_df: pd.DataFrame, variant: str, output_path: Path) -> None:
    variant_df = summary_df[summary_df["variant"] == variant]
    rankings = ["usage_proxy", "weight_only"]
    metric_specs = [
        ("normalized_state_divergence_mean", "Normalized State Divergence"),
        ("task_degradation_mean", "Task Loss Degradation"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    color_map = {"targeted": "#d62728", "random": "#7f7f7f"}

    for row_idx, ranking in enumerate(rankings):
        rank_df = variant_df[variant_df["ranking"] == ranking]
        for col_idx, (metric_key, title) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for condition in ["targeted", "random"]:
                cond_df = rank_df[rank_df["condition"] == condition].sort_values("k")
                std_key = metric_key.replace("_mean", "_std")
                ax.plot(
                    cond_df["k"],
                    cond_df[metric_key],
                    marker="o",
                    label=condition,
                    color=color_map[condition],
                )
                ax.fill_between(
                    cond_df["k"],
                    cond_df[metric_key] - cond_df[std_key],
                    cond_df[metric_key] + cond_df[std_key],
                    color=color_map[condition],
                    alpha=0.15,
                )
            ax.set_title(f"{ranking}: {title}")
            ax.set_xlabel("Masked synapse groups (k)")
            ax.grid(alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel(title)
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_gap(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    rankings = ["usage_proxy", "weight_only"]
    colors = {
        "baseline": "#1f77b4",
        "weak": "#ff7f0e",
        "moderate": "#2ca02c",
    }

    for ax, ranking in zip(axes, rankings):
        rank_df = summary_df[summary_df["ranking"] == ranking]
        for variant, color in colors.items():
            targeted = (
                rank_df[(rank_df["variant"] == variant) & (rank_df["condition"] == "targeted")]
                .sort_values("k")
                .set_index("k")
            )
            random = (
                rank_df[(rank_df["variant"] == variant) & (rank_df["condition"] == "random")]
                .sort_values("k")
                .set_index("k")
            )
            gap = (
                targeted["normalized_state_divergence_mean"]
                - random["normalized_state_divergence_mean"]
            )
            ax.plot(gap.index, gap.values, marker="o", color=color, label=variant)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_title(f"{ranking}: targeted - random")
        ax.set_xlabel("Masked synapse groups (k)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Gap in normalized divergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_stage4_report(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_path: Path,
    report_plot_dir: Path,
) -> None:
    lines = [
        "# Stage 4 Fragility Report",
        "",
        "## Setup",
        "",
        "- Models are the canonical Stage 3 MovingEdge variants.",
        "- All evaluations use the exact saved Stage 3 stimulus batch and fixed seed policy.",
        "- State divergence is computed on the pooled central-cell representation used by the Stage 3 decoder.",
        "- Task degradation is the masked task loss minus the unmasked task loss on the same batch.",
        "- Rankings compared: `usage_proxy = |effective_weight| * mean(|source activity|)` and `weight_only = |effective_weight|`.",
        "- Because `syn_strength` is shared across edge groups in flyvis, masking is applied at the trainable synapse-parameter-group level rather than on individual anatomical edges.",
        "",
        "## Targeted vs Random Summary",
        "",
        "| Variant | Ranking | Mean targeted-random divergence gap (k>0) | Mean targeted-random task degradation gap (k>0) | Positive divergence gap at all nonzero k? |",
        "|---|---:|---:|---:|---:|",
    ]

    for variant in ["baseline", "weak", "moderate"]:
        for ranking in ["usage_proxy", "weight_only"]:
            rank_df = summary_df[
                (summary_df["variant"] == variant) & (summary_df["ranking"] == ranking)
            ]
            targeted = (
                rank_df[rank_df["condition"] == "targeted"]
                .sort_values("k")
                .set_index("k")
            )
            random = (
                rank_df[rank_df["condition"] == "random"]
                .sort_values("k")
                .set_index("k")
            )
            divergence_gap = (
                targeted["normalized_state_divergence_mean"]
                - random["normalized_state_divergence_mean"]
            )
            task_gap = targeted["task_degradation_mean"] - random["task_degradation_mean"]
            nonzero_gap = divergence_gap[divergence_gap.index > 0]
            positive_all = bool((nonzero_gap > 0).all())
            lines.append(
                f"| {variant} | {ranking} | {nonzero_gap.mean():.4f} | "
                f"{task_gap[task_gap.index > 0].mean():.4f} | {positive_all} |"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Raw metrics table: `{raw_df.shape[0]}` rows saved under `{raw_df.attrs.get('path', 'results/stage4/movingedge_fragility/fragility_metrics_raw.csv')}`.",
            f"- Aggregated summary table saved under `{summary_df.attrs.get('path', 'results/stage4/movingedge_fragility/fragility_metrics_summary.csv')}`.",
            f"- Per-model plots saved under `{report_plot_dir}`.",
            "",
            "## Interpretation Guardrails",
            "",
            "- A positive targeted-vs-random gap is only a ranking-sensitive fragility signal, not direct proof of a biological tradeoff.",
            "- The result is meaningful only insofar as it survives normalization, matched random controls, and ranking changes.",
            "- This remains a fixed-batch MovingEdge probe rather than a full generalization benchmark.",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    stage3_root = args.stage3_root
    output_root = ensure_dir(args.output_root)
    plots_dir = ensure_dir(output_root / "plots")
    report_dir = ensure_dir(Path("reports"))

    stimuli, targets, metadata, feature_start, feature_stop, batch_config = (
        load_stage3_payload(stage3_root)
    )
    dt = float(batch_config["dt"])
    steady_state_seconds = float(batch_config["steady_state_seconds"])
    ks = [0, 10, 50, 100, 200]

    raw_rows: list[dict[str, float | int | str]] = []
    variant_paths = {
        variant: stage3_root / variant / "checkpoint.pt"
        for variant in ["baseline", "weak", "moderate"]
    }

    probe_network, _, _ = load_variant_model(
        variant_paths["baseline"], args.baseline_checkpoint
    )
    n_synapse_groups = probe_network.edge_params.syn_strength.raw_values.numel()

    random_generator = torch.Generator().manual_seed(args.seed)
    random_masks = {
        k: [
            torch.randperm(n_synapse_groups, generator=random_generator)[:k].tolist()
            for _ in range(args.random_repeats)
        ]
        for k in ks
    }

    for variant, checkpoint_path in variant_paths.items():
        network, decoder, stage3_payload = load_variant_model(
            checkpoint_path, args.baseline_checkpoint
        )

        with torch.no_grad():
            base_activity = run_network_batch(
                network=network,
                stimuli=stimuli,
                dt=dt,
                steady_state_seconds=steady_state_seconds,
            )
            base_pooled_state, base_final_state = pooled_decoder_features(
                activity=base_activity,
                network=network,
                start=feature_start,
                stop=feature_stop,
            )
            base_prediction = decoder(base_pooled_state)
            base_task_loss = float(F.mse_loss(base_prediction, targets).detach().cpu())

        rankings = {
            "usage_proxy": torch.argsort(
                usage_ranking_scores(network, base_activity), descending=True
            )
            .detach()
            .cpu()
            .tolist(),
            "weight_only": torch.argsort(
                weight_only_ranking_scores(network), descending=True
            )
            .detach()
            .cpu()
            .tolist(),
        }

        raw_rows.append(
            {
                "variant": variant,
                "ranking": "base",
                "condition": "base",
                "repeat": 0,
                "k": 0,
                "normalized_state_divergence": 0.0,
                "task_degradation": 0.0,
                "task_loss_masked": base_task_loss,
                "cosine_dissimilarity_final_state": 0.0,
                "lambda_act": float(stage3_payload["lambda_act"]),
            }
        )

        for ranking_name, ranked_edges in rankings.items():
            for k in ks:
                targeted_metrics = evaluate_condition(
                    network=network,
                    decoder=decoder,
                    stimuli=stimuli,
                    targets=targets,
                    feature_start=feature_start,
                    feature_stop=feature_stop,
                    steady_state_seconds=steady_state_seconds,
                    dt=dt,
                    edge_indices=ranked_edges[:k],
                    base_pooled_state=base_pooled_state,
                    base_final_state=base_final_state,
                    base_task_loss=base_task_loss,
                )
                raw_rows.append(
                    {
                        "variant": variant,
                        "ranking": ranking_name,
                        "condition": "targeted",
                        "repeat": 0,
                        "k": k,
                        "lambda_act": float(stage3_payload["lambda_act"]),
                        **targeted_metrics,
                    }
                )

                for repeat, edge_indices in enumerate(random_masks[k], start=1):
                    random_metrics = evaluate_condition(
                        network=network,
                        decoder=decoder,
                        stimuli=stimuli,
                        targets=targets,
                        feature_start=feature_start,
                        feature_stop=feature_stop,
                        steady_state_seconds=steady_state_seconds,
                        dt=dt,
                        edge_indices=edge_indices,
                        base_pooled_state=base_pooled_state,
                        base_final_state=base_final_state,
                        base_task_loss=base_task_loss,
                    )
                    raw_rows.append(
                        {
                            "variant": variant,
                            "ranking": ranking_name,
                            "condition": "random",
                            "repeat": repeat,
                            "k": k,
                            "lambda_act": float(stage3_payload["lambda_act"]),
                            **random_metrics,
                        }
                    )

    raw_df = pd.DataFrame(raw_rows)
    raw_path = output_root / "fragility_metrics_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    raw_df.attrs["path"] = str(raw_path)

    summary_df = summarize_random_and_targeted(
        raw_df[raw_df["condition"].isin(["targeted", "random"])]
    )
    summary_path = output_root / "fragility_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_df.attrs["path"] = str(summary_path)

    for variant in ["baseline", "weak", "moderate"]:
        plot_model_results(summary_df, variant, plots_dir / f"{variant}_fragility.png")
    plot_combined_gap(summary_df, plots_dir / "combined_targeted_random_gap.png")

    write_json(
        output_root / "stage4_summary.json",
        {
            "seed": args.seed,
            "random_repeats": args.random_repeats,
            "k_values": ks,
            "ablation_unit": "shared synapse-strength parameter groups",
            "stage3_root": str(stage3_root),
            "plots": {
                "baseline": str(plots_dir / "baseline_fragility.png"),
                "weak": str(plots_dir / "weak_fragility.png"),
                "moderate": str(plots_dir / "moderate_fragility.png"),
                "combined": str(plots_dir / "combined_targeted_random_gap.png"),
            },
            "summary_table": str(summary_path),
            "raw_table": str(raw_path),
        },
    )

    write_stage4_report(
        summary_df=summary_df,
        raw_df=raw_df,
        output_path=report_dir / "stage4_fragility_report.md",
        report_plot_dir=plots_dir,
    )


if __name__ == "__main__":
    main()
