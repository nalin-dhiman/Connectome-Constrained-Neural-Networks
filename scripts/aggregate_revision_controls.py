#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from revision_control_utils import bootstrap_mean_ci, format_mean_std, load_curve, stats_row, write_markdown


PHASES = {
    "original": {
        "steps_5_root": Path("results/main_results/stage3_connectome_vs_random_matched_steps"),
        "steps_10_root": Path("results/main_results/stage3_connectome_vs_random_matched_steps_10"),
        "left": ("connectome", "connectome_curve.csv"),
        "right": ("random", "random_curve.csv"),
        "title": "Original checkpoint-initialized comparison",
    },
    "phase1": {
        "steps_5_root": Path("results/revision_results/revision_phase1_random_init/steps_5"),
        "steps_10_root": Path("results/revision_results/revision_phase1_random_init/steps_10"),
        "left": ("connectome", "connectome_curve.csv"),
        "right": ("random", "random_curve.csv"),
        "title": "Phase 1 random-initialization control",
    },
    "phase2": {
        "steps_5_root": Path("results/revision_results/revision_phase2_degree_preserving/steps_5"),
        "steps_10_root": Path("results/revision_results/revision_phase2_degree_preserving/steps_10"),
        "left": ("connectome", "connectome_curve.csv"),
        "right": ("degreepres", "degreepres_curve.csv"),
        "title": "Phase 2 degree-preserving control",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate revision-control comparisons and write a summary report."
    )
    parser.add_argument(
        "--original-steps5-root",
        type=Path,
        default=PHASES["original"]["steps_5_root"],
    )
    parser.add_argument(
        "--original-steps10-root",
        type=Path,
        default=PHASES["original"]["steps_10_root"],
    )
    parser.add_argument(
        "--phase1-steps5-root",
        type=Path,
        default=PHASES["phase1"]["steps_5_root"],
    )
    parser.add_argument(
        "--phase1-steps10-root",
        type=Path,
        default=PHASES["phase1"]["steps_10_root"],
    )
    parser.add_argument(
        "--phase2-steps5-root",
        type=Path,
        default=PHASES["phase2"]["steps_5_root"],
    )
    parser.add_argument(
        "--phase2-steps10-root",
        type=Path,
        default=PHASES["phase2"]["steps_10_root"],
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/generated/revision_controls_summary.md"),
    )
    parser.add_argument(
        "--phase1-report-path",
        type=Path,
        default=Path("docs/generated/revision_phase1_random_init.md"),
    )
    parser.add_argument(
        "--phase2-report-path",
        type=Path,
        default=Path("docs/generated/revision_phase2_degree_preserving.md"),
    )
    return parser.parse_args()


def seed_dirs(root: Path) -> list[Path]:
    dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No seed directories found under {root}")
    return dirs


def phase_root_complete(root: Path, expected_curve_files: tuple[str, ...]) -> bool:
    if not root.exists():
        return False
    seed_paths = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not seed_paths:
        return False
    for seed_dir in seed_paths:
        if not (seed_dir / "summary.json").exists():
            return False
        for curve_file in expected_curve_files:
            if not (seed_dir / curve_file).exists():
                return False
    return True


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> float:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def summarize_phase_step(root: Path, left_file: str, right_file: str, iteration: int) -> dict[str, float]:
    left_rows = []
    right_rows = []
    delta_rows = []
    for seed_dir in seed_dirs(root):
        seed = int(seed_dir.name.split("_")[-1])
        left_curve = load_curve(seed_dir, left_file)
        right_curve = load_curve(seed_dir, right_file)
        left_rows.append(
            {
                "seed": seed,
                "loss": value_at_iteration(left_curve, iteration, "task_loss"),
                "activity": value_at_iteration(left_curve, iteration, "activity_abs_mean"),
                "elapsed": value_at_iteration(left_curve, iteration, "elapsed_sec"),
                "finite": bool(left_curve["finite"].all()),
            }
        )
        right_rows.append(
            {
                "seed": seed,
                "loss": value_at_iteration(right_curve, iteration, "task_loss"),
                "activity": value_at_iteration(right_curve, iteration, "activity_abs_mean"),
                "elapsed": value_at_iteration(right_curve, iteration, "elapsed_sec"),
                "finite": bool(right_curve["finite"].all()),
            }
        )
        delta_rows.append(
            {
                "seed": seed,
                "delta_loss": right_rows[-1]["loss"] - left_rows[-1]["loss"],
                "delta_activity": right_rows[-1]["activity"] - left_rows[-1]["activity"],
                "delta_elapsed": right_rows[-1]["elapsed"] - left_rows[-1]["elapsed"],
            }
        )

    left_df = pd.DataFrame(left_rows).sort_values("seed")
    right_df = pd.DataFrame(right_rows).sort_values("seed")
    delta_df = pd.DataFrame(delta_rows).sort_values("seed")

    summary = {
        "left_loss": stats_row(left_df["loss"].to_numpy()),
        "right_loss": stats_row(right_df["loss"].to_numpy()),
        "left_activity": stats_row(left_df["activity"].to_numpy()),
        "right_activity": stats_row(right_df["activity"].to_numpy()),
        "left_elapsed": stats_row(left_df["elapsed"].to_numpy()),
        "right_elapsed": stats_row(right_df["elapsed"].to_numpy()),
        "delta_loss_ci": bootstrap_mean_ci(delta_df["delta_loss"].to_numpy()),
        "delta_activity_ci": bootstrap_mean_ci(delta_df["delta_activity"].to_numpy()),
        "delta_elapsed_ci": bootstrap_mean_ci(delta_df["delta_elapsed"].to_numpy()),
        "all_finite": bool(left_df["finite"].all() and right_df["finite"].all()),
    }
    return summary


def phase_report_lines(phase_title: str, step5: dict | None, step10: dict | None, right_label: str) -> list[str]:
    lines = [
        f"# {phase_title}",
        "",
        "This report summarizes the revision-control comparison only. Positive deltas are defined as "
        f"`{right_label} - connectome`, so larger positive values mean the control is worse than the connectome.",
        "",
    ]
    for label, summary, iteration in [("5-step", step5, 5), ("10-step", step10, 10)]:
        if summary is None:
            continue
        lines.extend(
            [
                f"## {label}",
                "",
                "| Metric | Connectome | Control |",
                "|---|---:|---:|",
                f"| Loss @{iteration} | {format_mean_std(*summary['left_loss'])} | {format_mean_std(*summary['right_loss'])} |",
                f"| Activity @{iteration} | {format_mean_std(*summary['left_activity'])} | {format_mean_std(*summary['right_activity'])} |",
                f"| Elapsed @{iteration} | {format_mean_std(*summary['left_elapsed'])} | {format_mean_std(*summary['right_elapsed'])} |",
                "",
                f"- delta_loss @{iteration}: mean={summary['delta_loss_ci'][0]:.4f}, 95% CI=[{summary['delta_loss_ci'][1]:.4f}, {summary['delta_loss_ci'][2]:.4f}]",
                f"- delta_activity @{iteration}: mean={summary['delta_activity_ci'][0]:.4f}, 95% CI=[{summary['delta_activity_ci'][1]:.4f}, {summary['delta_activity_ci'][2]:.4f}]",
                f"- delta_elapsed @{iteration}: mean={summary['delta_elapsed_ci'][0]:.4f}, 95% CI=[{summary['delta_elapsed_ci'][1]:.4f}, {summary['delta_elapsed_ci'][2]:.4f}]",
                f"- all runs finite: {summary['all_finite']}",
                "",
            ]
        )
    return lines


def main() -> None:
    args = parse_args()
    for path in [args.report_path, args.phase1_report_path, args.phase2_report_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {path}")

    phases = {
        "original": {**PHASES["original"], "steps_5_root": args.original_steps5_root, "steps_10_root": args.original_steps10_root},
        "phase1": {**PHASES["phase1"], "steps_5_root": args.phase1_steps5_root, "steps_10_root": args.phase1_steps10_root},
        "phase2": {**PHASES["phase2"], "steps_5_root": args.phase2_steps5_root, "steps_10_root": args.phase2_steps10_root},
    }

    phase_summaries = {}
    for phase_key, spec in phases.items():
        step5 = None
        if phase_root_complete(spec["steps_5_root"], (spec["left"][1], spec["right"][1])):
            step5 = summarize_phase_step(
                spec["steps_5_root"],
                spec["left"][1],
                spec["right"][1],
                5,
            )
        step10 = None
        if phase_root_complete(spec["steps_10_root"], (spec["left"][1], spec["right"][1])):
            step10 = summarize_phase_step(
                spec["steps_10_root"],
                spec["left"][1],
                spec["right"][1],
                10,
            )
        phase_summaries[phase_key] = {"step5": step5, "step10": step10}

    if phase_summaries["phase1"]["step5"] is not None or phase_summaries["phase1"]["step10"] is not None:
        write_markdown(
            args.phase1_report_path,
            phase_report_lines(
                phases["phase1"]["title"],
                phase_summaries["phase1"]["step5"],
                phase_summaries["phase1"]["step10"],
                "random",
            ),
        )
    if phase_summaries["phase2"]["step5"] is not None or phase_summaries["phase2"]["step10"] is not None:
        write_markdown(
            args.phase2_report_path,
            phase_report_lines(
                phases["phase2"]["title"],
                phase_summaries["phase2"]["step5"],
                phase_summaries["phase2"]["step10"],
                "degree-preserving random",
            ),
        )

    orig5 = phase_summaries["original"]["step5"]
    p15 = phase_summaries["phase1"]["step5"]
    p25 = phase_summaries["phase2"]["step5"]
    p110 = phase_summaries["phase1"]["step10"]
    p210 = phase_summaries["phase2"]["step10"]

    lines = [
        "# Revision Controls Summary",
        "",
        "Question addressed:",
        "1. How much of the original effect survives random initialization?",
        "2. How much survives a degree-preserving null?",
        "3. Was spectral / initial-activity calibration required?",
        "",
        "Phase 3 status:",
        "- Not run yet. It was deferred until after Phases 1 and 2 because the main reviewer confounds are initialization and degree sequence.",
        "",
    ]
    lines.extend(
        [
            "## Available comparisons",
            "",
            "| Comparison | Step-5 delta loss | Step-5 delta activity | Step-5 delta elapsed |",
            "|---|---:|---:|---:|",
            f"| Original checkpoint-based | {orig5['delta_loss_ci'][0]:.4f} | {orig5['delta_activity_ci'][0]:.4f} | {orig5['delta_elapsed_ci'][0]:.4f} |",
        ]
    )
    if p15 is not None:
        lines.append(
            f"| Phase 1 random-init | {p15['delta_loss_ci'][0]:.4f} | {p15['delta_activity_ci'][0]:.4f} | {p15['delta_elapsed_ci'][0]:.4f} |"
        )
    else:
        lines.append("| Phase 1 random-init | pending | pending | pending |")
    if p25 is not None:
        lines.append(
            f"| Phase 2 degree-preserving | {p25['delta_loss_ci'][0]:.4f} | {p25['delta_activity_ci'][0]:.4f} | {p25['delta_elapsed_ci'][0]:.4f} |"
        )
    else:
        lines.append("| Phase 2 degree-preserving | pending | pending | pending |")
    lines.append("")
    if p110 is not None and p210 is not None:
        lines.extend(
            [
                "## Step-10 comparison",
                "",
                "| Comparison | Delta loss | Delta activity | Delta elapsed |",
                "|---|---:|---:|---:|",
                f"| Phase 1 random-init | {p110['delta_loss_ci'][0]:.4f} | {p110['delta_activity_ci'][0]:.4f} | {p110['delta_elapsed_ci'][0]:.4f} |",
                f"| Phase 2 degree-preserving | {p210['delta_loss_ci'][0]:.4f} | {p210['delta_activity_ci'][0]:.4f} | {p210['delta_elapsed_ci'][0]:.4f} |",
                "",
            ]
        )
    elif p110 is not None or p210 is not None:
        lines.extend(
            [
                "## Step-10 comparison",
                "",
                "| Comparison | Delta loss | Delta activity | Delta elapsed |",
                "|---|---:|---:|---:|",
            ]
        )
        if p110 is not None:
            lines.append(
                f"| Phase 1 random-init | {p110['delta_loss_ci'][0]:.4f} | {p110['delta_activity_ci'][0]:.4f} | {p110['delta_elapsed_ci'][0]:.4f} |"
            )
        else:
            lines.append("| Phase 1 random-init | pending | pending | pending |")
        if p210 is not None:
            lines.append(
                f"| Phase 2 degree-preserving | {p210['delta_loss_ci'][0]:.4f} | {p210['delta_activity_ci'][0]:.4f} | {p210['delta_elapsed_ci'][0]:.4f} |"
            )
        else:
            lines.append("| Phase 2 degree-preserving | pending | pending | pending |")
        lines.append("")

    phase1_survives = p15 is not None and (
        p15["delta_loss_ci"][1] > 0 and p15["delta_activity_ci"][1] > 0 and p15["delta_elapsed_ci"][1] > 0
    )
    phase2_survives = p25 is not None and (
        p25["delta_loss_ci"][1] > 0 and p25["delta_activity_ci"][1] > 0 and p25["delta_elapsed_ci"][1] > 0
    )
    if phase1_survives and phase2_survives:
        conclusion = "topology claim survives strongly"
    elif p15 is not None and p25 is not None and p15["delta_activity_ci"][1] > 0 and p25["delta_activity_ci"][1] > 0:
        conclusion = "topology claim survives partially"
    elif (p15 is not None and p15["delta_activity_ci"][0] <= 0) or (
        p25 is not None and p25["delta_activity_ci"][0] <= 0
    ):
        conclusion = "topology claim weakens materially"
    else:
        conclusion = "inconclusive"

    lines.extend(
        [
            "## Safe interpretation",
            "",
            f"- Overall label: {conclusion}",
            "- If Phase 1 remains positive, the original result is not purely a checkpoint artifact.",
            "- If Phase 2 remains positive, the effect is not explained solely by node-wise degree sequence.",
            "- Phase 3 calibration should only be run if large iteration-1 mismatches remain after these two revisions.",
        ]
    )
    write_markdown(args.report_path, lines)


if __name__ == "__main__":
    main()
