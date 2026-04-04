#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ITERATION_POINTS = (1, 3, 5)
CONDITIONS = (
    "connectome_baseline",
    "connectome_weak",
    "random_baseline",
    "random_weak",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate matched-step connectome-vs-random weak comparison."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/stage3_connectome_vs_random_matched_steps_weak"),
    )
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path(
            "results/stage3_connectome_vs_random_matched_steps_weak/aggregated_metrics.csv"
        ),
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path(
            "results/stage3_connectome_vs_random_matched_steps_weak/weak_effect_deltas.csv"
        ),
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path(
            "results/stage3_connectome_vs_random_matched_steps_weak/bootstrap_iter5.csv"
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/stage3_connectome_vs_random_matched_steps_weak.md"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    return parser.parse_args()


def seed_dirs(input_root: Path) -> list[Path]:
    dirs = sorted(path for path in input_root.glob("seed_*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No seed directories found under {input_root}")
    return dirs


def load_curve(seed_dir: Path, condition: str) -> pd.DataFrame:
    path = seed_dir / f"{condition}_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing curve file: {path}")
    return pd.read_csv(path)


def load_summary(seed_dir: Path) -> dict:
    path = seed_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text())


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> float:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def bootstrap_ci(values: np.ndarray, resamples: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = float(np.mean(sample))
    return (
        float(np.mean(values)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "n/a"
    if n == 1 or np.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    args = parse_args()
    for output_path in (
        args.aggregated_csv,
        args.delta_csv,
        args.bootstrap_csv,
        args.report_path,
    ):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    raw_rows: list[dict] = []
    delta_rows: list[dict] = []
    summary_by_seed: dict[int, dict] = {}

    for seed_dir in seed_dirs(args.input_root):
        seed = int(seed_dir.name.split("_")[-1])
        summary = load_summary(seed_dir)
        summary_by_seed[seed] = summary
        curves = {condition: load_curve(seed_dir, condition) for condition in CONDITIONS}

        for condition in CONDITIONS:
            curve = curves[condition]
            row = {
                "seed": seed,
                "condition": condition,
                "all_finite": bool(curve["finite"].all()),
            }
            for iteration in ITERATION_POINTS:
                row[f"loss_iter_{iteration}"] = value_at_iteration(
                    curve, iteration, "task_loss"
                )
                row[f"activity_iter_{iteration}"] = value_at_iteration(
                    curve, iteration, "activity_abs_mean"
                )
                row[f"elapsed_iter_{iteration}"] = value_at_iteration(
                    curve, iteration, "elapsed_sec"
                )
            raw_rows.append(row)

        connectome_base = curves["connectome_baseline"]
        connectome_weak = curves["connectome_weak"]
        random_base = curves["random_baseline"]
        random_weak = curves["random_weak"]

        delta_loss_connectome = value_at_iteration(connectome_weak, 5, "task_loss") - value_at_iteration(
            connectome_base, 5, "task_loss"
        )
        delta_activity_connectome = value_at_iteration(
            connectome_base, 5, "activity_abs_mean"
        ) - value_at_iteration(connectome_weak, 5, "activity_abs_mean")
        delta_loss_random = value_at_iteration(random_weak, 5, "task_loss") - value_at_iteration(
            random_base, 5, "task_loss"
        )
        delta_activity_random = value_at_iteration(
            random_base, 5, "activity_abs_mean"
        ) - value_at_iteration(random_weak, 5, "activity_abs_mean")

        delta_rows.append(
            {
                "seed": seed,
                "delta_loss_connectome": delta_loss_connectome,
                "delta_activity_connectome": delta_activity_connectome,
                "delta_loss_random": delta_loss_random,
                "delta_activity_random": delta_activity_random,
                "delta_activity_advantage": delta_activity_connectome - delta_activity_random,
                "delta_loss_advantage": delta_loss_connectome - delta_loss_random,
            }
        )

    raw_df = pd.DataFrame(raw_rows).sort_values(["condition", "seed"]).reset_index(drop=True)
    delta_df = pd.DataFrame(delta_rows).sort_values("seed").reset_index(drop=True)

    aggregated_rows: list[dict] = []
    for condition in CONDITIONS:
        condition_df = raw_df.loc[raw_df["condition"].eq(condition)].reset_index(drop=True)
        row = {"condition": condition, "all_finite": bool(condition_df["all_finite"].all())}
        for prefix in ("loss", "activity", "elapsed"):
            for iteration in ITERATION_POINTS:
                column = f"{prefix}_iter_{iteration}"
                values = condition_df[column].to_numpy(dtype=float)
                row[f"{column}_mean"] = float(np.nanmean(values))
                row[f"{column}_std"] = float(np.nanstd(values, ddof=1))
                row[f"{column}_n"] = int(np.sum(~np.isnan(values)))
        aggregated_rows.append(row)

    aggregated_df = pd.DataFrame(aggregated_rows)

    bootstrap_rows = []
    for metric in (
        "delta_loss_connectome",
        "delta_activity_connectome",
        "delta_loss_random",
        "delta_activity_random",
        "delta_activity_advantage",
        "delta_loss_advantage",
    ):
        values = delta_df[metric].to_numpy(dtype=float)
        mean, ci_low, ci_high = bootstrap_ci(values, args.bootstrap_resamples)
        bootstrap_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    args.aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(args.input_root / "raw_metrics.csv", index=False)
    aggregated_df.to_csv(args.aggregated_csv, index=False)
    delta_df.to_csv(args.delta_csv, index=False)
    bootstrap_df.to_csv(args.bootstrap_csv, index=False)

    agg = {row["condition"]: row for row in aggregated_df.to_dict(orient="records")}
    boot = {row["metric"]: row for row in bootstrap_df.to_dict(orient="records")}
    first_seed = min(summary_by_seed)
    first_condition = summary_by_seed[first_seed]["conditions"]["connectome_baseline"]
    model_table = {
        "nodes": first_condition["n_nodes"],
        "edges": first_condition["n_edges"],
        "self_loops": first_condition["self_loops"],
        "free_parameters": first_condition["free_parameters"],
        "fixed_parameters": first_condition["fixed_parameters"],
    }

    all_finite = bool(raw_df["all_finite"].all())
    activity_advantage = float(delta_df["delta_activity_advantage"].mean())
    loss_advantage = float(delta_df["delta_loss_advantage"].mean())

    if not all_finite:
        conclusion = "inconclusive"
    elif abs(activity_advantage) < 1e-6 and abs(loss_advantage) < 1e-6:
        conclusion = "no meaningful difference"
    elif activity_advantage > 0 and loss_advantage <= 0:
        conclusion = "connectome benefits more"
    elif activity_advantage <= 0 and loss_advantage > 0:
        conclusion = "random benefits more"
    elif activity_advantage <= 0:
        conclusion = "random benefits more"
    elif loss_advantage > 0:
        conclusion = "unstable"
    else:
        conclusion = "connectome benefits more"

    report_lines = [
        "# Stage 3 Connectome vs Random Matched Steps Weak",
        "",
        "Scope:",
        "- Matched-step comparison.",
        "- Same canonical Stage 3 setup.",
        "- Seeds [0, 1, 2].",
        "- Fixed 5 iterations.",
        "- Four conditions only.",
        "- Not a benchmark.",
        "",
        "Model/control table:",
        "",
        "| Graph | Nodes | Edges | Self-loops | Free params | Fixed params |",
        "|---|---:|---:|---:|---:|---:|",
        f"| connectome | {model_table['nodes']} | {model_table['edges']} | {model_table['self_loops']} | {model_table['free_parameters']} | {model_table['fixed_parameters']} |",
        f"| random | {model_table['nodes']} | {model_table['edges']} | {model_table['self_loops']} | {model_table['free_parameters']} | {model_table['fixed_parameters']} |",
        "",
        "Table A: loss",
        "",
        "| Condition | Iter 1 | Iter 3 | Iter 5 |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        report_lines.append(
            f"| {condition} | {fmt(agg[condition]['loss_iter_1_mean'], agg[condition]['loss_iter_1_std'], agg[condition]['loss_iter_1_n'])} | {fmt(agg[condition]['loss_iter_3_mean'], agg[condition]['loss_iter_3_std'], agg[condition]['loss_iter_3_n'])} | {fmt(agg[condition]['loss_iter_5_mean'], agg[condition]['loss_iter_5_std'], agg[condition]['loss_iter_5_n'])} |"
        )

    report_lines.extend(
        [
            "",
            "Table B: activity",
            "",
            "| Condition | Iter 1 | Iter 3 | Iter 5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        report_lines.append(
            f"| {condition} | {fmt(agg[condition]['activity_iter_1_mean'], agg[condition]['activity_iter_1_std'], agg[condition]['activity_iter_1_n'])} | {fmt(agg[condition]['activity_iter_3_mean'], agg[condition]['activity_iter_3_std'], agg[condition]['activity_iter_3_n'])} | {fmt(agg[condition]['activity_iter_5_mean'], agg[condition]['activity_iter_5_std'], agg[condition]['activity_iter_5_n'])} |"
        )

    report_lines.extend(
        [
            "",
            "Table C: elapsed time",
            "",
            "| Condition | Time at iter 5 |",
            "|---|---:|",
        ]
    )
    for condition in CONDITIONS:
        report_lines.append(
            f"| {condition} | {fmt(agg[condition]['elapsed_iter_5_mean'], agg[condition]['elapsed_iter_5_std'], agg[condition]['elapsed_iter_5_n'])} |"
        )

    report_lines.extend(
        [
            "",
            "Weak-effect comparison at iter 5:",
            "",
            f"- connectome baseline loss/activity: {agg['connectome_baseline']['loss_iter_5_mean']:.4f} / {agg['connectome_baseline']['activity_iter_5_mean']:.4f}",
            f"- connectome weak loss/activity: {agg['connectome_weak']['loss_iter_5_mean']:.4f} / {agg['connectome_weak']['activity_iter_5_mean']:.4f}",
            f"- delta_activity_connectome: {delta_df['delta_activity_connectome'].mean():.4f} +/- {delta_df['delta_activity_connectome'].std(ddof=1):.4f}",
            f"- delta_loss_connectome: {delta_df['delta_loss_connectome'].mean():.4f} +/- {delta_df['delta_loss_connectome'].std(ddof=1):.4f}",
            "",
            f"- random baseline loss/activity: {agg['random_baseline']['loss_iter_5_mean']:.4f} / {agg['random_baseline']['activity_iter_5_mean']:.4f}",
            f"- random weak loss/activity: {agg['random_weak']['loss_iter_5_mean']:.4f} / {agg['random_weak']['activity_iter_5_mean']:.4f}",
            f"- delta_activity_random: {delta_df['delta_activity_random'].mean():.4f} +/- {delta_df['delta_activity_random'].std(ddof=1):.4f}",
            f"- delta_loss_random: {delta_df['delta_loss_random'].mean():.4f} +/- {delta_df['delta_loss_random'].std(ddof=1):.4f}",
            "",
            "Comparative quantities:",
            "",
            f"- delta_activity_advantage: {delta_df['delta_activity_advantage'].mean():.4f} +/- {delta_df['delta_activity_advantage'].std(ddof=1):.4f}",
            f"- delta_loss_advantage: {delta_df['delta_loss_advantage'].mean():.4f} +/- {delta_df['delta_loss_advantage'].std(ddof=1):.4f}",
            "",
            "Bootstrap 95% CI for iter-5 deltas:",
            "",
            f"- delta_activity_connectome: mean={boot['delta_activity_connectome']['mean']:.4f}, CI=[{boot['delta_activity_connectome']['ci_low']:.4f}, {boot['delta_activity_connectome']['ci_high']:.4f}]",
            f"- delta_loss_connectome: mean={boot['delta_loss_connectome']['mean']:.4f}, CI=[{boot['delta_loss_connectome']['ci_low']:.4f}, {boot['delta_loss_connectome']['ci_high']:.4f}]",
            f"- delta_activity_random: mean={boot['delta_activity_random']['mean']:.4f}, CI=[{boot['delta_activity_random']['ci_low']:.4f}, {boot['delta_activity_random']['ci_high']:.4f}]",
            f"- delta_loss_random: mean={boot['delta_loss_random']['mean']:.4f}, CI=[{boot['delta_loss_random']['ci_low']:.4f}, {boot['delta_loss_random']['ci_high']:.4f}]",
            f"- delta_activity_advantage: mean={boot['delta_activity_advantage']['mean']:.4f}, CI=[{boot['delta_activity_advantage']['ci_low']:.4f}, {boot['delta_activity_advantage']['ci_high']:.4f}]",
            f"- delta_loss_advantage: mean={boot['delta_loss_advantage']['mean']:.4f}, CI=[{boot['delta_loss_advantage']['ci_low']:.4f}, {boot['delta_loss_advantage']['ci_high']:.4f}]",
            "",
            "Stability notes:",
            f"- all runs finite: {all_finite}",
            f"- conclusion label: {conclusion}",
        ]
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report_lines))

    print(aggregated_df.to_string(index=False))
    print(delta_df.to_string(index=False))
    print(bootstrap_df.to_string(index=False))
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
