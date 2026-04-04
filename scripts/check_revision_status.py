#!/usr/bin/env python

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PHASE_PATHS = {
    "phase1_steps5": REPO_ROOT / "results/revision_results/revision_phase1_random_init/steps_5",
    "phase1_steps10": REPO_ROOT / "results/revision_results/revision_phase1_random_init/steps_10",
    "phase2_steps5": REPO_ROOT / "results/revision_results/revision_phase2_degree_preserving/steps_5",
    "phase2_steps10": REPO_ROOT / "results/revision_results/revision_phase2_degree_preserving/steps_10",
}


def seed_status(root: Path, expected_models: list[str]) -> dict[int, str]:
    statuses: dict[int, str] = {}
    if not root.exists():
        return statuses
    for seed_dir in sorted(root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_")[1])
        except Exception:
            continue
        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            statuses[seed] = "partial"
            continue
        data = json.loads(summary_path.read_text())
        models_ok = []
        finite_ok = []
        for model in expected_models:
            model_info = data.get(model)
            if model_info is None:
                models_ok.append(False)
                finite_ok.append(False)
                continue
            models_ok.append(True)
            finite_ok.append(bool(model_info.get("remained_finite", False)))
        statuses[seed] = "done" if all(models_ok) and all(finite_ok) else "partial"
    return statuses


def print_phase(label: str, root: Path, expected_models: list[str]) -> None:
    print(f"{label}:")
    print(f"  path: {root}")
    if not root.exists():
        print("  status: missing")
        return
    statuses = seed_status(root, expected_models)
    if not statuses:
        print("  status: empty")
        return
    for seed, status in statuses.items():
        print(f"  seed_{seed}: {status}")


def print_running_processes() -> None:
    print("running_processes:")
    try:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                "ps -eo pid,etimes,pcpu,pmem,args --sort=-etimes | rg 'stage3_train_connectome_vs_random_matched_steps_fromscratch|stage3_train_connectome_vs_degreepreserving_matched_steps|build_degree_preserving_random_mask|aggregate_revision_controls' || true",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            print("  none visible from current process namespace")
            return
        for line in lines:
            print(f"  {line}")
    except Exception as exc:
        print(f"  unavailable: {exc}")


def main() -> None:
    print_phase("phase1_steps5", PHASE_PATHS["phase1_steps5"], ["connectome", "random"])
    print_phase("phase1_steps10", PHASE_PATHS["phase1_steps10"], ["connectome", "random"])
    print_phase("phase2_steps5", PHASE_PATHS["phase2_steps5"], ["connectome", "degreepres"])
    print_phase("phase2_steps10", PHASE_PATHS["phase2_steps10"], ["connectome", "degreepres"])

    missing = []
    for path in [
        REPO_ROOT / "data/metadata/degree_preserving_random_mask.summary.json",
        REPO_ROOT / "results/revision_results/revision_phase1_random_init/steps_5/summary.json",
        REPO_ROOT / "results/revision_results/revision_phase1_random_init/steps_10/summary.json",
        REPO_ROOT / "results/revision_results/revision_phase2_degree_preserving/steps_5/summary.json",
        REPO_ROOT / "results/revision_results/revision_phase2_degree_preserving/steps_10/summary.json",
        REPO_ROOT / "results/revision_results/revision_initial_activity/initial_activity_metrics.csv",
        REPO_ROOT / "results/revision_results/revision_degpres_ensemble/steps_5/aggregated_metrics.csv",
    ]:
        if not path.exists():
            missing.append(path)

    print("missing_outputs:")
    if not missing:
        print("  none")
    else:
        for path in missing:
            print(f"  {path}")

    print_running_processes()


if __name__ == "__main__":
    main()
