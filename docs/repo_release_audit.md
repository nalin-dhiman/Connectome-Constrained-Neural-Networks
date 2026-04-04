# Repository Release Audit

This audit was prepared while creating the clean release repository in
[`github_release_repo/`](/home/ub/Downloads/connectome_ann_blueprint/github_release_repo).
The goal was to keep the final control-study code, figures, and lightweight result
tables while excluding manuscript material and large intermediate artifacts.

## A. Core Validated Experiments

KEEP
- `scripts/stage3_train_connectome_vs_random_matched_steps.py`
- `scripts/stage3_aggregate_matched_steps.py`
- `scripts/stage3_train_connectome_vs_random_matched_steps_10.py`
- `scripts/stage3_aggregate_matched_steps_10.py`
- `results/stage3_connectome_vs_random_matched_steps/`
- `results/stage3_connectome_vs_random_matched_steps_10/`
- `results/mechanism/` lightweight CSV summaries and figure-ready outputs

## B. Revision-Control Experiments

KEEP
- `scripts/stage3_train_connectome_vs_random_matched_steps_fromscratch.py`
- `scripts/build_degree_preserving_random_mask.py`
- `scripts/build_degree_preserving_random_mask_ensemble.py`
- `scripts/stage3_train_connectome_vs_degreepreserving_matched_steps.py`
- `scripts/stage3_train_connectome_vs_degreepreserving_ensemble.py`
- `scripts/aggregate_revision_controls.py`
- `scripts/aggregate_degreepres_ensemble.py`
- `scripts/analyze_initial_activity_scale.py`
- `results/revision_phase1_random_init/`
- `results/revision_phase2_degree_preserving/`
- `results/revision_initial_activity/`
- `results/revision_degpres_ensemble/` lightweight summaries and CSV aggregates

## C. Plotting and Analysis Scripts

KEEP
- `scripts/revision_make_additional_plots.py`
- `scripts/analyze_node_activity_mechanism.py`
- `scripts/analyze_edge_usage_mechanism.py`
- `scripts/analyze_temporal_stability_mechanism.py`
- `scripts/aggregate_mechanism_results.py`

## D. CSV Tables Used in Figures or Paper Summaries

KEEP
- `paper_artifacts/tables/main_metrics.csv`
- `submission_package/tables/controls_summary.csv`
- `submission_package/tables/ensemble_variability.csv`
- `submission_package/tables/initial_activity.csv`
- `submission_package/tables/mechanism_summary.csv`
- `submission_package/tables/negative_controls.csv`

## E. Large Tensors, Checkpoints, and Heavy Binary Artifacts

EXCLUDE
- `results/stage3_connectome_vs_random_matched_steps_saved/*.pt`
- `results/revision_degpres_ensemble/masks/*.pt`
- `results/random_mask_selfloop.pt`
- `results/degree_preserving_random_mask.pt`
- training checkpoints, saved activities, and large tensors not needed for a lightweight release

Reason
- These files add substantial size and are not required to inspect the corrected claim,
  reproduce paper tables, or understand the control ladder.

## F. Paper / Submission Artifacts

EXCLUDE
- all `*.tex`
- `submission_package/`
- manuscript-only LaTeX assets

Reason
- The release repository is code-and-results only, not a manuscript repository.

## G. Archive / Exploratory Material

ARCHIVE
- weak-regularization experiments
- fragility experiments
- failed structured null controls
- exploratory reports and scripts not needed for the main corrected claim

Paths archived into the release bundle
- `archive/exploratory/scripts/`
- `archive/exploratory/reports/`

## Release Policy Summary

- KEEP: validated main comparisons, corrected revision controls, lightweight figures, CSV tables, and helper scripts
- ARCHIVE: exploratory or inconclusive control branches kept for transparency but not placed on the main path
- EXCLUDE: LaTeX, manuscript bundles, logs, caches, large masks, checkpoints, and saved-state tensors
