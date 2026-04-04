# Reproducibility Guide

## 1. External Requirements

- Python environment from [`environment/requirements.txt`](../environment/requirements.txt)
- Upstream `flyvis` package
- Access to the MovingEdge task assets used by `flyvis`
- Optional baseline checkpoint if you want to reproduce the original confounded comparison

Set external paths if they are not installed globally:

```bash
export FLYVIS_REPO_ROOT=/path/to/flyvis
export FLYVIS_BASELINE_CHECKPOINT=/path/to/chkpt_00000
```

## 2. Recommended Reproduction Order

### Main comparison

```bash
bash scripts/run_main_controls.sh
```

### Revision controls

```bash
bash scripts/run_revision_controls.sh
```

### Lightweight figure regeneration

```bash
bash scripts/make_figures.sh
```

## 3. What Is Bundled vs Regenerated

Bundled:
- final PNG/PDF figures
- lightweight CSV tables
- packaged markdown summaries

Regenerated on demand:
- random and degree-preserving masks
- fresh matched-step runs
- updated aggregate summaries

Excluded from the bundle:
- heavy checkpoints
- saved activity tensors
- ensemble mask payloads

Some heavier mechanism plots therefore remain archival outputs rather than lightweight rerunnable
targets inside this release.
