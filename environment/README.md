# Environment Notes

This release intentionally keeps the study-specific Python requirements lightweight.

## Included Requirements

The file [`requirements.txt`](requirements.txt) covers the direct dependencies used by the
packaged study scripts:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `datamate`

## External Dependency: `flyvis`

The study code relies on the upstream `flyvis` package and its associated assets.
Those are not bundled here because they are maintained separately and can be large.

Provide one of the following before running the training scripts:

1. Install `flyvis` into the current environment, or
2. Export `FLYVIS_REPO_ROOT=/path/to/flyvis`

If you want to reproduce the original checkpoint-based comparison, also export:

```bash
export FLYVIS_BASELINE_CHECKPOINT=/path/to/chkpt_00000
```

The corrected random-initialization controls do not load the checkpoint, but the upstream
dataset and network definitions are still required.
