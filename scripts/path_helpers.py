from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_flyvis_path(repo_root: Path) -> None:
    """Prefer an explicit flyvis checkout, otherwise fall back to installed flyvis."""
    candidates = []
    env_root = os.environ.get("FLYVIS_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(repo_root / "flyvis")
    candidates.append(repo_root / "src" / "external" / "flyvis")

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


def default_checkpoint_path() -> Path:
    return Path(
        os.environ.get(
            "FLYVIS_BASELINE_CHECKPOINT",
            "flyvis/data/results/flow/0000/000/chkpts/chkpt_00000",
        )
    )
