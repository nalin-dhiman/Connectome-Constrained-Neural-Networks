#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python scripts/revision_make_additional_plots.py

echo "Updated lightweight revision figures under figures/main and figures/supplement."
echo "Bundled mechanism-heavy figures remain included as final artifacts and require excluded saved-state tensors to regenerate fully."
