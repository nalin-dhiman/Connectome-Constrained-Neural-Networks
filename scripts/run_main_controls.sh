#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${1:-results/generated/main_controls}"
DOC_ROOT="${2:-docs/generated}"
MASK_PATH="${OUT_ROOT}/random_mask_selfloop.pt"
STEP5_ROOT="${OUT_ROOT}/stage3_connectome_vs_random_matched_steps"
STEP10_ROOT="${OUT_ROOT}/stage3_connectome_vs_random_matched_steps_10"

mkdir -p "${OUT_ROOT}" "${DOC_ROOT}"

python scripts/build_random_mask_selfloop_matched.py \
  --output-path "${MASK_PATH}"

python scripts/stage3_train_connectome_vs_random_matched_steps.py \
  --random-mask-path "${MASK_PATH}" \
  --output-root "${STEP5_ROOT}" \
  --report-path "${DOC_ROOT}/stage3_connectome_vs_random_matched_steps.md"

python scripts/stage3_aggregate_matched_steps.py \
  --input-root "${STEP5_ROOT}" \
  --aggregated-csv "${STEP5_ROOT}/aggregated_metrics.csv" \
  --delta-csv "${STEP5_ROOT}/delta_metrics.csv" \
  --bootstrap-csv "${STEP5_ROOT}/bootstrap_iter5.csv" \
  --report-path "${DOC_ROOT}/stage3_connectome_vs_random_matched_steps.md"

python scripts/stage3_train_connectome_vs_random_matched_steps_10.py \
  --random-mask-path "${MASK_PATH}" \
  --output-root "${STEP10_ROOT}" \
  --report-path "${DOC_ROOT}/stage3_connectome_vs_random_matched_steps_10_seed_summary.md"

python scripts/stage3_aggregate_matched_steps_10.py \
  --input-root "${STEP10_ROOT}" \
  --aggregated-csv "${STEP10_ROOT}/aggregated_metrics.csv" \
  --delta-csv "${STEP10_ROOT}/delta_metrics.csv" \
  --report-path "${DOC_ROOT}/stage3_connectome_vs_random_matched_steps_10.md"

echo "Main controls finished."
