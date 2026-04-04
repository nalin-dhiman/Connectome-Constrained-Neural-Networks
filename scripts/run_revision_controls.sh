#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT_ROOT="${1:-results/generated/revision_controls}"
DOC_ROOT="${2:-docs/generated}"
RANDOM_MASK_PATH="${OUT_ROOT}/random_mask_selfloop.pt"
DEG_MASK_PATH="${OUT_ROOT}/degree_preserving_random_mask.pt"
PHASE1_ROOT="${OUT_ROOT}/revision_phase1_random_init"
PHASE2_ROOT="${OUT_ROOT}/revision_phase2_degree_preserving"

mkdir -p "${OUT_ROOT}" "${DOC_ROOT}"

python scripts/build_random_mask_selfloop_matched.py \
  --output-path "${RANDOM_MASK_PATH}"

python scripts/stage3_train_connectome_vs_random_matched_steps_fromscratch.py \
  --random-mask-path "${RANDOM_MASK_PATH}" \
  --output-root "${PHASE1_ROOT}" \
  --max-iters 5

python scripts/stage3_train_connectome_vs_random_matched_steps_fromscratch.py \
  --random-mask-path "${RANDOM_MASK_PATH}" \
  --output-root "${PHASE1_ROOT}" \
  --max-iters 10

python scripts/build_degree_preserving_random_mask.py \
  --output-path "${DEG_MASK_PATH}"

python scripts/stage3_train_connectome_vs_degreepreserving_matched_steps.py \
  --degree-mask-path "${DEG_MASK_PATH}" \
  --output-root "${PHASE2_ROOT}" \
  --max-iters 5

python scripts/stage3_train_connectome_vs_degreepreserving_matched_steps.py \
  --degree-mask-path "${DEG_MASK_PATH}" \
  --output-root "${PHASE2_ROOT}" \
  --max-iters 10

python scripts/aggregate_revision_controls.py \
  --original-steps5-root "results/main_results/stage3_connectome_vs_random_matched_steps" \
  --original-steps10-root "results/main_results/stage3_connectome_vs_random_matched_steps_10" \
  --phase1-steps5-root "${PHASE1_ROOT}/steps_5" \
  --phase1-steps10-root "${PHASE1_ROOT}/steps_10" \
  --phase2-steps5-root "${PHASE2_ROOT}/steps_5" \
  --phase2-steps10-root "${PHASE2_ROOT}/steps_10" \
  --phase1-report-path "${DOC_ROOT}/revision_phase1_random_init.md" \
  --phase2-report-path "${DOC_ROOT}/revision_phase2_degree_preserving.md" \
  --report-path "${DOC_ROOT}/revision_controls_summary.md"

echo "Revision controls finished."
