# Data Policy

This release includes only lightweight metadata and configuration snapshots.

## Included

- Canonical Stage 3 configuration snapshot
- Summary CSVs used in the paper tables
- Mask summary JSON files for the naive random and degree-preserving controls

## Excluded

- Raw `flyvis` datasets
- Large saved activity tensors
- Checkpoints
- Heavy rewired mask payloads (`.pt`)

## How to Regenerate

1. Install `flyvis` and make its data assets available.
2. Build lightweight masks with:
   - `python scripts/build_random_mask_selfloop_matched.py`
   - `python scripts/build_degree_preserving_random_mask.py`
3. Run the main or revision control wrappers:
   - `bash scripts/run_main_controls.sh`
   - `bash scripts/run_revision_controls.sh`

This repository is therefore inspection-ready and script-ready, but not fully self-contained with
respect to the upstream `flyvis` assets.
