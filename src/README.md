# Source Layout Notes

The runnable study code is kept under [`scripts/`](../scripts) to preserve the original research
workflow. The subdirectories here document the intended conceptual split:

- `core/`: canonical experiment logic
- `controls/`: null-model and revision-control logic
- `analysis/`: mechanism and summary analyses
- `plotting/`: figure-generation helpers
- `utils/`: shared utilities
