# Physica A Revision Response Summary

This note integrates the already-completed corrected controls with the new targeted revision analyses requested for the Physica A submission.

## 1. Does the original strong topology claim survive?

No. The original checkpoint-based and naive-random comparison produced a strong apparent effect, but that result was confounded.

- Original step-5 mean delta_loss (control - connectome): 0.1841
- Original step-5 mean delta_activity (control - connectome): 1.2056
- Original step-5 mean delta_elapsed (control - connectome): 57.0037

## 2. What survives the corrected controls already in the paper?

- Phase 1 random-init step-5 mean delta_loss: -0.0020
- Phase 1 random-init step-5 mean delta_activity: 0.0323
- Phase 1 random-init step-5 mean delta_elapsed: 57.6023
- Phase 2 degree-preserving step-5 mean delta_loss: 0.0003
- Phase 2 degree-preserving step-5 mean delta_activity: -0.0106
- Phase 2 degree-preserving step-5 mean delta_elapsed: 8.3818

These corrected controls already show that the original loss advantage disappears under fair initialization and that the original activity advantage disappears under a stronger null.

## 3. Initial activity / dynamical-scale check

- Completed.
- Read: connectome and degree-preserving random start in broadly comparable dynamical regimes under the fair initialization route.
- Consequence: the optional calibration branch was not justified by the measured initial activity and gradient scales.

## 4. Degree-preserving ensemble robustness

- Completed.
- See `reports/revision_degpres_ensemble.md` for the full aggregated result.

## 5. Optional calibration

- Not run. The initial-activity check did not indicate a severe mismatch requiring calibration.

## Strongest safe final claim

Across the corrected controls, the strongest safe claim is that the original strong topology story does not survive, and the revised negative claim is strengthened further by the degree-preserving ensemble.

- Overall label: corrected claim strengthened