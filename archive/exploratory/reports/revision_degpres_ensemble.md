# Degree-Preserving Ensemble Revision

This report aggregates the 5-step degree-preserving ensemble over all sample-seed combinations. Connectome curves are reused from the completed fair-initialization Phase 2 runs because the connectome branch is invariant across degree-preserving samples.

- Degree-preserving samples: 5
- Seeds per sample: 3
- Total sample-seed combinations: 15
- All runs finite: True

| Metric @ iter 5 | Connectome | Degree-preserving |
|---|---:|---:|
| Loss | 0.5155 +/- 0.0067 | 0.5172 +/- 0.0061 |
| Activity | 0.5453 +/- 0.0185 | 0.5346 +/- 0.0147 |
| Elapsed (s) | 122.7463 +/- 2.1738 | 133.4701 +/- 7.0511 |

- delta_loss (degreepres - connectome): mean=0.0017, 95% CI=[0.0002, 0.0030]
- delta_activity (degreepres - connectome): mean=-0.0108, 95% CI=[-0.0146, -0.0067]
- delta_elapsed (degreepres - connectome): mean=10.7238, 95% CI=[6.1294, 14.8772]

## Interpretation
Across multiple degree-preserving rewired samples, the connectome does not recover a robust activity or loss advantage. This strengthens the corrected paper claim that the original effect was sensitive to null-model design.