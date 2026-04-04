# Revision Controls Summary

Question addressed:
1. How much of the original effect survives random initialization?
2. How much survives a degree-preserving null?
3. Was spectral / initial-activity calibration required?

Phase 3 status:
- Not run yet. It was deferred until after Phases 1 and 2 because the main reviewer confounds are initialization and degree sequence.

## Available comparisons

| Comparison | Step-5 delta loss | Step-5 delta activity | Step-5 delta elapsed |
|---|---:|---:|---:|
| Original checkpoint-based | 0.1841 | 1.2056 | 57.0037 |
| Phase 1 random-init | -0.0020 | 0.0323 | 57.6023 |
| Phase 2 degree-preserving | 0.0003 | -0.0106 | 8.3818 |

## Step-10 comparison

| Comparison | Delta loss | Delta activity | Delta elapsed |
|---|---:|---:|---:|
| Phase 1 random-init | -0.0020 | 0.0202 | 369.5855 |
| Phase 2 degree-preserving | -0.0018 | -0.0087 | 3.9814 |

## Safe interpretation

- Overall label: topology claim weakens materially
- If Phase 1 remains positive, the original result is not purely a checkpoint artifact.
- If Phase 2 remains positive, the effect is not explained solely by node-wise degree sequence.
- Phase 3 calibration should only be run if large iteration-1 mismatches remain after these two revisions.