# Phase 2 degree-preserving control

This report summarizes the revision-control comparison only. Positive deltas are defined as `degree-preserving random - connectome`, so larger positive values mean the control is worse than the connectome.

## 5-step

| Metric | Connectome | Control |
|---|---:|---:|
| Loss @5 | 0.5155 +/- 0.0080 | 0.5159 +/- 0.0079 |
| Activity @5 | 0.5453 +/- 0.0219 | 0.5348 +/- 0.0157 |
| Elapsed @5 | 122.7463 +/- 2.5721 | 131.1281 +/- 6.3404 |

- delta_loss @5: mean=0.0003, 95% CI=[-0.0032, 0.0028]
- delta_activity @5: mean=-0.0106, 95% CI=[-0.0156, -0.0037]
- delta_elapsed @5: mean=8.3818, 95% CI=[5.8359, 13.2410]
- all runs finite: True

## 10-step

| Metric | Connectome | Control |
|---|---:|---:|
| Loss @10 | 0.5045 +/- 0.0037 | 0.5027 +/- 0.0019 |
| Activity @10 | 0.5032 +/- 0.0025 | 0.4945 +/- 0.0043 |
| Elapsed @10 | 268.2284 +/- 20.6850 | 272.2098 +/- 20.5507 |

- delta_loss @10: mean=-0.0018, 95% CI=[-0.0058, 0.0014]
- delta_activity @10: mean=-0.0087, 95% CI=[-0.0151, -0.0042]
- delta_elapsed @10: mean=3.9814, 95% CI=[2.9416, 5.0700]
- all runs finite: True
