# Phase 1 random-initialization control

This report summarizes the revision-control comparison only. Positive deltas are defined as `random - connectome`, so larger positive values mean the control is worse than the connectome.

## 5-step

| Metric | Connectome | Control |
|---|---:|---:|
| Loss @5 | 0.5155 +/- 0.0080 | 0.5136 +/- 0.0075 |
| Activity @5 | 0.5453 +/- 0.0219 | 0.5776 +/- 0.0144 |
| Elapsed @5 | 268.3767 +/- 25.8935 | 325.9790 +/- 12.9902 |

- delta_loss @5: mean=-0.0020, 95% CI=[-0.0022, -0.0015]
- delta_activity @5: mean=0.0323, 95% CI=[0.0233, 0.0402]
- delta_elapsed @5: mean=57.6023, 95% CI=[46.5345, 72.6953]
- all runs finite: True

## 10-step

| Metric | Connectome | Control |
|---|---:|---:|
| Loss @10 | 0.5045 +/- 0.0037 | 0.5024 +/- 0.0010 |
| Activity @10 | 0.5032 +/- 0.0025 | 0.5234 +/- 0.0023 |
| Elapsed @10 | 341.7291 +/- 140.2522 | 711.3146 +/- 688.9756 |

- delta_loss @10: mean=-0.0020, 95% CI=[-0.0059, 0.0001]
- delta_activity @10: mean=0.0202, 95% CI=[0.0170, 0.0224]
- delta_elapsed @10: mean=369.5855, 95% CI=[50.0687, 1003.4508]
- all runs finite: True
