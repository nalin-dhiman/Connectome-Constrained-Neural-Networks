# Stage 3 Connectome vs Random Matched Steps 10

Scope:
- Matched-step extension.
- Same setup.
- Only iteration horizon increased.

Table A: loss

| Model | Iter 1 | Iter 3 | Iter 5 | Iter 10 |
|---|---:|---:|---:|---:|
| connectome | 0.8771 +/- 0.4965 | 0.5555 +/- 0.0739 | 0.5142 +/- 0.0147 | 0.4990 +/- 0.0087 |
| random | 2.6304 +/- 1.7846 | 0.6020 +/- 0.0765 | 0.6982 +/- 0.1524 | 0.5573 +/- 0.0549 |

Table B: activity

| Model | Iter 1 | Iter 3 | Iter 5 | Iter 10 |
|---|---:|---:|---:|---:|
| connectome | 0.6649 +/- 0.0000 | 0.6426 +/- 0.0478 | 0.6559 +/- 0.0542 | 0.7397 +/- 0.1044 |
| random | 2.7810 +/- 0.0000 | 2.1633 +/- 0.0785 | 1.8615 +/- 0.0095 | 1.3793 +/- 0.0679 |

Table C: time

| Model | Time at iter 10 |
|---|---:|
| connectome | 545.8036 +/- 23.5930 |
| random | 644.9179 +/- 18.3528 |

Delta summary at iter 10:

- delta_loss_iter10: 0.0583 +/- 0.0587
- delta_activity_iter10: 0.6397 +/- 0.0445
- delta_time_iter10: 99.1143 +/- 5.2490

Stability notes:
- all runs finite: True
- conclusion label: persists