# Stage 3 Connectome vs Random Matched Steps

Scope:
- Matched-step comparison, not a benchmark.
- Same canonical Stage 3 setup.
- Same seeds [0, 1, 2].
- Fixed 5 iterations for both models.
- No wall-clock stopping.

Model table:

| Model | Nodes | Edges | Self-loops | Free params | Fixed params |
|---|---:|---:|---:|---:|---:|
| connectome | 45669 | 1513231 | 12380 | 734 | 2959 |
| random | 45669 | 1513231 | 12380 | 734 | 2959 |

Table A: loss

| Model | Iter 1 | Iter 3 | Iter 5 |
|---|---:|---:|---:|
| connectome | 0.8771 +/- 0.4965 | 0.5555 +/- 0.0739 | 0.5142 +/- 0.0147 |
| random | 2.6304 +/- 1.7846 | 0.6020 +/- 0.0765 | 0.6982 +/- 0.1524 |

Table B: activity

| Model | Iter 1 | Iter 3 | Iter 5 |
|---|---:|---:|---:|
| connectome | 0.6649 +/- 0.0000 | 0.6426 +/- 0.0478 | 0.6559 +/- 0.0542 |
| random | 2.7810 +/- 0.0000 | 2.1633 +/- 0.0785 | 1.8615 +/- 0.0095 |

Table C: elapsed time

| Model | Time at iter 5 |
|---|---:|
| connectome | 251.9255 +/- 43.6888 |
| random | 308.9292 +/- 10.7444 |

Delta summary (random - connectome at iter 5):

- loss: 0.1841 +/- 0.1571
- activity: 1.2056 +/- 0.0584
- elapsed time: 57.0037 +/- 37.3630

Bootstrap 95% CI for iter-5 deltas:

- loss: mean=0.1841, CI=[0.0324, 0.3460]
- activity: mean=1.2056, CI=[1.1469, 1.2636]
- elapsed time: mean=57.0037, CI=[27.2029, 98.9214]

Stability notes:
- all runs finite: True
- conclusion label: optimization advantage