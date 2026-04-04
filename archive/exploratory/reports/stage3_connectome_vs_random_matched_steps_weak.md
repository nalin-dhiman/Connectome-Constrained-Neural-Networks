# Stage 3 Connectome vs Random Matched Steps Weak

Scope:
- Matched-step comparison.
- Same canonical Stage 3 setup.
- Seeds [0, 1, 2].
- Fixed 5 iterations.
- Four conditions only.
- Not a benchmark.

Model/control table:

| Graph | Nodes | Edges | Self-loops | Free params | Fixed params |
|---|---:|---:|---:|---:|---:|
| connectome | 45669 | 1513231 | 12380 | 734 | 2959 |
| random | 45669 | 1513231 | 12380 | 734 | 2959 |

Table A: loss

| Condition | Iter 1 | Iter 3 | Iter 5 |
|---|---:|---:|---:|
| connectome_baseline | 0.8771 +/- 0.4965 | 0.5555 +/- 0.0739 | 0.5142 +/- 0.0147 |
| connectome_weak | 0.8771 +/- 0.4965 | 0.5555 +/- 0.0738 | 0.5141 +/- 0.0144 |
| random_baseline | 2.6304 +/- 1.7846 | 0.6020 +/- 0.0765 | 0.6982 +/- 0.1524 |
| random_weak | 2.6304 +/- 1.7846 | 0.5986 +/- 0.0745 | 0.6948 +/- 0.1490 |

Table B: activity

| Condition | Iter 1 | Iter 3 | Iter 5 |
|---|---:|---:|---:|
| connectome_baseline | 0.6649 +/- 0.0000 | 0.6426 +/- 0.0478 | 0.6559 +/- 0.0542 |
| connectome_weak | 0.6649 +/- 0.0000 | 0.6419 +/- 0.0473 | 0.6527 +/- 0.0525 |
| random_baseline | 2.7810 +/- 0.0000 | 2.1633 +/- 0.0785 | 1.8615 +/- 0.0095 |
| random_weak | 2.7810 +/- 0.0000 | 2.1467 +/- 0.0653 | 1.8398 +/- 0.0196 |

Table C: elapsed time

| Condition | Time at iter 5 |
|---|---:|
| connectome_baseline | 265.0501 +/- 20.0902 |
| connectome_weak | 274.5530 +/- 12.8024 |
| random_baseline | 316.7945 +/- 7.7329 |
| random_weak | 330.4362 +/- 7.2519 |

Weak-effect comparison at iter 5:

- connectome baseline loss/activity: 0.5142 / 0.6559
- connectome weak loss/activity: 0.5141 / 0.6527
- delta_activity_connectome: 0.0032 +/- 0.0020
- delta_loss_connectome: -0.0000 +/- 0.0003

- random baseline loss/activity: 0.6982 / 1.8615
- random weak loss/activity: 0.6948 / 1.8398
- delta_activity_random: 0.0217 +/- 0.0191
- delta_loss_random: -0.0034 +/- 0.0035

Comparative quantities:

- delta_activity_advantage: -0.0185 +/- 0.0210
- delta_loss_advantage: 0.0034 +/- 0.0036

Bootstrap 95% CI for iter-5 deltas:

- delta_activity_connectome: mean=0.0032, CI=[0.0009, 0.0047]
- delta_loss_connectome: mean=-0.0000, CI=[-0.0003, 0.0002]
- delta_activity_random: mean=0.0217, CI=[0.0071, 0.0434]
- delta_loss_random: mean=-0.0034, CI=[-0.0063, 0.0005]
- delta_activity_advantage: mean=-0.0185, CI=[-0.0425, -0.0032]
- delta_loss_advantage: mean=0.0034, CI=[-0.0005, 0.0065]

Stability notes:
- all runs finite: True
- conclusion label: random benefits more