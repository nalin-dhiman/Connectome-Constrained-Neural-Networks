# Initial Activity / Dynamical-Scale Check

This report measures pre-training forward-pass activity under the same fair from-scratch initialization route used in the corrected experiments. The goal is to test whether the revised conclusion could still be explained by a trivial initial-scale mismatch.

- Seeds: [0, 1, 2]
- Canonical batch shape: (12, 269, 1, 721)
- Random mask path: results/random_mask_selfloop.pt
- Degree-preserving mask path: results/degree_preserving_random_mask.pt

| Model | Mean abs activity | Total abs activity | Node variance | Gradient norm |
|---|---:|---:|---:|---:|
| connectome | 0.568157 +/- 0.005894 | 83757482.67 +/- 868935.70 | 0.042987 +/- 0.002982 | 9.750689 +/- 4.205279 |
| degreepres | 0.577449 +/- 0.006228 | 85127285.33 +/- 918070.56 | 0.035447 +/- 0.002566 | 8.959382 +/- 3.962703 |
| random | 0.621578 +/- 0.007274 | 91632778.67 +/- 1072372.38 | 0.027629 +/- 0.001370 | 8.094053 +/- 2.951223 |

## Interpretation

- Relative mean-activity difference (connectome vs degree-preserving): 1.64%
- Gradient-norm ratio (larger/smaller, connectome vs degree-preserving): 1.088
- Qualitative severity: small
- Read: Connectome and degree-preserving random start in broadly comparable dynamical regimes under the fair initialization route. A calibration follow-up is not currently justified.