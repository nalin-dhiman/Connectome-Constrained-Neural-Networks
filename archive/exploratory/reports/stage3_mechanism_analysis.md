# Stage 3 Mechanism Analysis

Scope:
- No retraining.
- Uses saved 5-step matched-step artifacts only.
- Seeds: 0, 1, 2.
- Connectome vs random only.

Node activity results:

| Model | Gini | Top 1% frac | Top 5% frac | Top 10% frac | Mean abs activity |
|---|---:|---:|---:|---:|---:|
| connectome | 0.4319 +/- 0.0372 | 0.0365 +/- 0.0066 | 0.1589 +/- 0.0214 | 0.2807 +/- 0.0308 | 0.6648 +/- 0.0481 |
| random | 0.4694 +/- 0.0033 | 0.0804 +/- 0.0013 | 0.2187 +/- 0.0023 | 0.3333 +/- 0.0027 | 1.7355 +/- 0.0150 |

Edge usage results:

| Model | Gini | Top 1% frac | Top 5% frac | Top 10% frac | Mean usage |
|---|---:|---:|---:|---:|---:|
| connectome | 0.7786 +/- 0.0080 | 0.1840 +/- 0.0043 | 0.4370 +/- 0.0118 | 0.6031 +/- 0.0159 | 0.0111 +/- 0.0008 |
| random | 0.8232 +/- 0.0025 | 0.2536 +/- 0.0025 | 0.5282 +/- 0.0033 | 0.6884 +/- 0.0034 | 0.0352 +/- 0.0002 |

Temporal stability results:

| Model | Mean total activity over time | Mean node variance over time | Mean temporal variation |
|---|---:|---:|---:|
| connectome | 30358.9054 +/- 2195.8333 | 0.6169 +/- 0.2162 | 0.0525 +/- 0.0884 |
| random | 79258.2581 +/- 686.6842 | 5.1969 +/- 0.1039 | 0.0021 +/- 0.0001 |

Candidate explanation:
- The connectome differs from the random graph in more than one way, but the metrics do not cleanly reduce to a single concentration or stability story.
- The safest reading is that the advantage is mechanistically mixed.

Final one-line mechanism label: mixed mechanism.