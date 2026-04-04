# Stage 4 Fragility Report

## Setup

- Models evaluated: canonical Stage 3 `baseline`, `weak`, and `moderate` checkpoints.
- Stimulus batch: the exact fixed MovingEdge batch saved by Stage 3.
- State metric: normalized divergence on the pooled central-cell representation used by the Stage 3 decoder.
- Task metric: masked task loss minus unmasked task loss on the same batch.
- Optional representation metric: cosine dissimilarity of final central states.
- Rankings compared:
  - `usage_proxy = |effective_weight| * mean(|source activity|)`
  - `weight_only = |effective_weight|`
- Important implementation detail:
  - masking is applied at the shared `syn_strength` parameter-group level, because that is the actual trainable unit exposed by flyvis in this setup.

## Main Result

The original targeted-fragility story does **not** survive this cleaned-up evaluation.

What is true:

- For all three models, the `usage_proxy` ranking produces a larger targeted-than-random normalized divergence at small mask sizes (`k = 10, 50`).

What is not true:

- The targeted advantage does not persist at `k = 100` and `k = 200`.
- Under `weight_only`, the targeted advantage is weaker and not robust.
- The regularized models do not show a clear targeted-fragility increase over the baseline model.

## Small-k vs Large-k Behavior

### Usage proxy

At small mask sizes, targeted masking is more damaging than random masking:

- baseline divergence gap:
  - `k=10`: `+0.3267`
  - `k=50`: `+0.3360`
- weak divergence gap:
  - `k=10`: `+0.3438`
  - `k=50`: `+0.3366`
- moderate divergence gap:
  - `k=10`: `+0.3627`
  - `k=50`: `+0.3745`

At larger mask sizes, the sign reverses:

- baseline divergence gap:
  - `k=100`: `-1.1407`
  - `k=200`: `-2.8576`
- weak divergence gap:
  - `k=100`: `-0.9597`
  - `k=200`: `-3.0981`
- moderate divergence gap:
  - `k=100`: `-0.3768`
  - `k=200`: `-2.5668`

This reversal is accompanied by very large random-ablation variance, especially for `k >= 100`.

### Weight-only ranking

The weight-only ranking is even less supportive of a targeted-fragility story:

- it shows only small positive gaps at `k = 10, 50`,
- and it also reverses by `k = 100, 200`.

That means the targeted-vs-random gap is not robust to even this simple ranking change.

## Task Degradation

Task degradation tells the same basic story.

- At small `k`, targeted masking is sometimes slightly worse than random under the usage proxy.
- At larger `k`, random masking often becomes more damaging on average because the random condition becomes highly variable.
- There is no stable pattern in which the regularized models are more task-fragile than baseline under targeted masking.

## Interpretation

The safest interpretation is:

- the cleaned-up pipeline can detect a small targeted effect under one ranking at small ablation sizes,
- but the effect is not stable enough across `k` or across ranking choice to support a strong fragility conclusion.

So the scientifically honest conclusion is negative:

- **Stage 4 does not currently provide robust evidence that activity regularization increases targeted fragility on this MovingEdge probe.**

## Outputs

- Raw table: `results/stage4/movingedge_fragility/fragility_metrics_raw.csv`
- Summary table: `results/stage4/movingedge_fragility/fragility_metrics_summary.csv`
- Plots:
  - `results/stage4/movingedge_fragility/plots/baseline_fragility.png`
  - `results/stage4/movingedge_fragility/plots/weak_fragility.png`
  - `results/stage4/movingedge_fragility/plots/moderate_fragility.png`
  - `results/stage4/movingedge_fragility/plots/combined_targeted_random_gap.png`
