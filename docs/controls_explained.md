# Controls Explained

The repository is organized around a control ladder.

## 1. Original Observation

The initial comparison used:

- checkpoint initialization
- a naive random graph matched only on global counts

This setup produced apparently strong connectome advantages in loss, activity, and runtime, but the
result is now treated as confounded.

## 2. Initialization Control

The random-initialization control removes the possibility that the random graph is simply disadvantaged
because the checkpoint is already adapted to connectome topology.

Observed effect:
- most of the loss advantage disappears

## 3. Degree-Preserving Null

The degree-preserving control keeps directed in-degree and out-degree sequences fixed while rewiring
the graph through edge swaps. This tests whether the earlier activity effect was just due to using a
weak null model.

Observed effect:
- the activity advantage does not robustly persist

## 4. Ensemble and Initial-Scale Diagnostics

The revision package also includes:

- a 5-sample degree-preserving ensemble
- an initial-activity / gradient-scale diagnostic

Together these strengthen the corrected conclusion that the original topology claim is not robust.
