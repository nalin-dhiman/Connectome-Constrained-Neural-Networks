# Stage 3 Connectome vs Random Matched Steps Saved

Scope:
- Identical 5-step matched-step experiment.
- No scientific changes to architecture, loss, dataset, optimizer, or seeds.
- Additional artifacts only: trained checkpoints and one post-training activity snapshot per run.

## Seed 0

- connectome: iterations=5/5, finite=True, activity_shape=(12, 269, 45669)
- connectome checkpoints: network=results/stage3_connectome_vs_random_matched_steps_saved/seed_0/connectome_network.pt, decoder=results/stage3_connectome_vs_random_matched_steps_saved/seed_0/connectome_decoder.pt
- connectome activity tensor: path=results/stage3_connectome_vs_random_matched_steps_saved/seed_0/connectome_activity.pt, size_mb=562.36
- random: iterations=5/5, finite=True, activity_shape=(12, 269, 45669)
- random checkpoints: network=results/stage3_connectome_vs_random_matched_steps_saved/seed_0/random_network.pt, decoder=results/stage3_connectome_vs_random_matched_steps_saved/seed_0/random_decoder.pt
- random activity tensor: path=results/stage3_connectome_vs_random_matched_steps_saved/seed_0/random_activity.pt, size_mb=562.36

## Seed 1

- connectome: iterations=5/5, finite=True, activity_shape=(12, 269, 45669)
- connectome checkpoints: network=results/stage3_connectome_vs_random_matched_steps_saved/seed_1/connectome_network.pt, decoder=results/stage3_connectome_vs_random_matched_steps_saved/seed_1/connectome_decoder.pt
- connectome activity tensor: path=results/stage3_connectome_vs_random_matched_steps_saved/seed_1/connectome_activity.pt, size_mb=562.36
- random: iterations=5/5, finite=True, activity_shape=(12, 269, 45669)
- random checkpoints: network=results/stage3_connectome_vs_random_matched_steps_saved/seed_1/random_network.pt, decoder=results/stage3_connectome_vs_random_matched_steps_saved/seed_1/random_decoder.pt
- random activity tensor: path=results/stage3_connectome_vs_random_matched_steps_saved/seed_1/random_activity.pt, size_mb=562.36

## Seed 2

- connectome: iterations=5/5, finite=True, activity_shape=(12, 269, 45669)
- connectome checkpoints: network=results/stage3_connectome_vs_random_matched_steps_saved/seed_2/connectome_network.pt, decoder=results/stage3_connectome_vs_random_matched_steps_saved/seed_2/connectome_decoder.pt
- connectome activity tensor: path=results/stage3_connectome_vs_random_matched_steps_saved/seed_2/connectome_activity.pt, size_mb=562.36
- random: iterations=5/5, finite=True, activity_shape=(12, 269, 45669)
- random checkpoints: network=results/stage3_connectome_vs_random_matched_steps_saved/seed_2/random_network.pt, decoder=results/stage3_connectome_vs_random_matched_steps_saved/seed_2/random_decoder.pt
- random activity tensor: path=results/stage3_connectome_vs_random_matched_steps_saved/seed_2/random_activity.pt, size_mb=562.36

Confirmation:
- This rerun preserves trained state and one post-training activity tensor per run, so it is suitable for later mechanism analysis.