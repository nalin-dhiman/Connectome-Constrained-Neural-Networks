# Connectome-Constrained Neural Networks:Control-Study Release
<img width="814" height="145" alt="image" src="https://github.com/user-attachments/assets/072d4db4-50fe-44c6-9f26-3b400f01955c" />

This repository packages the code, curated results, figures, and lightweight tables for a controlled
connectome-vs-random graph study. The central corrected conclusion is deliberately narrow:

> Apparent topology advantages in connectome-constrained neural networks are highly sensitive to
> initialization and null-model design, and do not robustly persist under degree-preserving controls.

The release is designed for inspection, lightweight reproduction, and reviewer-friendly auditing.

## Repository Contains

- Core matched-step comparison code for the original connectome-vs-random analysis
- Revision-control experiments that remove checkpoint-initialization confounds and strengthen the null model
- Mechanism-analysis scripts and lightweight summary tables
- Final figures  as `.png` and `.pdf`
- Clear documentation for the control ladder and the limits of the claim


## Key Result

The project initially appeared to show strong connectome advantages over a sparse random control in
loss, activity, and runtime. Those early results were confounded by checkpoint-based initialization
and a weak random null. After switching to fair random initialization and then replacing the naive
random graph with a degree-preserving rewired control, most of the apparent advantage disappeared:
the loss gap largely vanished under random initialization, and the activity gap did not persist
under the stronger degree-preserving null.

## Directory Structure

```text
github_release_repo/
├── environment/         # lightweight dependency list and setup notes
├── src/                 # placeholders for code organization notes
├── scripts/             # runnable study and plotting entrypoints
├── configs/             # packaged canonical config snapshots
├── data/                # metadata only; heavy raw assets are excluded
├── results/
│   ├── main_results/    # original main comparison results and mechanism summaries
│   ├── revision_results/# corrected control results
│   ├── tables/          # figure/paper CSV tables
│   └── generated/       # default target for fresh reruns
├── figures/
│   ├── main/            # paper-ready main figures
│   └── supplement/      # supplementary figures and diagnostics
├── docs/                # project overview, controls, reproducibility, FAQ, git push notes
└── archive/             # exploratory or negative branches kept for transparency
```

## Quick Start

1. Create an environment from [`environment/requirements.txt`](environment/requirements.txt).
2. Install `flyvis` and provide access to the required upstream assets.
3. Set the external paths if needed:

```bash
export FLYVIS_REPO_ROOT=/path/to/flyvis
export FLYVIS_BASELINE_CHECKPOINT=/path/to/chkpt_00000
```

4. Inspect packaged results and figures directly, or rerun a lightweight control path with the scripts below.

## Installation Notes

Study scripts rely on the upstream `flyvis` package and its associated data/checkpoint assets.
Those are not bundled here because they are large and maintained separately. See:

- [environment/README.md](environment/README.md)
- [data/README.md](data/README.md)
- [docs/reproducibility.md](docs/reproducibility.md)

## Reproducing Results

Main checkpoint-based comparison:

```bash
bash scripts/run_main_controls.sh
```

Corrected revision controls:

```bash
bash scripts/run_revision_controls.sh
```

Regenerating lightweight diagnostic figures:

```bash
bash scripts/make_figures.sh
```

Status summary for packaged revision outputs:

```bash
bash scripts/check_status.sh
```

## Control Ladder

The repository is organized around a control ladder rather than a positive narrative:

1. Original observation: connectome vs naive random, checkpoint initialized
2. Initialization control: connectome vs naive random, random initialization
3. Stronger null: connectome vs degree-preserving rewired control
4. Degree-preserving ensemble and initial-activity diagnostics

See [docs/controls_explained.md](docs/controls_explained.md).

## Limitations

- Single task family: MovingEdge direction decoding
- Short-horizon optimization only: 5 and 10 update steps
- Three training seeds in the main corrected comparisons
- Degree-preserving rewiring preserves degree sequence, not all higher-order graph statistics
- Spectral properties were not explicitly matched
- Some structured synthetic controls were unstable and are archived as negative or inconclusive branches

## Citation
If you use this work, please cite:

Dhiman, N. (2026). Topological Sensitivity in Connectome-Constrained Neural Networks.
arXiv:2604.04033.
https://doi.org/10.48550/arXiv.2604.04033

```bibtex

@misc{dhiman2026topologicalsensitivityconnectomeconstrainedneural,
  title={Topological Sensitivity in Connectome-Constrained Neural Networks},
  author={Dhiman, Nalin},
  year={2026},
  eprint={2604.04033},
  archivePrefix={arXiv},
  primaryClass={q-bio.NC},
  doi={10.48550/arXiv.2604.04033},
  url={https://arxiv.org/abs/2604.04033}
}
 ```

![DOI](https://img.shields.io/badge/DOI-10.48550%252FarXiv.2604.04033-blue)(https://doi.org/10.48550/arXiv.2604.04033)

