# stars2dm

Stochastic-interpolant prediction of dark matter density maps from stellar density maps.

Code and trained models for the paper. Contains the model, the two trained checkpoints used
in the paper, the analysis and figure code that produces all 43 figures, and the map
generation code for readers with access to the raw simulations.

> **Status: under construction.** Documentation (`docs/REPRODUCE.md`, `docs/DATA.md`,
> `docs/FIGURE_MAP.md`, `docs/CAVEATS.md`), the archive manifest and `scripts/fetch_data.py`
> are not written yet. See the layout below for what is already here.

## Layout

| path | what |
|---|---|
| `src/baryonbridge/` | the model: stochastic interpolant, conditional UNet, DREAMS/TNG50/FIRE data modules |
| `src/paperfigs/` | analysis and figure generation; run from inside this directory |
| `src/mapgen/` | density-map construction from raw snapshots |
| `src/stars2dm_paths.py` | every data root, resolved once; override with `STARS2DM_*` env vars |
| `configs/` | `v95_fiducial.yaml` (main text), `v97_no_theta.yaml` (Appendix D ablation) |
| `param_sweep_configs/` | the five 5-point simulation-parameter sweep grids |
| `reference_figures/` | the 43 figures as published, to diff regenerated output against |
| `tests/` | release integrity check; needs no data |

## The two models

| tag | conditioning | role |
|---|---|---|
| `v95` | stellar density map + 5 simulation parameters | fiducial; every main-text figure |
| `v97` | stellar density map only | Appendix D, no-parameter ablation |

Both: 128x128 maps, `rhostar -> rhoDM`, radially weighted loss `w(R) = 1/max(R, 2 kpc)`,
820/102/102 split, seed 42, evaluated with the stochastic Heun sampler at 100 steps.

## Install

```bash
conda env create -f environment.yml      # not written yet; see docs/gpu-torch-pip-freeze.txt
pip install -e .
python -m stars2dm_paths                 # check where the code expects data
python tests/test_release_integrity.py   # no data needed
```
