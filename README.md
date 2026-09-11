# stars2dm

Stochastic-interpolant prediction of dark matter density maps from stellar density maps.

Code and trained models for the paper. Contains the model, the two trained checkpoints used
in the paper, the analysis and figure code that produces all 43 figures, and the map
generation code for readers with access to the raw simulations.

> **Status: the archive is not published yet.** `data/MANIFEST.yaml` carries real checksums
> for the maps and checkpoints, but `archive_base_url` is still a placeholder, so
> `scripts/fetch_data.py` cannot download. Everything that runs from data you already have
> works; `--list` and `--verify` work regardless.

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
| `scripts/` | the job chain, `01_make_maps.sh` through `07_figures.sh` |
| `src/paperfigs/figure_cache.py` | reduced arrays that stand in for the raw prediction draws |
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

Map generation needs a second environment (`environment-mapgen.yml`) — it uses astropy and
agama, which the training environment does not carry, plus three packages that are not on
PyPI. Most readers will download the finished maps instead.

## Running it

`scripts/env.sh` is the only file you need to edit to run this somewhere else: it holds the
conda activation line and the data roots. Everything else derives from it.

```bash
bash scripts/submit_all.sh train      # 02: both models, 12 h each on 2x v100-16
bash scripts/submit_all.sh infer      # 03, 04: DREAMS in 9 shards, TNG50, FIRE m12
bash scripts/submit_all.sh sweeps     # 05: five parameter sweeps, ~16.5 h each
bash scripts/submit_all.sh aniso      # 06: anisotropy metric, ~1 h on CPU
bash scripts/submit_all.sh figures    # 07: five figure stages, in parallel
```

Each stage consumes the previous one's output, so let one finish before submitting the next.
Individual jobs take `--export=ALL,VAR=...`; the header of each script documents its
variables and says why the resource request is what it is.

## Documentation

| | |
|---|---|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | the five entry points, with exact commands |
| [docs/DATA.md](docs/DATA.md) | what the data is, how it is named, what it should hash to |
| [docs/FIGURE_MAP.md](docs/FIGURE_MAP.md) | which script makes which of the 43 figures |
| [docs/CAVEATS.md](docs/CAVEATS.md) | twelve ways to get a plausible wrong answer |

**Read `docs/CAVEATS.md` before trusting a number you derive from this code.** Every entry in
it was live at some point and produced a believable result rather than an error.

## Three ways in

You do not need everything to reproduce everything. Pick the entry point that matches how
much you want to re-derive:

| you want to | you need | size | what to run |
|---|---|---|---|
| restyle or re-read a figure | figure cache + maps | ~0.5 GB | `CACHE_MODE=read` figure stages |
| re-derive figures from predictions | prediction draws + maps | ~15 GB | figure stages |
| regenerate predictions | checkpoints + maps | ~1 GB | inference, then figure stages |
| regenerate everything | raw simulation snapshots | TB | the whole chain from `01` |

The figure cache holds what the panels are actually drawn from. A population figure puts
1024 galaxies on one axis from an array of shape `(n_galaxies, 32)` — one value per galaxy
per radial annulus. Producing that array costs 6.7 GB of reading; the array is under a
megabyte. So the array is what gets cached, and the figures redraw from it **pixel for
pixel identically** — verified panel by panel, not assumed.

Building the cache is a side effect of running the figure stages once with the draws
present, so it cannot drift from the figures: the same run produced both.

```bash
CACHE_MODE=write sbatch --export=ALL,STAGE=dreams scripts/07_figures.sh   # build it
python scripts/pack_figure_cache.py --list                               # inspect it
CACHE_MODE=read  sbatch --export=ALL,STAGE=dreams scripts/07_figures.sh   # draws not needed
```

What the cache cannot do: change the annulus geometry, the histogram binning, or the
draw combiner. Those choices are made upstream of what it stores, and `--cache-mode read`
will serve you the arrays built under the published settings regardless. Every one of those
settings is part of the cache key, so a mismatch produces a cache miss rather than a quietly
wrong number — but acting on a different choice means fetching the draws.
