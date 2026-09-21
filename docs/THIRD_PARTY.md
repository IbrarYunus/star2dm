# Third-party code and attribution

## BaryonBridge

`src/baryonbridge/` is a trimmed fork of [BaryonBridge](https://github.com/florpi/BaryonBridge)
by Carolina Cuesta-Lazaro and collaborators, included here with her agreement as a co-author
of the paper, and distributed under this repository's MIT licence.

What this repository changed, relative to upstream:

- a radially dependent weighting of the velocity-matching loss (`use_radial_loss_weight`,
  `radial_loss_weight_mode`, `radial_loss_power`, `radial_loss_r_min_kpc`) — the method the
  paper describes
- DREAMS, TNG50 and FIRE data modules (`data/dreams.py`, `data/tng.py`, `data/fire.py`)
- per-variant normalization and clipping profiles (`config.py`)
- the stochastic Heun sampler and per-draw output options used at evaluation

What was removed for this release, because the paper does not use it: the CAMELS and toy data
modules, the point-cloud transformer model, the gas notebooks, and 38 of 40 experiment configs.

The fork lives in one directory so it can be updated from upstream, or lifted out entirely,
without touching the analysis code.

## Constants formerly vendored from `utilities`

Earlier versions of this code carried `constant.py`, a 451-line module of physical constants
and unit conversions by **Andrew Wetzel** and **Andrew Emerick**, distributed with their
[`utilities`](https://bitbucket.org/awetzel/utilities) package (a dependency of
`gizmo_analysis`, which the FIRE map path already requires).

Exactly three values from it were ever used — `km_per_kpc`, `sec_per_Gyr` and
`radian_per_degree` — so rather than republish 451 lines of someone else's work, this
repository defines those three in `src/mapgen/units.py`, using the same expressions, with the
resulting floats pinned by the release test. They are bit-identical to the originals.

If you need more of that module, install it upstream rather than copying it back.

## Simulation suites

The maps are built from simulations this project did not run:

| suite | reference |
|---|---|
| DREAMS MW zooms | Rose et al., the DREAMS Project |
| TNG50 | Nelson et al., Pillepich et al., IllustrisTNG |
| FIRE m12 | the FIRE-2 collaboration |

The paper cites these properly; this file exists so the code's provenance is clear even to
someone who reads only the repository. See `CITATION.cff` for how to cite this work itself.
