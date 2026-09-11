"""Filesystem roots for stars2dm. Every data path the code reads resolves through here.

Defaults put everything under ``<repo>/data``, which is what ``scripts/fetch_data.py``
populates. Point any root somewhere else with an environment variable -- on a cluster the
data usually lives on a scratch filesystem, not next to the source:

    STARS2DM_DATA_ROOT     parent of everything below      (default: <repo>/data)
    STARS2DM_DREAMS_ROOT   DREAMS maps, parameters, preds  (default: $DATA_ROOT/DREAMS_data)
    STARS2DM_TNG_ROOT      TNG50 maps and parameters       (default: $DATA_ROOT/TNG_data)
    STARS2DM_FIRE_ROOT     FIRE m12 maps and parameters    (default: $DATA_ROOT/FIRE_data)
    STARS2DM_PRED_ROOT     per-galaxy prediction npz       (default: $DREAMS_ROOT/Pred/CDM/MW_zooms/SB5)
    STARS2DM_CKPT_ROOT     model checkpoints               (default: $DATA_ROOT/checkpoints)
    STARS2DM_CACHE_ROOT    figure-data cache               (default: $DATA_ROOT/figure_cache)

``describe()`` prints the resolved roots and whether each exists; run
``python -m stars2dm_paths`` to check a setup before launching a long job.
"""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _root(env_var, default):
    value = os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else Path(default)


DATA_ROOT = _root("STARS2DM_DATA_ROOT", _REPO_ROOT / "data")
DREAMS_ROOT = _root("STARS2DM_DREAMS_ROOT", DATA_ROOT / "DREAMS_data")
TNG_ROOT = _root("STARS2DM_TNG_ROOT", DATA_ROOT / "TNG_data")
FIRE_ROOT = _root("STARS2DM_FIRE_ROOT", DATA_ROOT / "FIRE_data")
PRED_ROOT = _root("STARS2DM_PRED_ROOT", DREAMS_ROOT / "Pred/CDM/MW_zooms/SB5")
CKPT_ROOT = _root("STARS2DM_CKPT_ROOT", DATA_ROOT / "checkpoints")
CACHE_ROOT = _root("STARS2DM_CACHE_ROOT", DATA_ROOT / "figure_cache")

# ---------------------------------------------------------------------------
# The paired maps the paper's models were trained and evaluated on.
#
# The 'csv_fov_z0p350_0p350' suffix is not decoration: it records that the grid extent came
# from src/mapgen/redshift_sample_0p35.csv, one row per galaxy, all at z=0.35. Maps built
# with a different suffix are a different dataset and the normalization constants in
# baryonbridge/config.py do not apply to them.
# ---------------------------------------------------------------------------
MAP_SUFFIX = "csv_fov_z0p350_0p350"

DREAMS_MAPS_DIR = DREAMS_ROOT / "2D_maps/CDM/MW_zooms/SB5"
TNG_MAPS_DIR = TNG_ROOT / "2D_maps/TNG50"
FIRE_MAPS_DIR = FIRE_ROOT / "2D_maps/FIREm12"

DREAMS_RHODM_MAP = DREAMS_MAPS_DIR / f"Maps_rhoDM_TNG_SB5_128_z=0.00_{MAP_SUFFIX}.npy"
DREAMS_RHOSTAR_MAP = DREAMS_MAPS_DIR / f"Maps_rhostar_TNG_SB5_128_z=0.00_{MAP_SUFFIX}.npy"
TNG_RHODM_MAP = TNG_MAPS_DIR / f"Maps_rhoDM_TNG50_128_z=0.00_{MAP_SUFFIX}.npy"
TNG_RHOSTAR_MAP = TNG_MAPS_DIR / f"Maps_rhostar_TNG50_128_z=0.00_{MAP_SUFFIX}.npy"
FIRE_RHODM_MAP = FIRE_MAPS_DIR / f"Maps_rhoDM_FIREm12_128_z=0.00_{MAP_SUFFIX}.npy"
FIRE_RHOSTAR_MAP = FIRE_MAPS_DIR / f"Maps_rhostar_FIREm12_128_z=0.00_{MAP_SUFFIX}.npy"

DREAMS_THETA_PATH = DREAMS_ROOT / "Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt"
TNG_THETA_PATH = TNG_ROOT / "Parameters/TNG50/CDM_TNG_MW_SB5.txt"
FIRE_THETA_PATH = FIRE_ROOT / "Parameters/FIREm12/CDM_TNG_MW_SB5.txt"

CHECKPOINTS = {
    "v95": CKPT_ROOT / "last-v95.ckpt",   # fiducial, theta-conditioned
    "v97": CKPT_ROOT / "last-v97.ckpt",   # Appendix D, no-theta ablation
}

SWEEP_CONFIG_DIR = _REPO_ROOT / "param_sweep_configs"

# Particle-based bar strength for the same 1024 galaxies, measured in 3D from the snapshots.
# Independent of the maps, so it is the right thing to validate the map-based A_2 operator
# against (analyze_anisotropy.py crosscheck).
BAR_STRENGTH_CSV = _REPO_ROOT / "data/tables/bar_characterization_v21.csv"
REFERENCE_FIGURES = _REPO_ROOT / "reference_figures"

# ---------------------------------------------------------------------------
# Field of view: two numbers, and they deliberately disagree.
#
# The maps span +/-32.596465 kpc -- redshift_sample_0p35.csv carries
# physical_fov_kpc = 65.192930 for all 1024 rows, and get_2D_maps.py's --redshift-csv
# branch sets xmin/xmax = -/+fov/2 with no override.
#
# Every published radius is on a +/-32 kpc axis instead: build_radial_annulus_geometry
# defaults to +/-32 and every figure in the paper was produced that way. The two differ by
# 1.86%, so a radius quoted by this code is 1.86% low in absolute kpc (multiply by 1.0186),
# and the outermost annulus centre sits at 45.25 kpc where the true map corner is 46.10.
#
# This is PRESERVED, not fixed: changing it would move every number in the paper. The
# assertion below exists so the disagreement stays visible -- if you rebuild maps at a
# different FOV, it fires instead of silently rescaling every result.
# ---------------------------------------------------------------------------
MAP_FOV_HALF_KPC = 32.596465
ANALYSIS_FOV_HALF_KPC = 32.0
FOV_RATIO = MAP_FOV_HALF_KPC / ANALYSIS_FOV_HALF_KPC

assert abs(FOV_RATIO - 1.018639) < 1e-5, (
    f"Map FOV ({MAP_FOV_HALF_KPC} kpc) and analysis FOV ({ANALYSIS_FOV_HALF_KPC} kpc) no "
    f"longer stand in the published 1.0186 ratio. The paper's radii are on the "
    f"{ANALYSIS_FOV_HALF_KPC} kpc axis; see docs/CAVEATS.md before changing either.")


def describe():
    """Print every resolved root and whether it exists."""
    rows = [
        ("DATA_ROOT", DATA_ROOT), ("DREAMS_ROOT", DREAMS_ROOT), ("TNG_ROOT", TNG_ROOT),
        ("FIRE_ROOT", FIRE_ROOT), ("PRED_ROOT", PRED_ROOT), ("CKPT_ROOT", CKPT_ROOT),
        ("CACHE_ROOT", CACHE_ROOT),
    ]
    width = max(len(name) for name, _ in rows)
    for name, path in rows:
        print(f"{name:<{width}}  {'ok     ' if path.exists() else 'MISSING'}  {path}")


if __name__ == "__main__":
    describe()
