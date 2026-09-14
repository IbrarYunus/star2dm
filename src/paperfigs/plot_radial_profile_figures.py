#!/usr/bin/env python3
"""
Generate radial-profile comparison figures from saved npz radial profiles and pred draws.

Extracted from radial_profile_check_da_check.ipynb.
"""

import argparse
import fnmatch
import os
import time
from pathlib import Path

import figure_cache
import pred_data_loaders as _pdl
import stars2dm_paths
from pred_data_loaders import (
    DEFAULT_PRED_ROOT,
    aggregate_pixels,
    combine_draws,
    combine_label,
    build_structured_from_sim_maps,
    check_ood_path_examples,
    default_draws_file_pattern,
    effective_pred_root,
    load_all_data_structured as load_all_per_galaxy,
    load_pred_data_structured as load_pred_per_galaxy,
    parse_draws_slice as parse_draws_slice_shared,
    parse_version_tag,
    radial_profiles_exist,
    set_combiners,
)

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.stats as stats

from paper_style import (
    UNIT_NORMAL_LABEL,
    apply_paper_style,
    mark_fov_limit,
    no_grid,
    paper_grid,
    save_figure as save_paper_figure,
    style_azimuthal_axis,
    symmetric_ylim,
    to_degrees,
)

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

VERSION_TAG = "v56" # TNG50_v52 or FIREm12_v52 or v52
# Optional second version overlaid on scatter / deviation plots (None = single version only)
VERSION_TAG_2 = None
# Optional: galaxy IDs to highlight (e.g. the held-out test split) on single-version
# predict-vs-true panels, with dual R^2 (highlighted-only + all). Set via main()/CLI;
# mutually exclusive with VERSION_TAG_2.
HIGHLIGHT_GALAXY_IDS = None

PRED_ROOT = DEFAULT_PRED_ROOT
DATA_LAYOUT = "per_galaxy"
NUM_DRAWS = 10
DATA_DIR = stars2dm_paths.PRED_ROOT
# Radial-profile npz files (x1, x_pred profiles, radial_centers). Often the base version
# directory even when draws live in a suffixed folder (e.g. v47 vs v47_flat_prior_16).
RADIAL_PROFILE_DATA_DIR = None
# Pred draws npz files; None uses DATA_DIR.
# DRAWS_DATA_DIR = stars2dm_paths.PRED_ROOT / "v47_sn1_sweep_5"
# DRAWS_FILE_PATTERN = '*pred_draws_50_sn1_sweep_5.npz'
DRAWS_DATA_DIR = None
DRAWS_FILE_PATTERN = '*pred_draws_10.npz'
# Which draws to keep per galaxy file: Python slice [start:end] on draw axis before channel.
# Example: (1, 17) skips draw 0 and uses draws 1..16 from a 17-draw file.
# DRAWS_SLICE = (40, 50)
DRAWS_SLICE = None
DRAWS_CHANNEL = 0
# Optional tag appended to output PNG names only (not used for file discovery).
# DRAWS_OUTPUT_LABEL = "sn1_sweep_5_4"
DRAWS_OUTPUT_LABEL = None

# Independent draws controls for VERSION_TAG_2 overlay
DRAWS_DATA_DIR_2 = None
# e.g. '*pred_draws_640_FIREm12_flat_prior_baryon_64.npz' for FIREm12_v55 sweep
DRAWS_FILE_PATTERN_2 = '*pred_draws_640_FIREm12_flat_prior_baryon_64.npz'
DRAWS_SLICE_2 = None
# PNG name tag for v2 overlay, e.g. "flat_prior_baryon_64"
DRAWS_OUTPUT_LABEL_2 = "flat_prior_baryon_64"
NUM_DRAWS_2 = 640
OUTPUT_DIR = Path(__file__).resolve().parent
SMOKE_TEST_OUTPUT_DIR = Path(__file__).resolve().parent / "temp" / "radial_profile_figures"
SAVE_DPI = 300
# The ensemble/annulus combination convention itself lives in pred_data_loaders, which
# every figure script imports; set it with set_combiners() or the --ensemble-combine /
# --profile-agg flags.
# How the 922 train/val DREAMS galaxies are drawn behind the 102 held-out ones in the
# predicted-vs-true figure: as individual points, as density contours, or not at all.
# 'hide' also drops the DREAMS,all entry from the per-panel R^2 box.
#
# DREAMS_ALL_STYLES may list several; each is emitted as its own file from a single data
# load, since re-running the whole script per variant would re-read 1024 x 100 draws.
DREAMS_ALL_STYLE = 'points'
DREAMS_ALL_STYLES = ['points']

# None = all galaxies; or e.g. np.arange(0, 2) for a quick test
GALAXY_IDX_LIST = None

RADIAL_INDICES_MIXED = [2, 4, 6, 14, 17, 28]
RADIAL_INDICES_CENTER = [0, 1, 2, 3, 4, 5]
AXIS_LLIM = [6.9, 6.4, 6.1, 5.3, 5.1, 4.7]
AXIS_ULIM = [8.1, 7.6, 7.4, 6.7, 6.6, 6.1]

COMPUTE_DRAWS_PROFILES = True

PLOT_PRED_VS_TRUE = False
PLOT_PRED_VS_TRUE_MEDIAN_CC = True
PLOT_PRED_VS_TRUE_MEDIAN_CC_CENTER = True
PLOT_PRED_VS_TRUE_MEDIAN_CC_ERRORBARS = False
PLOT_STD_PRED_VS_TRUE = True
PLOT_STD_PRED_VS_TRUE_MEDIAN_CC = True
PLOT_DEVIATION_DISTRIBUTION = True

PLOT_PER_GALAXY_DEVIATION = True
PLOT_PER_GALAXY_DRAW_PANELS = True
PLOT_PER_GALAXY_PRIOR_POSTERIOR = True
# Galaxy IDs (not rows into the loaded draws) shown as the worked example in
# Figures 1-3. Override with --per-galaxy-indices.
PER_GALAXY_INDICES = [19]
PER_GALAXY_PRIOR_POSTERIOR_DPI = 200
PRIOR_POSTERIOR_MAX_BACKGROUND_LINES = 256
PRIOR_POSTERIOR_SHOW_BACKGROUND_LINES = True
# Cap on how many posterior draws are DRAWN. Ten fat semi-transparent curves read as an
# ensemble; a hundred composite to effective opacity 1 and render as a solid slab with the
# median invisible inside it. This caps the rendering only -- the median and any percentile
# band are still computed from every draw.
PRIOR_POSTERIOR_MAX_POSTERIOR_LINES = 20
PRIOR_POSTERIOR_DRAW_LINEWIDTH = 1.0
PRIOR_POSTERIOR_DRAW_ALPHA = 0.35
# Artists below this zorder are rasterized when saving to a vector format; the median, the
# truth curve and all text sit above it and stay vector. Only has an effect for PDF/SVG output.
PRIOR_POSTERIOR_RASTER_ZORDER = 4.5
SIM_FOR_MAPS = "DREAMS" # FIREm12 or TNG50 or DREAMS
PER_GALAXY_SAVE_SUMMARY = True
PER_GALAXY_SUMMARY_N_RINGS = 16
PER_GALAXY_SUMMARY_N_AZIMUTHAL = 8
PER_GALAXY_SAVE_DPI = 200
PER_GALAXY_DRAW_PANELS_DPI = 200
PER_GALAXY_DRAW_PANELS_N_SHOW = 4
# Text sizes below are set against the size the figure PRINTS at, not the size it is
# drawn at. Figure 1 is ~19 in wide and lands in a figure* at 0.95\linewidth = 488 pt,
# a scale of about 0.35, so every number here is divided by three on the page. The whole
# family was scaled by 1.6 together, which leaves the proportions between title, flow
# label and colorbar untouched and puts the tick labels near 8 pt printed.
DRAW_PANEL_TITLE_FONTSIZE = 22
DRAW_PANEL_FIG_TITLE_FONTSIZE = 29
# Capped by the width of 'stochastic' in the 81 px flow gutter; above 18 it slides
# under the first draw panel.
DRAW_PANEL_FLOW_FONTSIZE = 18
DRAW_PANEL_CBAR_LABEL_FONTSIZE = 29
DRAW_PANEL_CBAR_TICK_FONTSIZE = 23
# The four narrow draw columns are half the width of the flanking panels, and their
# titles sit inside them, so an oversized title runs out of frame and over its
# neighbour instead of pushing anything aside. 'draw 100 - true' is the longest and
# stops fitting above 16 pt, so this is a geometric ceiling, not a taste call.
DRAW_PANEL_DRAW_TITLE_FONTSIZE = 15
# Physical extent of one map panel, annotated on the conditioning panel of Figure 1 so
# the reader has a scale without having to find it in the caption.
DRAW_PANEL_SCALE_NOTE = '$64\\times64$ kpc'
DRAW_PANEL_SCALE_NOTE_FONTSIZE = 21
# The scatter and deviation-distribution figures print at a scale near 0.5, so these two
# land at roughly 10 pt and 8.5 pt on the page. The axis label has to stay above the tick
# label; 16/17 as they stood had it the wrong way round.
AXIS_LABEL_FONTSIZE = 20
AXIS_TICK_LABEL_FONTSIZE = 17
IN_PLOT_TEXTBOX_FONTSIZE = 14
RHO_MAP_VMIN = 5.5
RHO_MAP_VMAX = 9.5
COND_MAP_VMIN = 3.0
COND_MAP_VMAX = 11.0

# Show R^2 in scatter-panel text boxes; misleading when a v2 overlay also has
# many galaxies (R^2 reported is v1-only), so allow hiding via --hide-r2.
SHOW_R2 = True

# Colors for combined per-galaxy v1+v2 figures (residual hist, prior/posterior).
# v1 keeps C1 to match the single-version prior/posterior style.
PER_GALAXY_V1_COLOR = 'C1'
PER_GALAXY_V2_COLOR = 'C0'

SECOND_VERSION_MAX_SAMPLES_WARN = 10
V2_MARKER = 'D'
V2_MARKER_SIZE = 48
V2_EDGE_COLOR = 'darkblue'
V2_EDGE_WIDTH = 2.0
# The held-out DREAMS galaxies use the same overlay helper as the FIRE series, but must
# look different from it: plain filled circles. The heavy diamond outline obscured both
# the points underneath it and its own sigma fill colour, and the diamond shape is
# needed by the FIRE overlay in the very same figure to separate FIRE from TNG50.
HIGHLIGHT_MARKER = 'o'
HIGHLIGHT_MARKER_SIZE = 26
HIGHLIGHT_EDGE_COLOR = 'none'
HIGHLIGHT_EDGE_WIDTH = 0.0
# Enclosed-probability levels for the DREAMS-all contours (1, 1.5 and 2 sigma for a
# 2-D Gaussian), measured from the density ranking rather than assumed.
DREAMS_ALL_CONTOUR_LEVELS = (0.39, 0.68, 0.95)
DREAMS_ALL_CONTOUR_COLOR = '0.35'
V2_HIST_COLOR = 'darkblue'
# Deviation-histogram-only styling: v1 (background) is steelblue, v2/highlight overlay
# is a lighter blue fill (edge stays V2_HIST_EDGE_COLOR for a crisp, still-legible
# outline) so the darker overlay bars don't fully obscure the background histogram.
V1_HIST_COLOR = 'steelblue'
V2_HIST_FILL_COLOR = 'cornflowerblue'
V2_HIST_EDGE_COLOR = V2_EDGE_COLOR
V2_HIST_ALPHA = 0.45
DEVIATION_HIST_TEXTBOX_FONTSIZE = 14
DEVIATION_HIST_BINS_V1 = 24
# Plotted range of the Delta histograms. The quoted mean must lie inside a range that
# actually shows the distribution it summarizes: the parameter-sweep runs push Delta
# far off zero with a long one-sided tail, so those figures need a wider window than
# the near-symmetric fiducial/OOD ones. Override with --deviation-hist-xlim.
DEVIATION_HIST_XLIM = (-5, 5)

# --- The three stacked panels of the per-galaxy residual figure (paper Figure 2) ------
# They are placed one above another in LaTeX at a common \linewidth, so a difference in
# y-label width would push their axes to different x offsets. They therefore share a
# figure size AND an explicit margin box, and are saved with tight=False: bbox_inches
# ='tight' crops each file to its own content, which is exactly what breaks the stacking.
# These print at a scale near 0.32 -- a 10 in figure squeezed into one 242 pt column --
# so their text needs to be about three times the nominal size. The margin box grew with
# it: with tight=False nothing reflows, so a taller x label simply runs off the canvas.
RESIDUAL_PANEL_FIGSIZE = (10, 5)
RESIDUAL_PANEL_MARGINS = dict(left=0.160, right=0.975, bottom=0.225, top=0.965)
RESIDUAL_PANEL_LABEL_FONTSIZE = 30
RESIDUAL_PANEL_TICK_FONTSIZE = 26
RESIDUAL_PANEL_LEGEND_FONTSIZE = 22
RESIDUAL_PANEL_TEXTBOX_FONTSIZE = 24
PER_GALAXY_HIST_XLIM = (-5, 5)

# Paper Figure 3 (prior vs posterior) is a single wide panel reproduced at column width,
# so its text has to survive a large downscale.
PRIOR_POSTERIOR_LABEL_FONTSIZE = 28
PRIOR_POSTERIOR_TICK_FONTSIZE = 24
PRIOR_POSTERIOR_LEGEND_FONTSIZE = 22

# Normalization dictionary copied from dreams dataloader; updated 11/26/2025
NORM_DICT = {
    '3D': {
        'T': {
            256: {'mean': 3.84, 'std': 1.01, 'min': 0.},
            128: {'mean': 3.85, 'std': 1.01, 'min': 205.},
        },
        'Mcdm': {
            256: {'mean': 10.014, 'std': 0.55, 'min': 0.},
            128: {'mean': 10.04, 'std': 0.55, 'min': 5855244.}
        },
        'Mgas': {
            128: {'mean': 9.5, 'std': 0.56, 'min': 1218244.4}
        },
        'Vgas': {
            128: {'mean': 139.9, 'std': 78.8, 'min': 1.45}
        },
        'Vcdm': {
            128: {'mean': 145.7, 'std': 88.45, 'min': 1.68}
        },
    },
    '2D': {
        'MDM': {128: {'mean': 2.00, 'std': 2.96, 'min': 0.}},
        'Mstar': {128: {'mean': 3.39, 'std': 2.86, 'min': 0.}},
        'rhoDM': {
            128: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122},
            32: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122},
        },
        'rhostar': {
            128: {'mean': 4.5496, 'std': 1.1889, 'min': 0.7828},
            32: {'mean': 4.5496, 'std': 1.1889, 'min': 0.7828},
        },
        'EuVISnoise': {128: {'mean': 1.05e25, 'std': 1.24e26, 'min': -1.50e25}},
        'EuVISnoiselog10': {128: {'mean': 26.225, 'std': 0.0667, 'min': 24.97}},
        'EuVISfluxnoiselog10': {128: {'mean': 2.1312, 'std': 0.1084, 'min': -3.8622}},
        'Uavg': {128: {'mean': 0.23, 'std': 109.78, 'min': -537.0}},
        'Vavg': {128: {'mean': 0.99, 'std': 106.03, 'min': -516.0}},
        'Wavg': {128: {'mean': -0.15, 'std': 22.20, 'min': -341.0}},
        'vRavg': {128: {'mean': 0.61, 'std': 26.61, 'min': -351.51}},
        'vTavg': {128: {'mean': 125.79, 'std': 82.45, 'min': -310.0}},
    }
}

# Truth and conditioning map stacks, one pair per suite. These used to be written relative
# to the source tree, which worked only because the data happened to sit beside the checkout;
# they resolve through stars2dm_paths now so STARS2DM_*_ROOT redirects them.
MAP_PATHS = {
    "FIREm12": {
        "rhoDM": stars2dm_paths.FIRE_RHODM_MAP,
        "rhostar": stars2dm_paths.FIRE_RHOSTAR_MAP,
    },
    "TNG50": {
        "rhoDM": stars2dm_paths.TNG_RHODM_MAP,
        "rhostar": stars2dm_paths.TNG_RHOSTAR_MAP,
    },
    "DREAMS": {
        "rhoDM": stars2dm_paths.DREAMS_RHODM_MAP,
        "rhostar": stars2dm_paths.DREAMS_RHOSTAR_MAP,
    },
}


def version_label(tag, label):
    return f"{tag}_{label}" if label else tag


def short_version_label(tag, label):
    """Compact label for legends/text in combined v1+v2 figures."""
    return label if label else tag


def output_path(stem):
    prefix = version_label(VERSION_TAG, DRAWS_OUTPUT_LABEL)
    if VERSION_TAG_2:
        tag2 = version_label(VERSION_TAG_2, DRAWS_OUTPUT_LABEL_2)
        return OUTPUT_DIR / f"{prefix}_with_{tag2}_{stem}.png"
    if HIGHLIGHT_GALAXY_IDS is not None:
        return OUTPUT_DIR / f"{prefix}_test_highlight_{stem}.png"
    return OUTPUT_DIR / f"{prefix}_{stem}.png"


def output_path_single(stem, which='v1'):
    """Version-specific PNG path (no _with_), for per-version map-like figures."""
    if which == 'v2':
        prefix = version_label(VERSION_TAG_2, DRAWS_OUTPUT_LABEL_2)
    else:
        prefix = version_label(VERSION_TAG, DRAWS_OUTPUT_LABEL)
    return OUTPUT_DIR / f"{prefix}_{stem}.png"


def parse_draws_slice(spec):
    if spec is None:
        return slice(None)
    if isinstance(spec, slice):
        return spec
    if isinstance(spec, tuple):
        if len(spec) == 2:
            return slice(spec[0], spec[1])
        raise ValueError(f"DRAWS_SLICE tuple must be (start, end), got {spec}")
    spec = str(spec).strip()
    if spec == '' or spec.lower() == 'none':
        return slice(None)
    if ':' not in spec:
        raise ValueError(f"DRAWS_SLICE must be start:end, got {spec!r}")
    start_str, end_str = spec.split(':', 1)
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None
    return slice(start, end)


def resolve_radial_profile_dir(data_dir=None, version_tag=None):
    if data_dir is not None:
        return Path(data_dir)
    if RADIAL_PROFILE_DATA_DIR is not None:
        return Path(RADIAL_PROFILE_DATA_DIR)
    tag = version_tag or VERSION_TAG
    sim_label, _ = parse_version_tag(tag)
    return effective_pred_root(PRED_ROOT, sim_label)


def resolve_draws_data_dir(
    data_dir=None,
    version_tag=None,
    draws_data_dir=None,
    use_script_draws_dir=True,
):
    if draws_data_dir is not None:
        return Path(draws_data_dir)
    if use_script_draws_dir and DRAWS_DATA_DIR is not None:
        return Path(DRAWS_DATA_DIR)
    if data_dir is not None:
        return Path(data_dir)
    if DATA_LAYOUT == "per_galaxy":
        return PRED_ROOT
    if version_tag is not None:
        sim_label, _ = parse_version_tag(version_tag)
        return effective_pred_root(PRED_ROOT, sim_label)
    return Path(DATA_DIR)


def sim_label_for_maps(version_tag):
    sim_label, _ = parse_version_tag(version_tag)
    if sim_label:
        return sim_label
    return "DREAMS"


def sim_display_label(version_tag):
    """Short sim name for R^2 subscripts: DREAMS/TNG50/FIRE (not the raw version tag)."""
    sim = sim_label_for_maps(version_tag)
    return "FIRE" if sim == "FIREm12" else sim


def resolve_sim_for_maps(version_tag):
    """Map version tag to MAP_PATHS key; raise if unknown."""
    sim_for_maps = sim_label_for_maps(version_tag)
    if sim_for_maps not in MAP_PATHS:
        raise ValueError(
            f"No true maps configured for version_tag={version_tag!r} "
            f"(sim={sim_for_maps!r}); choose from {list(MAP_PATHS)}"
        )
    return sim_for_maps


def apply_true_x1_from_maps(structured, version_tag, radial_centers_fallback=None):
    """Replace x1 radial profiles with true values from MAP_PATHS for OOD sims."""
    sim_for_maps = resolve_sim_for_maps(version_tag)
    if sim_for_maps == "DREAMS":
        return structured

    rho_map_path = MAP_PATHS[sim_for_maps]["rhoDM"]
    radial_centers = radial_centers_fallback
    if radial_centers is None and len(structured["radial_centers"]) > 0:
        radial_centers = structured["radial_centers"][0]

    map_structured = build_structured_from_sim_maps(
        rho_map_path,
        structured["galaxy_ids"].tolist(),
        radial_centers=radial_centers,
    )
    map_ids = map_structured["galaxy_ids"]
    id_to_row = {int(gid): i for i, gid in enumerate(map_ids)}

    x1_profiles = structured["x1_profiles"].copy()
    x1_stds = structured["x1_stds"].copy()
    for i, gid in enumerate(structured["galaxy_ids"]):
        row = id_to_row.get(int(gid))
        if row is None:
            print(
                f"Warning: galaxy {gid} out of range for {sim_for_maps} maps "
                f"({len(map_ids)} galaxies); keeping existing x1 profile"
            )
            continue
        x1_profiles[i] = map_structured["x1_profiles"][row]
        x1_stds[i] = map_structured["x1_stds"][row]

    structured = dict(structured)
    structured["x1_profiles"] = x1_profiles
    structured["x1_stds"] = x1_stds
    return structured


def load_structured_data_for_version(
    version_tag,
    galaxy_indices,
    radial_centers_fallback=None,
    file_pattern=None,
):
    """Load radial-profile npz or synthesize true x1 profiles from sim maps."""
    sim_for_maps = resolve_sim_for_maps(version_tag)
    rho_map_path = MAP_PATHS[sim_for_maps]["rhoDM"]

    if radial_profiles_exist(PRED_ROOT, version_tag, galaxy_indices):
        structured = load_all_per_galaxy(
            PRED_ROOT,
            version_tag,
            galaxy_indices=galaxy_indices,
            layout="per_galaxy",
        )
        if sim_for_maps != "DREAMS":
            print(
                f"Replacing x1 profiles with true {sim_for_maps} maps: {rho_map_path}"
            )
            structured = apply_true_x1_from_maps(
                structured,
                version_tag,
                radial_centers_fallback=radial_centers_fallback,
            )
        return structured

    print(
        f"No radial profile npz for version={version_tag!r}; "
        f"using true {sim_for_maps} maps for x1 profiles: {rho_map_path}"
    )
    if galaxy_indices is None:
        galaxy_indices = discover_galaxy_indices_for_version(
            version_tag, file_pattern=file_pattern
        )
    return build_structured_from_sim_maps(
        rho_map_path,
        galaxy_indices,
        radial_centers=radial_centers_fallback,
    )


def discover_galaxy_indices_for_version(
    version_tag,
    num_draws=None,
    file_pattern=None,
):
    from pred_data_loaders import discover_galaxy_indices

    nd = num_draws if num_draws is not None else NUM_DRAWS
    pattern = file_pattern if file_pattern is not None else DRAWS_FILE_PATTERN
    return discover_galaxy_indices(PRED_ROOT, version_tag, nd, file_pattern=pattern)


def effective_draws_file_pattern(version_tag, num_draws, file_pattern=None):
    if file_pattern is not None:
        return file_pattern
    if DRAWS_FILE_PATTERN and DRAWS_FILE_PATTERN != '*pred_draws_10.npz':
        return DRAWS_FILE_PATTERN
    return default_draws_file_pattern(version_tag, num_draws)


def format_draws_slice(draw_slice):
    if draw_slice is None or draw_slice == slice(None):
        return 'all'
    return f"{draw_slice.start}:{draw_slice.stop}"


def extract_draws_from_npz(data, draw_slice, channel):
    draws = data['draws']
    draws = draws[draw_slice]
    if draws.ndim == 4:
        draws = draws[:, channel]
    elif draws.ndim != 3:
        raise ValueError(f"Unexpected draws shape after slice: {draws.shape}")
    return draws


def count_radial_profile_samples(data_dir=None, pred_root=None, version_tag=None):
    if DATA_LAYOUT == "per_galaxy":
        from pred_data_loaders import radial_profile_path

        pred_root = Path(pred_root or PRED_ROOT)
        tag = version_tag or VERSION_TAG
        sim_label, _ = parse_version_tag(tag)
        scan_root = effective_pred_root(pred_root, sim_label)
        count = 0
        if not scan_root.exists():
            return 0
        for entry in sorted(scan_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("galaxy_"):
                continue
            galaxy_idx = int(entry.name.split("_")[1])
            if radial_profile_path(pred_root, galaxy_idx, tag).exists():
                count += 1
        return count

    npz_files = [f for f in os.listdir(data_dir) if f.endswith('radial_profile.npz')]
    return len(npz_files)


def warn_if_second_version_large(n_samples, threshold=SECOND_VERSION_MAX_SAMPLES_WARN):
    if n_samples > threshold:
        print(
            f"WARNING: second version has {n_samples} samples "
            f"(>{threshold}). Overlay may be crowded."
        )


def get_radial_panel_arrays(
    structured_data,
    radial_idx,
    xpred_to_plot,
    xpred_cc_to_plot,
    downsample_factor=1,
    uerr=None,
    lerr=None,
):
    if downsample_factor > 1:
        sl = slice(None, None, downsample_factor)
        x1_values = structured_data['x1_profiles'][sl, radial_idx]
        x1_pred_values = xpred_to_plot[sl, radial_idx]
        x1_std_values = structured_data['x1_stds'][sl, radial_idx]
        x1_pred_std_values = xpred_cc_to_plot[sl, radial_idx]
        x1_pred_uerr_values = uerr[sl, radial_idx] if uerr is not None else None
        x1_pred_lerr_values = lerr[sl, radial_idx] if lerr is not None else None
    else:
        x1_values = structured_data['x1_profiles'][:, radial_idx]
        x1_pred_values = xpred_to_plot[:, radial_idx]
        x1_std_values = structured_data['x1_stds'][:, radial_idx]
        x1_pred_std_values = xpred_cc_to_plot[:, radial_idx]
        x1_pred_uerr_values = uerr[:, radial_idx] if uerr is not None else None
        x1_pred_lerr_values = lerr[:, radial_idx] if lerr is not None else None
    return (
        x1_values,
        x1_pred_values,
        x1_std_values,
        x1_pred_std_values,
        x1_pred_uerr_values,
        x1_pred_lerr_values,
    )


def resolve_xpred_arrays(structured_data, xpred_to_plot, xpred_cc_to_plot):
    if xpred_to_plot is None:
        xpred_to_plot = structured_data['x_pred_profiles']
    if xpred_cc_to_plot is None:
        xpred_cc_to_plot = structured_data['x_pred_stds']
    n_struct = len(structured_data['galaxy_ids'])
    if xpred_to_plot.shape[0] != n_struct:
        raise ValueError(
            f"Pred profile rows ({xpred_to_plot.shape[0]}) do not match "
            f"structured galaxies ({n_struct}); check draw file discovery "
            f"for version tag and DRAWS_FILE_PATTERN"
        )
    return xpred_to_plot, xpred_cc_to_plot


def align_draws_to_structured(all_draws, structured_data):
    """Filter/reorder draw arrays to structured_data galaxy_ids (same order)."""
    target_ids = np.asarray(structured_data['galaxy_ids'])
    source_ids = np.asarray(all_draws['galaxy_ids'])
    if len(source_ids) == len(target_ids) and np.array_equal(source_ids, target_ids):
        return all_draws

    id_to_row = {int(gid): i for i, gid in enumerate(source_ids)}
    rows = []
    missing = []
    for gid in target_ids:
        row = id_to_row.get(int(gid))
        if row is None:
            missing.append(int(gid))
        else:
            rows.append(row)

    if missing:
        raise FileNotFoundError(
            f"Draw files missing for {len(missing)} structured galaxies "
            f"(first few: {missing[:5]})"
        )

    rows = np.array(rows)
    return {
        'galaxy_ids': target_ids.copy(),
        'x_pred_draws': all_draws['x_pred_draws'][rows],
    }


def plot_v2_scatter_overlay(
    ax,
    x,
    y,
    c,
    cmap='viridis_r',
    vmin=None,
    vmax=None,
    marker=None,
    size=None,
    edgecolor=None,
    edgewidth=None,
):
    """Overlay one highlighted galaxy series. Returns the collection, so it can carry
    the shared colour bar when the background scatter is not drawn."""
    return ax.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        marker=V2_MARKER if marker is None else marker,
        s=V2_MARKER_SIZE if size is None else size,
        edgecolors=V2_EDGE_COLOR if edgecolor is None else edgecolor,
        linewidths=V2_EDGE_WIDTH if edgewidth is None else edgewidth,
        alpha=0.95,
        rasterized=True,
        zorder=4,
    )


def plot_dreams_all_contours(ax, x, y, color=DREAMS_ALL_CONTOUR_COLOR):
    """Draw the train/val DREAMS population as density contours rather than points.

    At 922 galaxies per panel the background scatter buries the held-out sample it is
    supposed to provide context for. Contour levels are chosen so each encloses a stated
    fraction of the sample, read off the sorted density rather than assumed Gaussian.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[finite]
    y = np.asarray(y)[finite]
    if x.size < 20 or np.std(x) <= 0 or np.std(y) <= 0:
        print(f"Skipping DREAMS-all contours: {x.size} usable points")
        return None

    kde = stats.gaussian_kde(np.vstack([x, y]))
    pad_x = 0.05 * float(x.max() - x.min())
    pad_y = 0.05 * float(y.max() - y.min())
    gx = np.linspace(x.min() - pad_x, x.max() + pad_x, 120)
    gy = np.linspace(y.min() - pad_y, y.max() + pad_y, 120)
    gxx, gyy = np.meshgrid(gx, gy)
    dens = kde(np.vstack([gxx.ravel(), gyy.ravel()])).reshape(gxx.shape)

    ordered = np.sort(dens.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    levels = []
    for frac in DREAMS_ALL_CONTOUR_LEVELS:
        idx = min(int(np.searchsorted(cumulative, frac)), ordered.size - 1)
        levels.append(float(ordered[idx]))
    levels = sorted(set(levels))
    if not levels:
        return None
    return ax.contour(
        gxx, gyy, dens,
        levels=levels,
        colors=color,
        linewidths=1.0,
        alpha=0.85,
        zorder=1.5,
    )


def compute_deviation_sigmas(
    x1_values,
    x1_pred_values,
    x1_pred_std_values,
    x1_pred_uerr_values=None,
    x1_pred_lerr_values=None,
):
    deviations = x1_pred_values - x1_values
    if x1_pred_uerr_values is None and x1_pred_lerr_values is None:
        return deviations / x1_pred_std_values
    sigmas = np.zeros_like(deviations)
    pos_mask = deviations > 0
    neg_mask = deviations < 0
    sigmas[pos_mask] = deviations[pos_mask] / x1_pred_lerr_values[pos_mask]
    sigmas[neg_mask] = deviations[neg_mask] / x1_pred_uerr_values[neg_mask]
    return sigmas


def histogram_max_density(sigmas, n_bins, xlim=None):
    # Resolved at call time, not bound as a default, so --deviation-hist-xlim reaches it.
    xlim = xlim if xlim is not None else DEVIATION_HIST_XLIM
    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    counts, _ = np.histogram(sigmas, bins=bins, density=True)
    if len(counts) == 0:
        return 0.0
    return float(np.max(counts))


def plot_v2_deviation_histogram(ax, sigmas_v2, max_density_v1, n_bins_v1=DEVIATION_HIST_BINS_V1):
    n_v2 = len(sigmas_v2)
    n_bins_v2 = max(5, n_bins_v1 // 3)
    if n_v2 < n_bins_v1:
        n_bins_v2 = max(5, min(n_bins_v2, max(5, n_v2)))
    bins_v2 = np.linspace(DEVIATION_HIST_XLIM[0], DEVIATION_HIST_XLIM[1], n_bins_v2 + 1)
    counts_v2, _ = np.histogram(sigmas_v2, bins=bins_v2, density=True)
    if counts_v2.max() > 0 and counts_v2.max() > max_density_v1:
        counts_v2 = counts_v2 * (max_density_v1 / counts_v2.max())
    bin_widths = np.diff(bins_v2)
    bin_centers = 0.5 * (bins_v2[:-1] + bins_v2[1:])
    ax.bar(
        bin_centers,
        counts_v2,
        width=bin_widths,
        align='center',
        color=V2_HIST_FILL_COLOR,
        alpha=V2_HIST_ALPHA,
        edgecolor=V2_HIST_EDGE_COLOR,
        linewidth=1.0,
        zorder=2,
    )


def needs_draws_profiles():
    return (
        COMPUTE_DRAWS_PROFILES
        or PLOT_PRED_VS_TRUE_MEDIAN_CC
        or PLOT_PRED_VS_TRUE_MEDIAN_CC_CENTER
        or PLOT_PRED_VS_TRUE_MEDIAN_CC_ERRORBARS
        or PLOT_STD_PRED_VS_TRUE_MEDIAN_CC
        or PLOT_DEVIATION_DISTRIBUTION
        or PLOT_PER_GALAXY_DEVIATION
        or PLOT_PER_GALAXY_DRAW_PANELS
        or PLOT_PER_GALAXY_PRIOR_POSTERIOR
    )


def dreams_all_suffix():
    """Filename tag separating the DREAMS-all rendering variants of the scatter figure.

    Empty unless the held-out galaxies are actually being highlighted, since the
    background-series style has no effect on the out-of-domain (TNG50/FIRE) figures.
    """
    if HIGHLIGHT_GALAXY_IDS is None or DREAMS_ALL_STYLE == 'points':
        return ''
    return f'_dreamsall_{DREAMS_ALL_STYLE}'


def save_figure(fig, path, tight=True):
    save_paper_figure(fig, path, dpi=SAVE_DPI, tight=tight)


def load_all_data_structured(data_dir, idx_list=None, version_tag=None):
    """Load all radial-profile npz files into structured arrays."""
    if DATA_LAYOUT == "per_galaxy":
        tag = version_tag or VERSION_TAG
        return load_all_per_galaxy(
            PRED_ROOT,
            tag,
            galaxy_indices=idx_list,
            layout="per_galaxy",
        )

    npz_files = [f for f in os.listdir(data_dir) if f.endswith('radial_profile.npz')]

    print(f"Loading {len(npz_files)} radial profile files...")

    galaxy_ids = []
    radial_centers_all = []
    x0_profiles_all = []
    x1_profiles_all = []
    x_pred_profiles_all = []
    x0_stds_all = []
    x1_stds_all = []
    x_pred_stds_all = []

    for filename in npz_files:
        filepath = os.path.join(data_dir, filename)
        data = np.load(filepath)
        galaxy_id = int(filename.split('_')[1])

        galaxy_ids.append(galaxy_id)
        radial_centers_all.append(data['radial_centers'])
        x0_profiles_all.append(data['x0_radial_profile'])
        x1_profiles_all.append(data['x1_radial_profile'])
        x_pred_profiles_all.append(data['x_pred_radial_profile'])
        x0_stds_all.append(data['x0_radial_profile_std'])
        x1_stds_all.append(data['x1_radial_profile_std'])
        x_pred_stds_all.append(data['x_pred_radial_profile_std'])
        data.close()

    galaxy_ids = np.array(galaxy_ids)
    sort_idx = np.argsort(galaxy_ids)
    galaxy_ids = galaxy_ids[sort_idx]
    radial_centers_all = np.array(radial_centers_all)[sort_idx]
    x0_profiles_all = np.array(x0_profiles_all)[sort_idx]
    x1_profiles_all = np.array(x1_profiles_all)[sort_idx]
    x_pred_profiles_all = np.array(x_pred_profiles_all)[sort_idx]
    x0_stds_all = np.array(x0_stds_all)[sort_idx]
    x1_stds_all = np.array(x1_stds_all)[sort_idx]
    x_pred_stds_all = np.array(x_pred_stds_all)[sort_idx]

    if idx_list is not None:
        print(f"Selecting {len(idx_list)} galaxies by index")
        galaxy_ids = galaxy_ids[idx_list]
        radial_centers_all = radial_centers_all[idx_list]
        x0_profiles_all = x0_profiles_all[idx_list]
        x1_profiles_all = x1_profiles_all[idx_list]
        x_pred_profiles_all = x_pred_profiles_all[idx_list]
        x0_stds_all = x0_stds_all[idx_list]
        x1_stds_all = x1_stds_all[idx_list]
        x_pred_stds_all = x_pred_stds_all[idx_list]

    return {
        'galaxy_ids': np.array(galaxy_ids),
        'radial_centers': np.array(radial_centers_all),
        'x0_profiles': np.array(x0_profiles_all),
        'x1_profiles': np.array(x1_profiles_all),
        'x_pred_profiles': np.array(x_pred_profiles_all),
        'x0_stds': np.array(x0_stds_all),
        'x1_stds': np.array(x1_stds_all),
        'x_pred_stds': np.array(x_pred_stds_all),
    }


def load_pred_data_structured(
    data_dir,
    idx_list=None,
    file_pattern=None,
    draw_slice=None,
    channel=None,
    version_tag=None,
    num_draws=None,
    output_suffix=None,
):
    """Load pred draws npz files and rescale to rhoDM physical scale."""
    if draw_slice is None:
        draw_slice = parse_draws_slice(DRAWS_SLICE)
    if channel is None:
        channel = DRAWS_CHANNEL

    if DATA_LAYOUT == "per_galaxy":
        tag = version_tag or VERSION_TAG
        nd = num_draws if num_draws is not None else NUM_DRAWS
        pattern = file_pattern if file_pattern is not None else DRAWS_FILE_PATTERN
        pred_root = Path(data_dir) if data_dir is not None else PRED_ROOT
        sim_label, _ = parse_version_tag(tag)
        if sim_label and pred_root.name == sim_label:
            pred_root = pred_root.parent
        return load_pred_per_galaxy(
            pred_root,
            tag,
            num_draws=nd,
            file_pattern=pattern,
            galaxy_indices=idx_list,
            draw_slice=draw_slice,
            channel=channel,
            layout="per_galaxy",
        )

    if file_pattern is None:
        file_pattern = DRAWS_FILE_PATTERN

    npz_files = sorted(
        f for f in os.listdir(data_dir) if fnmatch.fnmatch(f, file_pattern)
    )
    if not npz_files:
        raise FileNotFoundError(
            f"No files matching {file_pattern!r} in {data_dir}"
        )

    print(
        f"Loading {len(npz_files)} pred draws files "
        f"(pattern={file_pattern!r}, slice=[{format_draws_slice(draw_slice)}])..."
    )

    galaxy_ids = []
    x_pred_draws_all = []

    mean_rhoDM = NORM_DICT['2D']['rhoDM'][128]['mean']
    std_rhoDM = NORM_DICT['2D']['rhoDM'][128]['std']

    for filename in npz_files:
        filepath = os.path.join(data_dir, filename)
        data = np.load(filepath)
        galaxy_id = int(filename.split('_')[1])

        draws_temp = extract_draws_from_npz(data, draw_slice, channel)
        draws_temp = draws_temp * std_rhoDM + mean_rhoDM
        galaxy_ids.append(galaxy_id)
        x_pred_draws_all.append(draws_temp)
        data.close()

    galaxy_ids = np.array(galaxy_ids)
    n_draws_per_galaxy = {arr.shape[0] for arr in x_pred_draws_all}
    if len(n_draws_per_galaxy) != 1:
        raise ValueError(
            f"Inconsistent number of draws after slicing: {n_draws_per_galaxy}"
        )
    x_pred_draws_all = np.array(x_pred_draws_all)
    sort_idx = np.argsort(galaxy_ids)
    x_pred_draws_all = x_pred_draws_all[sort_idx]
    galaxy_ids = galaxy_ids[sort_idx]

    if idx_list is not None:
        galaxy_ids = galaxy_ids[idx_list]
        x_pred_draws_all = x_pred_draws_all[idx_list]

    print(f"Pred draws array shape: {x_pred_draws_all.shape}")

    return {
        'galaxy_ids': np.array(galaxy_ids),
        'x_pred_draws': np.array(x_pred_draws_all),
    }


def build_radial_annulus_geometry(
    resolution,
    xmin=-32,
    xmax=32,
    ymin=-32,
    ymax=32,
):
    r_min = 0.0
    r_max = float(np.sqrt(xmax**2 + ymax**2))
    num_bins = max(1, int(resolution // 4))
    radial_edges = np.linspace(r_min, r_max, num_bins + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")

    mask = np.full((num_bins, resolution, resolution), False)
    for i in range(num_bins):
        r1 = radial_edges[i]
        r2 = radial_edges[i + 1]
        if i < num_bins - 1:
            in_bin = ((x_coords**2 + y_coords**2) >= r1**2) & (
                (x_coords**2 + y_coords**2) < r2**2
            )
        else:
            in_bin = ((x_coords**2 + y_coords**2) >= r1**2) & (
                (x_coords**2 + y_coords**2) <= r2**2
            )
        if in_bin.any():
            mask[i] = in_bin

    return radial_edges, radial_centers, mask


def compute_radial_profile_from_map(map_2d, mask):
    num_bins = mask.shape[0]
    profile = np.full(num_bins, np.nan)
    for j in range(num_bins):
        vals = map_2d[mask[j]]
        if vals.size > 0:
            profile[j] = aggregate_pixels(vals)
    return profile


def compute_radial_profiles_from_map_stack(maps_stack, mask):
    n_maps = maps_stack.shape[0]
    num_bins = mask.shape[0]
    profiles = np.full((n_maps, num_bins), np.nan)
    for d in range(n_maps):
        profiles[d] = compute_radial_profile_from_map(maps_stack[d], mask)
    return profiles


def _rasterize_axes_artists(ax):
    """Blanket rasterization of every child artist.

    Kept for callers that genuinely want it, but prefer _rasterize_below() for anything headed
    for PDF: this also rasterizes Text, so axis labels and legends come out as bitmaps.
    """
    for artist in ax.get_children():
        if hasattr(artist, 'set_rasterized'):
            artist.set_rasterized(True)


def _rasterize_below(ax, zorder):
    """Rasterize dense artists while leaving text, legend and spines as vector.

    Only affects vector output (PDF/SVG); a no-op for PNG. Relies on the caller having laid out
    zorders so the heavy collections sit below `zorder` and the annotation sits above it.
    """
    ax.set_rasterization_zorder(zorder)


def _thin_for_plotting(profiles, max_lines):
    """Evenly spaced subset of rows, for rendering only.

    Deterministic rather than randomly sampled so the same figure is reproducible across runs.
    Never use the result for a statistic -- medians and percentiles must see every draw.
    """
    n = len(profiles)
    if max_lines is None or n <= max_lines:
        return profiles
    return profiles[np.linspace(0, n - 1, max_lines, dtype=int)]


def _add_posterior_draw_lines(ax, radial_centers, profiles, color, zorder=4,
                              max_lines=PRIOR_POSTERIOR_MAX_POSTERIOR_LINES):
    """Posterior draws as one LineCollection instead of a loop of ax.plot calls.

    Mirrors how the prior background lines are drawn a few lines above: subsample, thin, single
    artist. At 100 draws the old per-draw loop produced 100 artists at linewidth 2.8 / alpha
    0.55, which saturates to an opaque block.
    """
    shown = _thin_for_plotting(profiles, max_lines)
    segments = [np.column_stack([radial_centers, prof]) for prof in shown]
    ax.add_collection(LineCollection(
        segments,
        colors=color,
        alpha=PRIOR_POSTERIOR_DRAW_ALPHA,
        linewidths=PRIOR_POSTERIOR_DRAW_LINEWIDTH,
        rasterized=True,
        zorder=zorder,
    ))
    return len(shown)


def _style_standard_axis(ax, labelsize=None):
    ax.tick_params(
        axis='both',
        which='both',
        direction='in',
        top=True,
        right=True,
        labelsize=AXIS_TICK_LABEL_FONTSIZE if labelsize is None else labelsize,
    )


def _new_residual_panel():
    """A blank panel sized and margined to stack with its siblings in LaTeX."""
    fig = plt.figure(figsize=RESIDUAL_PANEL_FIGSIZE)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(**RESIDUAL_PANEL_MARGINS)
    return fig, ax


def _finish_residual_panel(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=RESIDUAL_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=RESIDUAL_PANEL_LABEL_FONTSIZE)
    _style_standard_axis(ax, labelsize=RESIDUAL_PANEL_TICK_FONTSIZE)


def _style_colorbar_ticks(cbar):
    cbar.ax.tick_params(direction='in', labelsize=AXIS_TICK_LABEL_FONTSIZE)


def plot_galaxy_prior_posterior_radial_profile(
    radial_centers,
    true_profile,
    other_true_profiles,
    posterior_profiles,
    galaxy_id,
    save_stem=None,
    save_dpi=200,
    show_background_lines=PRIOR_POSTERIOR_SHOW_BACKGROUND_LINES,
    max_background_lines=PRIOR_POSTERIOR_MAX_BACKGROUND_LINES,
    posterior_profiles_2=None,
    posterior_label='Prediction draws',
    posterior_label_2=None,
):
    prior_mean = np.nanmean(other_true_profiles, axis=0)
    prior_std = np.nanstd(other_true_profiles, axis=0)
    posterior_median = combine_draws(posterior_profiles, axis=0, nan_safe=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    if show_background_lines and len(other_true_profiles) > 0:
        n_lines = len(other_true_profiles)
        if n_lines > max_background_lines:
            line_idx = np.linspace(0, n_lines - 1, max_background_lines, dtype=int)
            background_profiles = other_true_profiles[line_idx]
        else:
            background_profiles = other_true_profiles
        segments = [
            np.column_stack([radial_centers, prof])
            for prof in background_profiles
        ]
        lc = LineCollection(
            segments,
            colors='0.72',
            alpha=0.30,
            linewidths=0.7,
            rasterized=True,
            zorder=1,
        )
        ax.add_collection(lc)
        # The collection's own legend key inherits its 0.7 pt width at alpha 0.30, which
        # is all but invisible at print size. Carry the label on an empty proxy line
        # instead, so the key reads clearly while the plotted profiles stay faint. Added
        # here rather than at the legend call so it keeps its place in the label order.
        ax.plot(
            [], [],
            color='0.55',
            linewidth=2.6,
            solid_capstyle='butt',
            label='Prior samples',
        )

    # ax.fill_between(
    #     radial_centers,
    #     prior_mean - prior_std,
    #     prior_mean + prior_std,
    #     color='0.82',
    #     alpha=0.45,
    #     rasterized=True,
    #     zorder=2,
    # )
    ax.plot(
        radial_centers,
        prior_mean,
        color='0.45',
        linewidth=1.5,
        linestyle='--',
        rasterized=True,
        zorder=3,
        label='Prior mean',
    )

    # posterior_median above is computed from ALL draws; only the rendering is thinned.
    n_shown = _add_posterior_draw_lines(
        ax, radial_centers, posterior_profiles, PER_GALAXY_V1_COLOR
    )

    ax.plot(
        radial_centers,
        posterior_median,
        color=PER_GALAXY_V1_COLOR,
        linewidth=2.8,
        linestyle='-',
        rasterized=True,
        zorder=5,
        label=(
            posterior_label
            if n_shown == len(posterior_profiles)
            else f'{posterior_label} ({n_shown} of {len(posterior_profiles)} shown)'
        ),
    )

    if posterior_profiles_2 is not None:
        posterior_median_2 = combine_draws(posterior_profiles_2, axis=0, nan_safe=True)
        n_shown_2 = _add_posterior_draw_lines(
            ax, radial_centers, posterior_profiles_2, PER_GALAXY_V2_COLOR
        )
        label_2 = posterior_label_2 or 'Prediction draws (v2)'
        ax.plot(
            radial_centers,
            posterior_median_2,
            color=PER_GALAXY_V2_COLOR,
            linewidth=2.8,
            linestyle='-',
            rasterized=True,
            zorder=5,
            label=(
                label_2
                if n_shown_2 == len(posterior_profiles_2)
                else f'{label_2} ({n_shown_2} of {len(posterior_profiles_2)} shown)'
            ),
        )

    ax.plot(
        radial_centers,
        true_profile,
        color='black',
        linewidth=3.0,
        rasterized=True,
        zorder=6,
        label='True',
    )

    ax.set_xlabel('Radius [kpc]', fontsize=PRIOR_POSTERIOR_LABEL_FONTSIZE)
    _style_standard_axis(ax, labelsize=PRIOR_POSTERIOR_TICK_FONTSIZE)
    ax.set_ylabel(
        'log $\\rho_{\\mathrm{DM}}$ [$M_{\\odot}\\,\\mathrm{kpc}^{-3}$]',
        fontsize=PRIOR_POSTERIOR_LABEL_FONTSIZE,
    )
    # ax.set_title(
    #     f'Prior vs posterior radial profile (galaxy {galaxy_id})',
    #     fontsize=15,
    # )
    paper_grid(ax, alpha=0.25)
    ax.legend(loc='upper right', fontsize=PRIOR_POSTERIOR_LEGEND_FONTSIZE, framealpha=0.9)
    _style_standard_axis(ax)

    y_arrays = [
        true_profile,
        posterior_profiles.ravel(),
        prior_mean - prior_std,
        prior_mean + prior_std,
    ]
    if posterior_profiles_2 is not None:
        y_arrays.append(posterior_profiles_2.ravel())
    y_vals = np.concatenate(y_arrays)
    y_vals = y_vals[np.isfinite(y_vals)]
    y_margin = 0.08 * (np.max(y_vals) - np.min(y_vals))
    ax.set_xlim(radial_centers[0], radial_centers[-1])
    ax.set_ylim(np.min(y_vals) - y_margin, np.max(y_vals) + y_margin)

    # Prior lines z=1, prior mean z=3, posterior draws z=4, medians z=5, truth z=6: rasterizing
    # below 4.5 catches the dense collections and leaves the medians, truth and text vector.
    _rasterize_below(ax, PRIOR_POSTERIOR_RASTER_ZORDER)
    fig.tight_layout()

    if save_stem is not None:
        out_path = f"{save_stem}_prior_posterior_radial.png"
        save_paper_figure(fig, out_path, dpi=save_dpi)
    else:
        plt.show()

    return {
        'prior_mean': prior_mean,
        'prior_std': prior_std,
        'posterior_median': posterior_median,
    }


def _warn_if_draw_nans(draws, context="prediction draws"):
    n_nan = int(np.isnan(draws).sum())
    if n_nan:
        print(
            f"Warning: {context} contain {n_nan} NaN values "
            f"({100.0 * n_nan / draws.size:.2f}% of elements); "
            f"using NaN-safe aggregation for median/std"
        )
    return n_nan > 0


def _r_squared(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1 - (ss_res / ss_tot)


def _escape_mathtext_label(label):
    # Escape literal underscores (e.g. in a version-tag fallback label like
    # "v95_heun100") so mathtext doesn't parse them as nested subscripts.
    return label.replace('_', r'\_')


def _r2_line(label, value):
    label_escaped = _escape_mathtext_label(label)
    return f'\n$R^2_{{\\mathrm{{{label_escaped}}}}}$ = {value:.2f}'


def _delta_line(label, mean, std):
    label_escaped = _escape_mathtext_label(label)
    return f'\n$\\Delta_{{\\mathrm{{{label_escaped}}}}}$={mean:.2f}$\\pm${std:.2f}'


def _dual_panel_labels():
    """Short (DREAMS/TNG50/FIRE) v1/v2 labels for dual-panel text (R^2, Delta); falls
    back to version-tag labels if both sides resolve to the same sim (e.g. DREAMS vs
    DREAMS across two model versions), since the sim name alone would be ambiguous."""
    label_v1 = sim_display_label(VERSION_TAG)
    label_v2 = sim_display_label(VERSION_TAG_2)
    if label_v1 == label_v2:
        label_v1 = short_version_label(VERSION_TAG, DRAWS_OUTPUT_LABEL)
        label_v2 = short_version_label(VERSION_TAG_2, DRAWS_OUTPUT_LABEL_2)
    return label_v1, label_v2


def _finite_axis_limits(arrays, fallback=(0.0, 1.0)):
    vals = np.concatenate([np.ravel(a) for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    return float(np.min(vals)), float(np.max(vals))


def _radial_bin_draw_stats(draws_temp_j, use_nan_safe):
    # Returns, in order: the annulus profile value (pixel aggregate of the per-pixel
    # ensemble combination), its spatial scatter, the per-pixel 84th/16th percentile
    # pixel aggregates and their spatial scatters, and finally the DRAW-TO-DRAW
    # spread of the annulus-averaged profile itself.
    #
    # That last quantity is the statistically matched uncertainty for a
    # profile-level residual: the per-pixel percentile width does not shrink when
    # pixels are averaged, whereas the error on the annulus average does. Using
    # the per-pixel width as the denominator inflates it by ~1.8x at these radii
    # and correspondingly understates the significance of profile-level residuals.
    if draws_temp_j.size == 0:
        return None
    if use_nan_safe:
        if not np.isfinite(draws_temp_j).any():
            return None
        pixel_central = combine_draws(draws_temp_j, axis=0, nan_safe=True)
        upper_perpixel = np.nanpercentile(draws_temp_j, 84, axis=0)
        lower_perpixel = np.nanpercentile(draws_temp_j, 16, axis=0)
        per_draw_profile = aggregate_pixels(draws_temp_j, axis=1, nan_safe=True)
        return (
            aggregate_pixels(pixel_central, nan_safe=True),
            np.nanstd(pixel_central),
            aggregate_pixels(upper_perpixel, nan_safe=True),
            aggregate_pixels(lower_perpixel, nan_safe=True),
            np.nanstd(upper_perpixel),
            np.nanstd(lower_perpixel),
            np.nanstd(per_draw_profile),
        )

    pixel_central = combine_draws(draws_temp_j, axis=0)
    upper_perpixel = np.percentile(draws_temp_j, 84, axis=0)
    lower_perpixel = np.percentile(draws_temp_j, 16, axis=0)
    per_draw_profile = aggregate_pixels(draws_temp_j, axis=1)
    return (
        aggregate_pixels(pixel_central),
        np.std(pixel_central),
        aggregate_pixels(upper_perpixel),
        aggregate_pixels(lower_perpixel),
        np.std(upper_perpixel),
        np.std(lower_perpixel),
        np.std(per_draw_profile),
    )


def compute_radial_profiles_from_draws(
    draws,
    xmin=-32,
    xmax=32,
    ymin=-32,
    ymax=32,
):
    n_galaxies = draws.shape[0]
    n_draws = draws.shape[1]
    resolution = draws.shape[2]

    r_min = 0.0
    r_max = float(np.sqrt(xmax**2 + ymax**2))
    num_bins = max(1, int(resolution // 4))
    radial_edges = np.linspace(r_min, r_max, num_bins + 1)

    print(
        f"n_galaxies: {n_galaxies}, n_draws: {n_draws}, "
        f"resolution: {resolution}, num_bins: {num_bins}"
    )
    has_draw_nans = _warn_if_draw_nans(draws)

    x_pred_radial_profile_combined = np.full((n_galaxies, num_bins), np.nan)
    x_pred_radial_profile_median_std = np.full((n_galaxies, num_bins), np.nan)
    x_pred_radial_profile_upper_perpixel_agg = np.full((n_galaxies, num_bins), np.nan)
    x_pred_radial_profile_lower_perpixel_agg = np.full((n_galaxies, num_bins), np.nan)
    x_pred_radial_profile_upper_perpixel_std = np.full((n_galaxies, num_bins), np.nan)
    x_pred_radial_profile_lower_perpixel_std = np.full((n_galaxies, num_bins), np.nan)
    x_pred_radial_profile_std_draws_profile = np.full((n_galaxies, num_bins), np.nan)

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")

    mask = np.full((num_bins, resolution, resolution), False)
    for i in range(num_bins):
        r1 = radial_edges[i]
        r2 = radial_edges[i + 1]
        if i < num_bins - 1:
            in_bin = ((x_coords**2 + y_coords**2) >= r1**2) & (
                (x_coords**2 + y_coords**2) < r2**2
            )
        else:
            in_bin = ((x_coords**2 + y_coords**2) >= r1**2) & (
                (x_coords**2 + y_coords**2) <= r2**2
            )
        if in_bin.any():
            mask[i] = in_bin

    for i in range(n_galaxies):
        draws_temp = draws[i]
        for j in range(num_bins):
            draws_temp_j = draws_temp[:, mask[j]]
            use_nan_safe = has_draw_nans and np.isnan(draws_temp_j).any()
            bin_stats = _radial_bin_draw_stats(draws_temp_j, use_nan_safe)
            if bin_stats is None:
                continue
            (
                x_pred_radial_profile_combined[i, j],
                x_pred_radial_profile_median_std[i, j],
                x_pred_radial_profile_upper_perpixel_agg[i, j],
                x_pred_radial_profile_lower_perpixel_agg[i, j],
                x_pred_radial_profile_upper_perpixel_std[i, j],
                x_pred_radial_profile_lower_perpixel_std[i, j],
                x_pred_radial_profile_std_draws_profile[i, j],
            ) = bin_stats

    return (
        x_pred_radial_profile_combined,
        x_pred_radial_profile_median_std,
        x_pred_radial_profile_upper_perpixel_agg,
        x_pred_radial_profile_lower_perpixel_agg,
        x_pred_radial_profile_upper_perpixel_std,
        x_pred_radial_profile_lower_perpixel_std,
        x_pred_radial_profile_std_draws_profile,
    )


def compute_and_derive_draw_profiles(x_pred_draws):
    (
        x_pred_radial_profile_combined,
        x_pred_radial_profile_median_std,
        x_pred_radial_profile_upper_perpixel_agg,
        x_pred_radial_profile_lower_perpixel_agg,
        _upper_std,
        _lower_std,
        x_pred_radial_profile_std_draws_profile,
    ) = compute_radial_profiles_from_draws(x_pred_draws)

    x_pred_radial_profile_std_perpixel_agg = (
        x_pred_radial_profile_upper_perpixel_agg - x_pred_radial_profile_lower_perpixel_agg
    ) / 2
    x_pred_radial_profile_uerr_perpixel_agg = (
        x_pred_radial_profile_upper_perpixel_agg - x_pred_radial_profile_combined
    )
    x_pred_radial_profile_lerr_perpixel_agg = (
        x_pred_radial_profile_combined - x_pred_radial_profile_lower_perpixel_agg
    )

    return {
        'combined_profile': x_pred_radial_profile_combined,
        'median_std': x_pred_radial_profile_median_std,
        'std_perpixel_agg': x_pred_radial_profile_std_perpixel_agg,
        'std_draws_profile': x_pred_radial_profile_std_draws_profile,
        'uerr_perpixel_agg': x_pred_radial_profile_uerr_perpixel_agg,
        'lerr_perpixel_agg': x_pred_radial_profile_lerr_perpixel_agg,
    }


# Denominator used for profile-level Delta residuals.
# 'profile'  : draw-to-draw spread of the annulus-averaged profile (matched to the
#              profile-level numerator; correct for calibration statements)
# 'perpixel' : per-pixel (P84-P16)/2 medianed over the annulus (legacy behavior,
#              reproduces figures generated before 2026-07-31)
DEVIATION_DENOMINATOR = 'profile'


def deviation_denominator(draw_profiles, mode=None):
    mode = mode or DEVIATION_DENOMINATOR
    if mode == 'profile':
        return draw_profiles['std_draws_profile']
    if mode == 'perpixel':
        return draw_profiles['std_perpixel_agg']
    raise ValueError(
        f"Unknown deviation denominator {mode!r}; use 'profile' or 'perpixel'"
    )


def plot_multiple_radial_centers(
    radial_indices,
    structured_data,
    n_cols=3,
    xpred_to_plot=None,
    xpred_cc_to_plot=None,
    min_val_fixed=None,
    max_val_fixed=None,
    downsample_factor=1,
    do_error_bars=False,
    uerr=None,
    lerr=None,
    axis_llim=None,
    axis_ulim=None,
    overlay=None,
    highlight_galaxy_ids=None,
):
    n_plots = len(radial_indices)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    radial_centers = structured_data['radial_centers'][0]
    vmin, vmax = 0.05, 0.2

    xpred_to_plot, xpred_cc_to_plot = resolve_xpred_arrays(
        structured_data, xpred_to_plot, xpred_cc_to_plot
    )
    if overlay is not None:
        overlay['xpred_to_plot'], overlay['xpred_cc_to_plot'] = resolve_xpred_arrays(
            overlay['structured_data'],
            overlay.get('xpred_to_plot'),
            overlay.get('xpred_cc_to_plot'),
        )
    if highlight_galaxy_ids is not None and overlay is not None:
        print("Warning: highlight_galaxy_ids ignored because overlay is set")
        highlight_galaxy_ids = None

    scatter = None
    for i, radial_idx in enumerate(radial_indices):
        ax = axes[i]

        (
            x1_values,
            x1_pred_values,
            x1_std_values,
            x1_pred_std_values,
            x1_pred_uerr_values,
            x1_pred_lerr_values,
        ) = get_radial_panel_arrays(
            structured_data,
            radial_idx,
            xpred_to_plot,
            xpred_cc_to_plot,
            downsample_factor=downsample_factor,
            uerr=uerr,
            lerr=lerr,
        )

        show_background_points = (
            highlight_galaxy_ids is None or DREAMS_ALL_STYLE == 'points'
        )
        if show_background_points:
            scatter = ax.scatter(
                x1_values,
                x1_pred_values,
                alpha=0.6,
                s=10,
                c=x1_pred_std_values,
                cmap='viridis_r',
                vmin=vmin,
                vmax=vmax,
                # 1024 galaxies x 6 radial bins of points; rasterizing keeps the eventual PDF from
                # carrying every marker as a separate vector path. No effect on PNG output.
                rasterized=True,
            )
        elif DREAMS_ALL_STYLE == 'contours':
            plot_dreams_all_contours(ax, x1_values, x1_pred_values)
        if do_error_bars:
            if uerr is not None and lerr is not None:
                ax.errorbar(
                    x1_values,
                    x1_pred_values,
                    xerr=x1_std_values,
                    yerr=[x1_pred_lerr_values, x1_pred_uerr_values],
                    linestyle='none',
                    ecolor='k',
                    alpha=0.5,
                )
            else:
                ax.errorbar(
                    x1_values,
                    x1_pred_values,
                    xerr=x1_std_values,
                    yerr=x1_pred_std_values,
                    linestyle='none',
                    ecolor='k',
                    alpha=0.5,
                )

        highlight_mask = None
        if highlight_galaxy_ids is not None:
            gal_ids_panel = structured_data['galaxy_ids'][slice(None, None, downsample_factor)]
            highlight_mask = np.isin(gal_ids_panel, highlight_galaxy_ids)
            if highlight_mask.any():
                highlight_scatter = plot_v2_scatter_overlay(
                    ax,
                    x1_values[highlight_mask],
                    x1_pred_values[highlight_mask],
                    c=x1_pred_std_values[highlight_mask],
                    cmap='viridis_r',
                    vmin=vmin,
                    vmax=vmax,
                    marker=HIGHLIGHT_MARKER,
                    size=HIGHLIGHT_MARKER_SIZE,
                    edgecolor=HIGHLIGHT_EDGE_COLOR,
                    edgewidth=HIGHLIGHT_EDGE_WIDTH,
                )
                # The colour bar is normally carried by the background scatter; when that
                # is hidden or replaced by contours it has to come from here instead.
                if not show_background_points:
                    scatter = highlight_scatter

        if overlay is not None:
            (
                x1_v2,
                x1_pred_v2,
                _x1_std_v2,
                _x1_pred_std_v2,
                _uerr_v2,
                _lerr_v2,
            ) = get_radial_panel_arrays(
                overlay['structured_data'],
                radial_idx,
                overlay['xpred_to_plot'],
                overlay['xpred_cc_to_plot'],
                downsample_factor=downsample_factor,
                uerr=overlay.get('uerr'),
                lerr=overlay.get('lerr'),
            )
            plot_v2_scatter_overlay(
                ax,
                x1_v2,
                x1_pred_v2,
                c=_x1_pred_std_v2,
                cmap='viridis_r',
                vmin=vmin,
                vmax=vmax,
            )

        if min_val_fixed is not None:
            min_val = min_val_fixed
        elif axis_llim is not None:
            min_val = axis_llim[i]
        else:
            min_val = min(np.min(x1_values), np.min(x1_pred_values))
        if max_val_fixed is not None:
            max_val = max_val_fixed
        elif axis_ulim is not None:
            max_val = axis_ulim[i]
        else:
            max_val = max(np.max(x1_values), np.max(x1_pred_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.8)

        if axis_llim is not None and axis_ulim is not None:
            ax.set_xlim(axis_llim[i], axis_ulim[i])
            ax.set_ylim(axis_llim[i], axis_ulim[i])

        r_squared = _r_squared(x1_values, x1_pred_values)
        r_squared_highlight = None
        if highlight_mask is not None and highlight_mask.sum() > 1:
            r_squared_highlight = _r_squared(
                x1_values[highlight_mask], x1_pred_values[highlight_mask]
            )
        r_squared_v2 = None
        if overlay is not None:
            r_squared_v2 = _r_squared(x1_v2, x1_pred_v2)

        chi2 = np.sum((x1_values - x1_pred_values) ** 2 / x1_pred_std_values ** 2)
        p_value = 1 - stats.chi2.cdf(chi2, len(x1_values) - 1)
        print(f"r={radial_centers[radial_idx]:.1f} kpc: chi2={chi2:.3f}, p-value={p_value:.3f}")

        max_deviation = np.max(np.abs(x1_values - x1_pred_values) / x1_pred_std_values)
        print(f"  max deviation = {max_deviation:.3f}-sigma")

        if i % n_cols == 0:
            ax.set_ylabel(r'log($\rho_{\mathrm{DM}}$) Predicted', fontsize=AXIS_LABEL_FONTSIZE)
        if i >= n_cols * (n_rows - 1):
            ax.set_xlabel(r'log($\rho_{\mathrm{DM}}$) True', fontsize=AXIS_LABEL_FONTSIZE)
        panel_text = f'$r=${radial_centers[radial_idx]:.1f} kpc'
        if SHOW_R2:
            if r_squared_highlight is not None:
                panel_text += _r2_line('DREAMS,test', r_squared_highlight)
                if DREAMS_ALL_STYLE != 'hide':
                    panel_text += _r2_line('DREAMS,all', r_squared)
            elif r_squared_v2 is not None:
                label_v1, label_v2 = _dual_panel_labels()
                panel_text += _r2_line(label_v1, r_squared)
                panel_text += _r2_line(label_v2, r_squared_v2)
            else:
                panel_text += f'\n$R^2=${r_squared:.2f}'
        ax.text(
            0.05,
            0.95,
            panel_text,
            transform=ax.transAxes,
            fontsize=IN_PLOT_TEXTBOX_FONTSIZE,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
            zorder=5,
        )
        paper_grid(ax, alpha=0.3)
        _style_standard_axis(ax)
        if not do_error_bars:
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.set_aspect('equal', adjustable='datalim')

    if not do_error_bars and scatter is not None:
        cbar = fig.colorbar(
            scatter, ax=axes, orientation='vertical', fraction=0.025, pad=0.01
        )
        cbar.set_label('$\\sigma$ log($\\rho_{\\mathrm{DM}}$)', fontsize=AXIS_LABEL_FONTSIZE)
        _style_colorbar_ticks(cbar)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    return fig, axes


def plot_multiple_radial_centers_scatter(
    radial_indices,
    structured_data,
    n_cols=3,
    xpred_to_plot=None,
    xpred_cc_to_plot=None,
    min_val_fixed=None,
    max_val_fixed=None,
    overlay=None,
):
    n_plots = len(radial_indices)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    radial_centers = structured_data['radial_centers'][0]
    vmin, vmax = 5.5, 10.0

    xpred_to_plot, xpred_cc_to_plot = resolve_xpred_arrays(
        structured_data, xpred_to_plot, xpred_cc_to_plot
    )
    if overlay is not None:
        overlay['xpred_to_plot'], overlay['xpred_cc_to_plot'] = resolve_xpred_arrays(
            overlay['structured_data'],
            overlay.get('xpred_to_plot'),
            overlay.get('xpred_cc_to_plot'),
        )

    scatter = None
    for i, radial_idx in enumerate(radial_indices):
        ax = axes[i]

        x1_values = structured_data['x1_profiles'][:, radial_idx]
        x1_std_values = structured_data['x1_stds'][:, radial_idx]
        x1_pred_std_values = xpred_cc_to_plot[:, radial_idx]

        scatter = ax.scatter(
            x1_std_values,
            x1_pred_std_values,
            alpha=0.6,
            s=10,
            c=x1_values,
            cmap='viridis_r',
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        if overlay is not None:
            x1_v2 = overlay['structured_data']['x1_profiles'][:, radial_idx]
            x1_std_v2 = overlay['structured_data']['x1_stds'][:, radial_idx]
            x1_pred_std_v2 = overlay['xpred_cc_to_plot'][:, radial_idx]
            plot_v2_scatter_overlay(
                ax,
                x1_std_v2,
                x1_pred_std_v2,
                c=x1_v2,
                cmap='viridis_r',
                vmin=vmin,
                vmax=vmax,
            )

        if min_val_fixed is not None:
            min_val = min_val_fixed
        else:
            limit_arrays = [x1_std_values, x1_pred_std_values]
            if overlay is not None:
                limit_arrays.extend([x1_std_v2, x1_pred_std_v2])
            min_val, _ = _finite_axis_limits(limit_arrays)
        if max_val_fixed is not None:
            max_val = max_val_fixed
        else:
            limit_arrays = [x1_std_values, x1_pred_std_values]
            if overlay is not None:
                limit_arrays.extend([x1_std_v2, x1_pred_std_v2])
            _, max_val = _finite_axis_limits(limit_arrays)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, alpha=0.8)

        r_squared = _r_squared(x1_std_values, x1_pred_std_values)
        r_squared_v2 = None
        if overlay is not None:
            r_squared_v2 = _r_squared(x1_std_v2, x1_pred_std_v2)

        if i % n_cols == 0:
            ax.set_ylabel(r'log($\rho_{\mathrm{DM}}$) Predicted', fontsize=AXIS_LABEL_FONTSIZE)
        if i >= n_cols * (n_rows - 1):
            ax.set_xlabel(r'log($\rho_{\mathrm{DM}}$) True', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        panel_text = f'$r=${radial_centers[radial_idx]:.1f} kpc'
        if SHOW_R2:
            if r_squared_v2 is not None:
                label_v1, label_v2 = _dual_panel_labels()
                panel_text += _r2_line(label_v1, r_squared)
                panel_text += _r2_line(label_v2, r_squared_v2)
            else:
                panel_text += f'\n$R^2=${r_squared:.2f}'
        ax.text(
            0.05,
            0.95,
            panel_text,
            transform=ax.transAxes,
            fontsize=IN_PLOT_TEXTBOX_FONTSIZE,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
            zorder=5,
        )
        paper_grid(ax, alpha=0.3)
        _style_standard_axis(ax)
        ax.set_aspect('equal', adjustable='box')

    if scatter is not None:
        cbar = fig.colorbar(
            scatter, ax=axes, orientation='vertical', fraction=0.025, pad=0.01
        )
        cbar.set_label('log($\\rho_{\\mathrm{DM}}$)', fontsize=AXIS_LABEL_FONTSIZE)
        _style_colorbar_ticks(cbar)

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    return fig, axes


def plot_deviation_distribution(
    radial_indices,
    structured_data,
    n_cols=3,
    xpred_to_plot=None,
    xpred_cc_to_plot=None,
    downsample_factor=1,
    do_error_bars=False,
    uerr=None,
    lerr=None,
    overlay=None,
    highlight_galaxy_ids=None,
):
    n_plots = len(radial_indices)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    xpred_to_plot, xpred_cc_to_plot = resolve_xpred_arrays(
        structured_data, xpred_to_plot, xpred_cc_to_plot
    )
    if overlay is not None:
        overlay['xpred_to_plot'], overlay['xpred_cc_to_plot'] = resolve_xpred_arrays(
            overlay['structured_data'],
            overlay.get('xpred_to_plot'),
            overlay.get('xpred_cc_to_plot'),
        )
    if highlight_galaxy_ids is not None and overlay is not None:
        print("Warning: highlight_galaxy_ids ignored because overlay is set")
        highlight_galaxy_ids = None

    hist_bins_v1 = np.linspace(DEVIATION_HIST_XLIM[0], DEVIATION_HIST_XLIM[1], DEVIATION_HIST_BINS_V1 + 1)
    radial_centers = structured_data['radial_centers'][0]

    for i, radial_idx in enumerate(radial_indices):
        ax = axes[i]

        (
            x1_values,
            x1_pred_values,
            _x1_std_values,
            x1_pred_std_values,
            x1_pred_uerr_values,
            x1_pred_lerr_values,
        ) = get_radial_panel_arrays(
            structured_data,
            radial_idx,
            xpred_to_plot,
            xpred_cc_to_plot,
            downsample_factor=downsample_factor,
            uerr=uerr,
            lerr=lerr,
        )

        sigmas = compute_deviation_sigmas(
            x1_values,
            x1_pred_values,
            x1_pred_std_values,
            x1_pred_uerr_values,
            x1_pred_lerr_values,
        )

        ax.hist(sigmas, bins=hist_bins_v1, density=True, alpha=0.5, color=V1_HIST_COLOR, zorder=1)
        max_density_v1 = histogram_max_density(sigmas, DEVIATION_HIST_BINS_V1)

        sigmas_v2 = None
        if overlay is not None:
            (
                x1_v2,
                x1_pred_v2,
                _x1_std_v2,
                x1_pred_std_v2,
                uerr_v2,
                lerr_v2,
            ) = get_radial_panel_arrays(
                overlay['structured_data'],
                radial_idx,
                overlay['xpred_to_plot'],
                overlay['xpred_cc_to_plot'],
                downsample_factor=downsample_factor,
                # compute_deviation_sigmas ignores xpred_cc_to_plot whenever uerr/lerr
                # are supplied, so passing the overlay's per-pixel percentile errors
                # here would silently divide v2 by a different quantity than v1 (which
                # is never given uerr/lerr in this figure). That put TNG50 on the
                # profile denominator and the FIRE overlay on the per-pixel one in the
                # same panel. Only the legacy 'perpixel' mode keeps the old behavior.
                uerr=overlay.get('uerr') if DEVIATION_DENOMINATOR == 'perpixel' else None,
                lerr=overlay.get('lerr') if DEVIATION_DENOMINATOR == 'perpixel' else None,
            )
            sigmas_v2 = compute_deviation_sigmas(
                x1_v2,
                x1_pred_v2,
                x1_pred_std_v2,
                uerr_v2,
                lerr_v2,
            )
            plot_v2_deviation_histogram(ax, sigmas_v2, max_density_v1)

        sigmas_highlight = None
        if highlight_galaxy_ids is not None:
            gal_ids_panel = structured_data['galaxy_ids'][slice(None, None, downsample_factor)]
            highlight_mask = np.isin(gal_ids_panel, highlight_galaxy_ids)
            if highlight_mask.sum() > 1:
                sigmas_highlight = sigmas[highlight_mask]
                plot_v2_deviation_histogram(ax, sigmas_highlight, max_density_v1)

        # Mean, matching plot_calibration_figures.py, so the quoted Delta for a given
        # sample is the same number in every figure it appears in.
        mean_sigma = np.mean(sigmas)
        std_sigma = np.std(sigmas)
        ax.axvline(mean_sigma, color=V1_HIST_COLOR, linestyle='--', linewidth=2, zorder=3)

        sigmas_overlay = sigmas_highlight if sigmas_highlight is not None else sigmas_v2
        if sigmas_overlay is not None:
            ax.axvline(
                np.mean(sigmas_overlay),
                color=V2_HIST_EDGE_COLOR,
                linestyle='--',
                linewidth=2,
                zorder=3,
            )

        panel_text = f'$r=${radial_centers[radial_idx]:.1f} kpc'
        if sigmas_highlight is not None:
            panel_text += _delta_line(
                'DREAMS,test', np.mean(sigmas_highlight), np.std(sigmas_highlight)
            )
            panel_text += _delta_line('DREAMS,all', mean_sigma, std_sigma)
        elif sigmas_v2 is not None:
            label_v1, label_v2 = _dual_panel_labels()
            panel_text += _delta_line(label_v1, mean_sigma, std_sigma)
            panel_text += _delta_line(
                label_v2, np.mean(sigmas_v2), np.std(sigmas_v2)
            )
        else:
            panel_text += f'\n$\\Delta$={mean_sigma:.2f}$\\pm${std_sigma:.2f}'
        ax.text(
            0.05,
            0.95,
            panel_text,
            transform=ax.transAxes,
            fontsize=DEVIATION_HIST_TEXTBOX_FONTSIZE,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
            zorder=5,
        )

        x = np.linspace(-3, 3, 100)
        y = np.exp(-x**2 / (2 * 1**2)) / (1 * np.sqrt(2 * np.pi))
        ax.plot(x, y, 'k--', linewidth=2)

        if i % n_cols == 0:
            ax.set_ylabel(r'Probability', fontsize=AXIS_LABEL_FONTSIZE)
        if i >= n_cols * (n_rows - 1):
            ax.set_xlabel(r'Residual [$\Delta$]', fontsize=AXIS_LABEL_FONTSIZE)
        paper_grid(ax, alpha=0.3)
        _style_standard_axis(ax)
        ax.set_xlim(DEVIATION_HIST_XLIM[0], DEVIATION_HIST_XLIM[1])

    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    return fig, axes


def build_overlay_context(structured_data, draw_profiles=None, all_draws=None):
    ctx = {'structured_data': structured_data, 'all_draws': all_draws}
    if draw_profiles is not None:
        ctx['xpred_to_plot'] = draw_profiles['combined_profile']
        ctx['xpred_cc_to_plot'] = deviation_denominator(draw_profiles)
        ctx['uerr'] = draw_profiles['uerr_perpixel_agg']
        ctx['lerr'] = draw_profiles['lerr_perpixel_agg']
    return ctx


def load_second_version_bundle(
    data_dir,
    galaxy_idx_list,
    compute_draws,
    version_tag=None,
    pred_root=None,
    num_draws=None,
    draws_dir=None,
    file_pattern=None,
    draw_slice=None,
    radial_centers_fallback=None,
    cache=None,
):
    tag = version_tag or VERSION_TAG_2
    nd = num_draws if num_draws is not None else (NUM_DRAWS_2 or NUM_DRAWS)
    draw_slice_resolved = parse_draws_slice(
        draw_slice if draw_slice is not None else DRAWS_SLICE_2
    )
    pattern = file_pattern if file_pattern is not None else DRAWS_FILE_PATTERN_2

    # Ahead of the discovery below, which globs for draw files: a reader working from the
    # cache has none, so discovery would raise before the cache was ever consulted.
    if cache is not None and cache.readable:
        print(f"Reading overlay reduced data: {cache.describe()}")
        structured_v2, draw_profiles_v2 = cache.load_profiles()
        structured_v2 = {k: np.asarray(v) for k, v in structured_v2.items()}
        cached_ids = cache.cached_galaxy_ids()
        if cached_ids:
            draws_v2 = {'galaxy_ids': np.array(cached_ids),
                        'x_pred_draws': np.stack([cache.load_galaxy_draws(g) for g in cached_ids])}
        else:
            draws_v2 = {'galaxy_ids': np.array([], dtype=int),
                        'x_pred_draws': np.empty((0, 0, 0, 0), dtype=np.float32)}
        return build_overlay_context(structured_v2, draw_profiles_v2, all_draws=draws_v2)

    if DATA_LAYOUT == "per_galaxy":
        root = Path(pred_root or PRED_ROOT)
        v2_pattern = effective_draws_file_pattern(tag, nd, pattern)
        v2_galaxy_indices = discover_galaxy_indices_for_version(
            tag, num_draws=nd, file_pattern=v2_pattern
        )
        if galaxy_idx_list is not None:
            allowed = set(int(g) for g in galaxy_idx_list)
            v2_galaxy_indices = [g for g in v2_galaxy_indices if g in allowed]
        if not v2_galaxy_indices:
            raise FileNotFoundError(
                f"No overlay galaxies found for version={tag!r} "
                f"with pattern={v2_pattern!r}"
            )
        print(
            f"Overlay version {tag}: {len(v2_galaxy_indices)} galaxies with draw files"
        )

        n_samples = count_radial_profile_samples(pred_root=root, version_tag=tag)
        if n_samples == 0 and radial_centers_fallback is not None:
            n_samples = len(v2_galaxy_indices)
        warn_if_second_version_large(n_samples)
        print(
            f"Second version ({tag}): {n_samples} galaxies for overlay "
            f"({len(v2_galaxy_indices)} with draw files)"
        )

        structured_v2 = load_structured_data_for_version(
            tag,
            v2_galaxy_indices,
            radial_centers_fallback=radial_centers_fallback,
            file_pattern=v2_pattern,
        )
        draw_profiles_v2 = None
        draws_v2 = None
        if compute_draws:
            draws_dir_v2 = resolve_draws_data_dir(
                data_dir,
                version_tag=tag,
                draws_data_dir=draws_dir if draws_dir is not None else DRAWS_DATA_DIR_2,
                use_script_draws_dir=False,
            )
            draws_v2 = load_pred_data_structured(
                str(draws_dir_v2),
                idx_list=v2_galaxy_indices,
                file_pattern=v2_pattern,
                draw_slice=draw_slice_resolved,
                version_tag=tag,
                num_draws=nd,
            )
            draws_v2 = align_draws_to_structured(draws_v2, structured_v2)
            draw_profiles_v2 = compute_and_derive_draw_profiles(draws_v2['x_pred_draws'])
        if cache is not None and cache.writable:
            cache.save_profiles(structured_v2, draw_profiles_v2)
            if draws_v2 is not None and len(draws_v2['galaxy_ids']):
                for galaxy_idx in PER_GALAXY_INDICES:
                    row = _draws_row_for_galaxy(draws_v2, galaxy_idx)
                    if row is not None:
                        cache.save_galaxy_draws(galaxy_idx, draws_v2['x_pred_draws'][row])
            print(f"Wrote overlay reduced data: {cache.describe()}")
        return build_overlay_context(structured_v2, draw_profiles_v2, all_draws=draws_v2)

    n_samples = count_radial_profile_samples(data_dir)
    warn_if_second_version_large(n_samples)
    print(f"Second version: {n_samples} radial-profile files in {data_dir}")

    structured_v2 = load_all_data_structured(data_dir, idx_list=galaxy_idx_list, version_tag=tag)
    draw_profiles_v2 = None
    draws_v2 = None
    if compute_draws:
        draws_dir_v2 = resolve_draws_data_dir(data_dir)
        draws_v2 = load_pred_data_structured(
            draws_dir_v2,
            idx_list=galaxy_idx_list,
            file_pattern=pattern or DRAWS_FILE_PATTERN,
            version_tag=tag,
            num_draws=nd,
        )
        draw_profiles_v2 = compute_and_derive_draw_profiles(draws_v2['x_pred_draws'])
    return build_overlay_context(structured_v2, draw_profiles_v2, all_draws=draws_v2)


def compute_deviation_map(
    draws,
    true_map,
    condition_map,
    galaxy_idx,
    save_stem=None,
    save_dpi=200,
    draw_row=None,
    plot_hist=True,
):
    # draw_row: row into the draws array when it differs from the map index
    # (e.g. v2 overlay draws aligned to a filtered galaxy list).
    draws_temp = draws[galaxy_idx if draw_row is None else draw_row]
    has_draw_nans = _warn_if_draw_nans(draws_temp, f"galaxy {galaxy_idx} prediction draws")
    if has_draw_nans:
        x_pred_median = combine_draws(draws_temp, axis=0, nan_safe=True)
        x_pred_std = np.nanstd(draws_temp, axis=0)
    else:
        x_pred_median = combine_draws(draws_temp, axis=0)
        x_pred_std = np.std(draws_temp, axis=0)

    x1_values = true_map[galaxy_idx]
    condition_map_values = condition_map[galaxy_idx]

    deviation_map = (x_pred_median - x1_values) / x_pred_std

    if np.isnan(deviation_map).any():
        print("Warning: deviation_map has nan values")
    if np.isinf(deviation_map).any():
        print("Warning: deviation_map has inf values")

    fig_maps = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(condition_map_values, cmap='gray_r', vmin=3, vmax=11, aspect='equal', origin='lower')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(x1_values, cmap='viridis', vmin=5.5, vmax=9.5, aspect='equal', origin='lower')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(x_pred_median, cmap='viridis', vmin=5.5, vmax=9.5, aspect='equal', origin='lower')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow(deviation_map, cmap='bwr', vmin=-5, vmax=5, aspect='equal', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('Residual [$\\Delta$]', rotation=270, labelpad=15, fontsize=AXIS_LABEL_FONTSIZE)
    _style_colorbar_ticks(cbar)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_stem is not None:
        save_paper_figure(fig_maps, f"{save_stem}_maps.png", dpi=save_dpi)
    else:
        plt.show()

    if plot_hist:
        fig_hist, ax = _new_residual_panel()
        mean_sigma = np.mean(deviation_map)
        std_sigma = np.std(deviation_map)
        ax.hist(
            deviation_map.flatten(),
            bins=100,
            density=True,
            color='blue',
            label=f'$\\Delta$, example galaxy {galaxy_idx}',
        )
        ax.axvline(mean_sigma, color='r', linestyle='--', linewidth=2)
        ax.axvline(mean_sigma + std_sigma, color='g', linestyle='--', linewidth=2)
        ax.axvline(mean_sigma - std_sigma, color='g', linestyle='--', linewidth=2)
        # Drawn across the whole axis, not just [-3, 3] as before -- a reference curve
        # that stops short of the frame reads as a fit that ran out of data.
        x = np.linspace(PER_GALAXY_HIST_XLIM[0], PER_GALAXY_HIST_XLIM[1], 400)
        y = np.exp(-x**2 / 2.0) / np.sqrt(2 * np.pi)
        ax.plot(x, y, 'k--', linewidth=2, label=UNIT_NORMAL_LABEL)
        ax.text(
            0.05,
            0.95,
            f'$\\Delta = {mean_sigma:.2f}\\pm{std_sigma:.2f}$',
            transform=ax.transAxes,
            fontsize=RESIDUAL_PANEL_TEXTBOX_FONTSIZE,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )
        ax.legend(loc='upper right', fontsize=RESIDUAL_PANEL_LEGEND_FONTSIZE)
        _finish_residual_panel(ax, 'Residual [$\\Delta$]', 'Normalized counts')
        ax.set_xlim(*PER_GALAXY_HIST_XLIM)
        if save_stem is not None:
            save_paper_figure(
                fig_hist, f"{save_stem}_hist.png", dpi=save_dpi, tight=False
            )
        else:
            plt.show()

    return deviation_map


def plot_deviation_hist_pair(
    deviation_map_v1,
    deviation_map_v2,
    label_v1,
    label_v2,
    save_stem=None,
    save_dpi=200,
):
    """Single-galaxy residual histogram with both versions in one figure."""
    fig_hist = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for dev_map, color, label in (
        (deviation_map_v1, PER_GALAXY_V1_COLOR, label_v1),
        (deviation_map_v2, PER_GALAXY_V2_COLOR, label_v2),
    ):
        mean_sigma = np.mean(dev_map)
        std_sigma = np.std(dev_map)
        ax.hist(
            dev_map.flatten(),
            bins=100,
            density=True,
            color=color,
            alpha=0.55,
            label=f'{label}: {mean_sigma:.2f}$\\pm${std_sigma:.2f}',
        )
        ax.axvline(mean_sigma, color=color, linestyle='--', linewidth=2)
    x = np.linspace(-3, 3, 100)
    y = np.exp(-x**2 / (2 * 1**2)) / (1 * np.sqrt(2 * np.pi))
    ax.plot(x, y, 'k--', linewidth=2)
    ax.legend(loc='upper left', fontsize=IN_PLOT_TEXTBOX_FONTSIZE, framealpha=0.9)
    plt.xlabel('Residual [$\\Delta$]', fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel('Normalized Counts', fontsize=AXIS_LABEL_FONTSIZE)
    _style_standard_axis(ax)
    plt.xlim(-5, 5)
    if save_stem is not None:
        save_paper_figure(fig_hist, f"{save_stem}_hist.png", dpi=save_dpi)
    else:
        plt.show()


def _imshow_map_panel(ax, data, cmap, vmin, vmax, title=None, title_fontsize=DRAW_PANEL_TITLE_FONTSIZE):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
    no_grid(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.text(
            0.97,
            0.97,
            title,
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=title_fontsize,
            color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.55, pad=0.3),
        )
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return im


def _style_colorbar_axis(cax, label):
    cax.set_ylabel(label, fontsize=DRAW_PANEL_CBAR_LABEL_FONTSIZE, labelpad=8)
    cax.tick_params(labelsize=DRAW_PANEL_CBAR_TICK_FONTSIZE, direction='in')


def plot_galaxy_stochastic_draw_panels(
    draws,
    true_map,
    condition_map,
    galaxy_idx,
    save_stem=None,
    save_dpi=200,
    n_draws_to_show=PER_GALAXY_DRAW_PANELS_N_SHOW,
    draw_row=None,
):
    """
    Horizontal single-galaxy figure: condition + truth (left), example stochastic
    pred/residual pairs (center), combined pred/residual after combining draws (right).
    """
    draws_temp = draws[galaxy_idx if draw_row is None else draw_row]
    n_draws = draws_temp.shape[0]
    x1_values = true_map[galaxy_idx]
    condition_map_values = condition_map[galaxy_idx]

    pred_median = combine_draws(draws_temp, axis=0)
    residual_draws = draws_temp - x1_values[None, ...]
    residual_median = pred_median - x1_values

    n_show = min(n_draws_to_show, n_draws)
    draw_indices = np.linspace(0, n_draws - 1, n_show, dtype=int)

    res_lim = float(
        max(
            np.max(np.abs(residual_draws[draw_indices])),
            np.max(np.abs(residual_median)),
            0.1,
        )
    )

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(
        8,
        8,
        width_ratios=[1.15, 0.30, 0.72, 0.72, 0.72, 0.72, 0.34, 1.15],
        height_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
        wspace=0.14,
        hspace=0.18,
        left=0.08,
        right=0.88,
        top=0.93,
        bottom=0.06,
    )

    ax_cond = fig.add_subplot(gs[0:4, 0])
    ax_true = fig.add_subplot(gs[4:8, 0])
    im_cond = _imshow_map_panel(
        ax_cond,
        condition_map_values,
        'gray_r',
        COND_MAP_VMIN,
        COND_MAP_VMAX,
        title='Condition log $\\rho_{\\mathrm{star}}$',
    )
    ax_cond.text(
        0.03,
        0.03,
        DRAW_PANEL_SCALE_NOTE,
        transform=ax_cond.transAxes,
        ha='left',
        va='bottom',
        fontsize=DRAW_PANEL_SCALE_NOTE_FONTSIZE,
        color='black',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, pad=0.25),
    )
    im_rho = _imshow_map_panel(
        ax_true,
        x1_values,
        'viridis',
        RHO_MAP_VMIN,
        RHO_MAP_VMAX,
        title='True log $\\rho_{\\mathrm{DM}}$',
    )

    ax_flow_left = fig.add_subplot(gs[2:6, 1])
    ax_flow_left.axis('off')
    ax_flow_left.annotate(
        '',
        xy=(0.96, 0.5),
        xytext=(0.04, 0.5),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', lw=2.2, color='0.4'),
    )
    ax_flow_left.text(
        0.5,
        0.84,
        'stochastic\ndraws',
        ha='center',
        va='center',
        fontsize=DRAW_PANEL_FLOW_FONTSIZE,
        style='italic',
        transform=ax_flow_left.transAxes,
    )

    rho_im = im_rho
    res_im = None
    for i, draw_idx in enumerate(draw_indices):
        col = 2 + i
        ax_pred = fig.add_subplot(gs[0:4, col])
        ax_res = fig.add_subplot(gs[4:8, col])
        rho_im = _imshow_map_panel(
            ax_pred,
            draws_temp[draw_idx],
            'viridis',
            RHO_MAP_VMIN,
            RHO_MAP_VMAX,
            title=f'draw {draw_idx + 1}',
            title_fontsize=DRAW_PANEL_DRAW_TITLE_FONTSIZE,
        )
        res_im = _imshow_map_panel(
            ax_res,
            residual_draws[draw_idx],
            'bwr',
            -res_lim,
            res_lim,
            title=f'draw {draw_idx + 1} − true',
            title_fontsize=DRAW_PANEL_DRAW_TITLE_FONTSIZE,
        )

    ax_flow_right = fig.add_subplot(gs[2:6, 6])
    ax_flow_right.axis('off')
    ax_flow_right.annotate(
        '',
        xy=(0.96, 0.5),
        xytext=(0.04, 0.5),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='->', lw=2.2, color='0.4'),
    )
    ax_flow_right.text(
        0.5,
        0.84,
        'combine\ndraws',
        ha='center',
        va='center',
        style='italic',
        fontsize=DRAW_PANEL_FLOW_FONTSIZE,
        transform=ax_flow_right.transAxes,
    )

    ax_median = fig.add_subplot(gs[0:4, 7])
    ax_median_res = fig.add_subplot(gs[4:8, 7])
    rho_im = _imshow_map_panel(
        ax_median,
        pred_median,
        'viridis',
        RHO_MAP_VMIN,
        RHO_MAP_VMAX,
        title=f'{combine_label().capitalize()} prediction',
    )
    res_im = _imshow_map_panel(
        ax_median_res,
        residual_median,
        'bwr',
        -res_lim,
        res_lim,
        title=f'{combine_label()} − true',
    )

    cond_cbar_ax = fig.add_axes([0.014, 0.54, 0.012, 0.36])
    fig.colorbar(im_cond, cax=cond_cbar_ax)
    _style_colorbar_axis(cond_cbar_ax, 'log $\\rho_{\\mathrm{star}}$')

    true_cbar_ax = fig.add_axes([0.014, 0.12, 0.012, 0.36])
    fig.colorbar(im_rho, cax=true_cbar_ax)
    _style_colorbar_axis(true_cbar_ax, 'log $\\rho_{\\mathrm{DM}}$')

    cbar_rho_ax = fig.add_axes([0.905, 0.56, 0.012, 0.32])
    fig.colorbar(rho_im, cax=cbar_rho_ax)
    _style_colorbar_axis(cbar_rho_ax, 'log $\\rho_{\\mathrm{DM}}$')

    cbar_res_ax = fig.add_axes([0.905, 0.12, 0.012, 0.32])
    fig.colorbar(res_im, cax=cbar_res_ax)
    _style_colorbar_axis(cbar_res_ax, '$\\Delta$ log $\\rho_{\\mathrm{DM}}$')

    # fig.suptitle(
    #     f'Example stochastic posterior predictions (galaxy {galaxy_idx})',
    #     fontsize=DRAW_PANEL_FIG_TITLE_FONTSIZE,
    #     y=0.98,
    # )

    if save_stem is not None:
        out_path = f"{save_stem}_draw_panels.png"
        save_paper_figure(fig, out_path, dpi=save_dpi)
    else:
        plt.show()

    return {
        'pred_median': pred_median,
        'residual_draws': residual_draws,
        'residual_median': residual_median,
    }


def compute_deviation_summary(
    deviation_map,
    N_rings,
    N_azimuthal_bins,
    xmin=-32,
    xmax=32,
    ymin=-32,
    ymax=32,
    save_stem=None,
    save_dpi=200,
    if_plot_binned_map=True,
):
    mean_deviation = np.mean(deviation_map)
    std_deviation = np.std(deviation_map)

    if np.abs(deviation_map).max() > 100:
        print(f"Warning: deviation_map has very large values; max: {np.abs(deviation_map).max()}")

    resolution = deviation_map.shape[0]
    assert deviation_map.shape[1] == resolution

    num_1sigma = np.sum(np.abs(deviation_map) < 1)
    num_2sigma = np.sum(np.abs(deviation_map) < 2)
    num_3sigma = np.sum(np.abs(deviation_map) < 3)

    print(f"mean_deviation: {mean_deviation}, std_deviation: {std_deviation}")
    print(
        f"f(1sigma): {num_1sigma / resolution**2}, "
        f"f(2sigma): {num_2sigma / resolution**2}, "
        f"f(3sigma): {num_3sigma / resolution**2}"
    )

    r_min = 0.0
    r_max = float(np.sqrt(xmax**2 + ymax**2))
    theta_min = -np.pi
    theta_max = np.pi

    num_rbins = min(N_rings, int(resolution // 4))
    if num_rbins != N_rings:
        print(f"Warning: num_rbins is not set to N_rings; num_rbins: {num_rbins}, N_rings: {N_rings}")
    num_thetabins = min(N_azimuthal_bins, 8)
    if num_thetabins != N_azimuthal_bins:
        print(
            f"Warning: num_thetabins is not set to N_azimuthal_bins; "
            f"num_thetabins: {num_thetabins}, N_azimuthal_bins: {N_azimuthal_bins}"
        )
    radial_edges = np.linspace(r_min, r_max, num_rbins + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    theta_edges = np.linspace(theta_min, theta_max, num_thetabins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    print(f"resolution: {resolution}, num_rbins: {num_rbins}, num_thetabins: {num_thetabins}")

    if save_stem is not None:
        save_stem = f"{save_stem}_N{N_rings}x{N_azimuthal_bins}"
        print(f"save_stem: {save_stem}")

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")
    r_coords = np.sqrt(x_coords**2 + y_coords**2)
    theta_coords = np.arctan2(y_coords, x_coords)

    mask = np.full((num_rbins, num_thetabins, resolution, resolution), False)
    mask_sum = np.zeros((num_rbins, num_thetabins))
    for i in range(num_rbins):
        for j in range(num_thetabins):
            r1 = radial_edges[i]
            r2 = radial_edges[i + 1]
            theta1 = theta_edges[j]
            theta2 = theta_edges[j + 1]
            if i < num_rbins - 1 and j < num_thetabins - 1:
                in_bin = (
                    (r_coords >= r1) & (r_coords < r2)
                    & (theta1 <= theta_coords) & (theta_coords < theta2)
                )
            elif i < num_rbins - 1 and j == num_thetabins - 1:
                in_bin = (
                    (r_coords >= r1) & (r_coords <= r2)
                    & (theta1 <= theta_coords) & (theta_coords <= theta2)
                )
            elif i == num_rbins - 1 and j < num_thetabins - 1:
                in_bin = (
                    (r_coords >= r1) & (r_coords <= r2)
                    & (theta1 <= theta_coords) & (theta_coords < theta2)
                )
            else:
                in_bin = (
                    (r_coords >= r1) & (r_coords <= r2)
                    & (theta1 <= theta_coords) & (theta_coords <= theta2)
                )
            if in_bin.any():
                mask[i, j] = in_bin
                mask_sum[i, j] = np.sum(in_bin)

    mask_radial = np.full((num_rbins, resolution, resolution), False)
    for i in range(num_rbins):
        r1 = radial_edges[i]
        r2 = radial_edges[i + 1]
        if i < num_rbins - 1:
            in_bin = (r_coords >= r1) & (r_coords < r2)
        else:
            in_bin = (r_coords >= r1) & (r_coords <= r2)
        if in_bin.any():
            mask_radial[i] = in_bin

    mask_theta = np.full((num_thetabins, resolution, resolution), False)
    for i in range(num_thetabins):
        theta1 = theta_edges[i]
        theta2 = theta_edges[i + 1]
        if i < num_thetabins - 1:
            in_bin = (theta1 <= theta_coords) & (theta_coords < theta2)
        elif i == num_thetabins - 1:
            in_bin = (theta1 <= theta_coords) & (theta_coords <= theta2)
        else:
            raise ValueError(f"i: {i}, num_thetabins: {num_thetabins}")
        if in_bin.any():
            mask_theta[i] = in_bin

    assert np.sum(mask_sum) == resolution**2
    if (mask_sum < 10).any():
        print("Warning: some bins have less than 10 pixels")

    deviation_map_in_bins = np.zeros((num_rbins, num_thetabins))
    deviation_map_in_bins_std = np.zeros((num_rbins, num_thetabins))
    for i in range(num_rbins):
        for j in range(num_thetabins):
            deviation_map_in_bins[i, j] = np.mean(deviation_map[mask[i, j]])
            deviation_map_in_bins_std[i, j] = np.std(deviation_map[mask[i, j]])

    deviation_map_in_bins_radial = np.zeros((num_rbins))
    deviation_map_in_bins_std_radial = np.zeros((num_rbins))
    for i in range(num_rbins):
        deviation_map_in_bins_radial[i] = np.mean(deviation_map[mask_radial[i]])
        deviation_map_in_bins_std_radial[i] = np.std(deviation_map[mask_radial[i]])

    deviation_map_in_bins_theta = np.zeros((num_thetabins))
    deviation_map_in_bins_std_theta = np.zeros((num_thetabins))
    for i in range(num_thetabins):
        deviation_map_in_bins_theta[i] = np.mean(deviation_map[mask_theta[i]])
        deviation_map_in_bins_std_theta[i] = np.std(deviation_map[mask_theta[i]])

    if if_plot_binned_map:
        fig_binned_map = plt.figure(figsize=(10, 5))
        plt.imshow(deviation_map, cmap='bwr', vmin=-5, vmax=5, aspect='equal', origin='lower')
        cbar = plt.colorbar()
        cbar.set_label('Residual [$\\Delta$]', rotation=270, labelpad=15, fontsize=AXIS_LABEL_FONTSIZE)
        _style_colorbar_ticks(cbar)
        plt.xticks([])
        plt.yticks([])
        kpc_to_resolution = resolution / (xmax - xmin)
        for i in range(num_rbins):
            circle = plt.Circle(
                (resolution / 2, resolution / 2),
                radial_edges[i] * kpc_to_resolution,
                fill=False,
                color='black',
                linewidth=1,
            )
            plt.gca().add_artist(circle)
        for i in range(num_thetabins):
            wedge = patches.Wedge(
                (resolution / 2, resolution / 2),
                radial_edges[-1] * kpc_to_resolution,
                theta_edges[i] * 180 / np.pi + 90,
                theta_edges[i + 1] * 180 / np.pi + 90,
                fill=False,
                color='black',
                linewidth=1,
            )
            plt.gca().add_artist(wedge)
        for i in range(num_rbins):
            for j in range(num_thetabins):
                text_loc_y = np.mean(x_coords[mask[i, j]]) * resolution / (xmax - xmin) + resolution / 2
                text_loc_x = np.mean(y_coords[mask[i, j]]) * resolution / (ymax - ymin) + resolution / 2
                plt.text(
                    text_loc_x,
                    text_loc_y,
                    f'{deviation_map_in_bins[i, j]:.2f}\n$\\pm${deviation_map_in_bins_std[i, j]:.2f}',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=IN_PLOT_TEXTBOX_FONTSIZE - 2,
                )
        if save_stem is not None:
            save_paper_figure(fig_binned_map, f"{save_stem}_binned_map.png", dpi=save_dpi)
        else:
            plt.show()

    fig_radial, ax_radial = _new_residual_panel()
    ax_radial.axhline(0, color='0.7', linestyle='--', linewidth=1.0)
    ax_radial.errorbar(
        radial_centers,
        deviation_map_in_bins_radial,
        yerr=deviation_map_in_bins_std_radial,
        fmt='o',
        color='black',
        elinewidth=2,
        capsize=5,
        markersize=5,
    )
    _finish_residual_panel(ax_radial, 'Radius [kpc]', 'Residual [$\\Delta$]')
    symmetric_ylim(ax_radial)
    # Annuli beyond the inscribed radius of the square field of view are corner-only.
    mark_fov_limit(ax_radial)
    if save_stem is not None:
        save_paper_figure(
            fig_radial, f"{save_stem}_radial.png", dpi=save_dpi, tight=False
        )
    else:
        plt.show()

    fig_azimuthal, ax_azimuthal = _new_residual_panel()
    ax_azimuthal.axhline(0, color='0.7', linestyle='--', linewidth=1.0)
    ax_azimuthal.errorbar(
        to_degrees(theta_centers),
        deviation_map_in_bins_theta,
        yerr=deviation_map_in_bins_std_theta,
        fmt='o',
        color='black',
        elinewidth=2,
        capsize=5,
        markersize=5,
    )
    _finish_residual_panel(
        ax_azimuthal, 'Azimuthal angle [deg]', 'Residual [$\\Delta$]'
    )
    style_azimuthal_axis(ax_azimuthal)
    symmetric_ylim(ax_azimuthal)
    if save_stem is not None:
        save_paper_figure(
            fig_azimuthal, f"{save_stem}_azimuthal.png", dpi=save_dpi, tight=False
        )
    else:
        plt.show()

    return (
        deviation_map_in_bins,
        deviation_map_in_bins_std,
        deviation_map_in_bins_radial,
        deviation_map_in_bins_std_radial,
        deviation_map_in_bins_theta,
        deviation_map_in_bins_std_theta,
    )


def load_sim_maps(sim_name):
    if sim_name not in MAP_PATHS:
        raise ValueError(f"Unknown SIM_FOR_MAPS={sim_name!r}; choose from {list(MAP_PATHS)}")
    paths = MAP_PATHS[sim_name]
    print(f"Loading maps for {sim_name}...")
    x_map_true = np.load(paths["rhoDM"])
    cond_map_true = np.load(paths["rhostar"])
    return x_map_true, cond_map_true


def _draws_row_for_galaxy(draws_bundle, galaxy_id):
    """Row of a loaded draws array holding this galaxy, or None if it was not loaded."""
    rows = np.where(np.asarray(draws_bundle['galaxy_ids']) == int(galaxy_id))[0]
    if len(rows) == 0:
        return None
    return int(rows[0])


def _v2_draw_row(overlay_draws, galaxy_id):
    return _draws_row_for_galaxy(overlay_draws, galaxy_id)


def plot_per_galaxy_draw_panels(all_draws, idx_list, overlay_draws=None):
    x_map_true, cond_map_true = load_sim_maps(SIM_FOR_MAPS)

    for galaxy_idx in PER_GALAXY_INDICES:
        draw_row = _draws_row_for_galaxy(all_draws, galaxy_idx)
        if draw_row is None:
            print(f"Skipping galaxy {galaxy_idx} (not in loaded subset)")
            continue

        save_stem = str(
            output_path_single(f"galaxy_{galaxy_idx}_stochastic", 'v1').with_suffix('')
        )
        print(f"Per-galaxy draw panels for galaxy {galaxy_idx} (draws row {draw_row}) "
              f"-> {save_stem}_draw_panels")

        plot_galaxy_stochastic_draw_panels(
            all_draws['x_pred_draws'],
            x_map_true,
            cond_map_true,
            galaxy_idx,
            draw_row=draw_row,
            save_stem=save_stem,
            save_dpi=PER_GALAXY_DRAW_PANELS_DPI,
        )

        if overlay_draws is not None:
            galaxy_id = galaxy_idx
            v2_row = _v2_draw_row(overlay_draws, galaxy_id)
            if v2_row is None:
                print(f"Skipping v2 draw panels for galaxy {galaxy_id} (no overlay draws)")
                continue
            save_stem_v2 = str(
                output_path_single(f"galaxy_{galaxy_idx}_stochastic", 'v2').with_suffix('')
            )
            print(
                f"Per-galaxy v2 draw panels for index {galaxy_idx} "
                f"-> {save_stem_v2}_draw_panels.png"
            )
            plot_galaxy_stochastic_draw_panels(
                overlay_draws['x_pred_draws'],
                x_map_true,
                cond_map_true,
                galaxy_idx,
                save_stem=save_stem_v2,
                save_dpi=PER_GALAXY_DRAW_PANELS_DPI,
                draw_row=v2_row,
            )


def plot_per_galaxy_prior_posterior(all_draws, idx_list, overlay_draws=None):
    x_map_true, _ = load_sim_maps(SIM_FOR_MAPS)

    resolution = all_draws['x_pred_draws'].shape[-1]
    _, radial_centers, mask = build_radial_annulus_geometry(resolution)

    n_galaxies_maps = x_map_true.shape[0]
    print(f"Computing true radial profiles from {SIM_FOR_MAPS} maps for {n_galaxies_maps} galaxies...")
    all_true_profiles = np.array([
        compute_radial_profile_from_map(x_map_true[i], mask)
        for i in range(n_galaxies_maps)
    ])

    for target_galaxy_id in PER_GALAXY_INDICES:
        draw_idx = _draws_row_for_galaxy(all_draws, target_galaxy_id)
        if draw_idx is None:
            print(f"Skipping galaxy {target_galaxy_id} (not in loaded subset)")
            continue

        if target_galaxy_id < 0 or target_galaxy_id >= n_galaxies_maps:
            raise ValueError(
                f"galaxy_id={target_galaxy_id} out of range for {SIM_FOR_MAPS} maps "
                f"with {n_galaxies_maps} galaxies"
            )

        true_profile = all_true_profiles[target_galaxy_id]
        other_mask = np.ones(n_galaxies_maps, dtype=bool)
        other_mask[target_galaxy_id] = False
        other_true_profiles = all_true_profiles[other_mask]

        draws_galaxy = all_draws['x_pred_draws'][draw_idx]
        posterior_profiles = compute_radial_profiles_from_map_stack(draws_galaxy, mask)

        posterior_profiles_2 = None
        posterior_label = 'Prediction draws'
        posterior_label_2 = None
        if overlay_draws is not None:
            v2_row = _v2_draw_row(overlay_draws, target_galaxy_id)
            if v2_row is None:
                print(
                    f"No v2 draws for galaxy {target_galaxy_id}; "
                    f"prior/posterior figure will show v1 only"
                )
            else:
                draws_galaxy_2 = overlay_draws['x_pred_draws'][v2_row]
                posterior_profiles_2 = compute_radial_profiles_from_map_stack(
                    draws_galaxy_2, mask
                )
                posterior_label = short_version_label(VERSION_TAG, DRAWS_OUTPUT_LABEL)
                posterior_label_2 = short_version_label(VERSION_TAG_2, DRAWS_OUTPUT_LABEL_2)

        save_stem = str(
            output_path(f"galaxy_{target_galaxy_id}_prior_posterior").with_suffix('')
        )
        print(
            f"Prior/posterior radial profile for galaxy {target_galaxy_id} "
            f"-> {save_stem}_prior_posterior_radial.png"
        )

        plot_galaxy_prior_posterior_radial_profile(
            radial_centers,
            true_profile,
            other_true_profiles,
            posterior_profiles,
            target_galaxy_id,
            save_stem=save_stem,
            save_dpi=PER_GALAXY_PRIOR_POSTERIOR_DPI,
            posterior_profiles_2=posterior_profiles_2,
            posterior_label=posterior_label,
            posterior_label_2=posterior_label_2,
        )


def plot_per_galaxy_deviation(all_draws, idx_list, overlay_draws=None):
    x_map_true, cond_map_true = load_sim_maps(SIM_FOR_MAPS)

    for galaxy_idx in PER_GALAXY_INDICES:
        draw_row = _draws_row_for_galaxy(all_draws, galaxy_idx)
        if draw_row is None:
            print(f"Skipping galaxy {galaxy_idx} (not in loaded subset)")
            continue

        save_stem = str(
            output_path_single(f"galaxy_{galaxy_idx}_deviation", 'v1').with_suffix('')
        )
        print(f"Per-galaxy deviation for galaxy {galaxy_idx} (draws row {draw_row}) "
              f"-> {save_stem}_*")

        # With an overlay the residual histogram is produced once, combined.
        deviation_map = compute_deviation_map(
            all_draws['x_pred_draws'],
            x_map_true,
            cond_map_true,
            galaxy_idx,
            draw_row=draw_row,
            save_stem=save_stem,
            save_dpi=PER_GALAXY_SAVE_DPI,
            plot_hist=(overlay_draws is None),
        )

        if PER_GALAXY_SAVE_SUMMARY:
            compute_deviation_summary(
                deviation_map,
                PER_GALAXY_SUMMARY_N_RINGS,
                PER_GALAXY_SUMMARY_N_AZIMUTHAL,
                save_stem=save_stem,
                save_dpi=PER_GALAXY_SAVE_DPI,
            )

        if overlay_draws is not None:
            galaxy_id = galaxy_idx
            v2_row = _v2_draw_row(overlay_draws, galaxy_id)
            if v2_row is None:
                print(f"Skipping v2 deviation for galaxy {galaxy_id} (no overlay draws)")
                continue
            save_stem_v2 = str(
                output_path_single(f"galaxy_{galaxy_idx}_deviation", 'v2').with_suffix('')
            )
            print(f"Per-galaxy v2 deviation for index {galaxy_idx} -> {save_stem_v2}_*.png")
            deviation_map_v2 = compute_deviation_map(
                overlay_draws['x_pred_draws'],
                x_map_true,
                cond_map_true,
                galaxy_idx,
                save_stem=save_stem_v2,
                save_dpi=PER_GALAXY_SAVE_DPI,
                draw_row=v2_row,
                plot_hist=False,
            )
            if PER_GALAXY_SAVE_SUMMARY:
                compute_deviation_summary(
                    deviation_map_v2,
                    PER_GALAXY_SUMMARY_N_RINGS,
                    PER_GALAXY_SUMMARY_N_AZIMUTHAL,
                    save_stem=save_stem_v2,
                    save_dpi=PER_GALAXY_SAVE_DPI,
                )

            save_stem_pair = str(
                output_path(f"galaxy_{galaxy_idx}_deviation").with_suffix('')
            )
            plot_deviation_hist_pair(
                deviation_map,
                deviation_map_v2,
                short_version_label(VERSION_TAG, DRAWS_OUTPUT_LABEL),
                short_version_label(VERSION_TAG_2, DRAWS_OUTPUT_LABEL_2),
                save_stem=save_stem_pair,
                save_dpi=PER_GALAXY_SAVE_DPI,
            )
            print(f"Combined residual histogram -> {save_stem_pair}_hist.png")


def main(
    data_dir=None,
    output_dir=None,
    galaxy_idx_list=None,
    version_tag=None,
    version_tag_2=None,
    data_dir_2=None,
    radial_profile_dir=None,
    draws_dir=None,
    draws_file_pattern=None,
    draws_slice=None,
    draws_output_label=None,
    draws_dir_2=None,
    draws_file_pattern_2=None,
    draws_slice_2=None,
    draws_output_label_2=None,
    num_draws_2=None,
    pred_root=None,
    data_layout=None,
    num_draws=None,
    show_r2=None,
    highlight_test_galaxies=False,
    test_galaxy_range=None,
    deviation_denominator_mode=None,
    deviation_hist_xlim=None,
    ensemble_combine=None,
    profile_agg=None,
    dreams_all_style=None,
    per_galaxy_indices=None,
    cache_dir=None,
    cache_mode="auto",
):
    global VERSION_TAG, VERSION_TAG_2, DATA_DIR, OUTPUT_DIR, GALAXY_IDX_LIST
    global RADIAL_PROFILE_DATA_DIR, DRAWS_DATA_DIR, DRAWS_FILE_PATTERN, DRAWS_SLICE, DRAWS_OUTPUT_LABEL
    global DRAWS_DATA_DIR_2, DRAWS_FILE_PATTERN_2, DRAWS_SLICE_2, DRAWS_OUTPUT_LABEL_2, NUM_DRAWS_2
    global PRED_ROOT, DATA_LAYOUT, NUM_DRAWS, SIM_FOR_MAPS, SHOW_R2, HIGHLIGHT_GALAXY_IDS
    global DEVIATION_DENOMINATOR, DEVIATION_HIST_XLIM
    global DREAMS_ALL_STYLE, DREAMS_ALL_STYLES, PER_GALAXY_INDICES

    apply_paper_style()

    combine, agg = set_combiners(ensemble=ensemble_combine, profile=profile_agg)
    if per_galaxy_indices is not None:
        PER_GALAXY_INDICES = [int(p) for p in str(per_galaxy_indices).split(',') if p.strip()]
        print(f"per-galaxy example figures for galaxies: {PER_GALAXY_INDICES}")
    if dreams_all_style is not None:
        DREAMS_ALL_STYLES = [
            part.strip() for part in str(dreams_all_style).split(',') if part.strip()
        ]
        for style in DREAMS_ALL_STYLES:
            if style not in ('points', 'contours', 'hide'):
                raise ValueError(f"Unknown DREAMS-all style: {style}")
        DREAMS_ALL_STYLE = DREAMS_ALL_STYLES[0]
    print(
        f"ensemble combine: {combine} | annulus aggregation: {agg} "
        f"| DREAMS-all styles: {DREAMS_ALL_STYLES}"
    )

    if deviation_denominator_mode is not None:
        DEVIATION_DENOMINATOR = deviation_denominator_mode
    if deviation_hist_xlim is not None:
        DEVIATION_HIST_XLIM = tuple(deviation_hist_xlim)
    if show_r2 is not None:
        SHOW_R2 = show_r2
    if highlight_test_galaxies:
        if version_tag_2 is not None:
            raise ValueError(
                "--highlight-test-galaxies is mutually exclusive with --version2"
            )
        start_str, end_str = (test_galaxy_range or '0:102').split(':')
        HIGHLIGHT_GALAXY_IDS = np.arange(int(start_str), int(end_str))
    if pred_root is not None:
        PRED_ROOT = Path(pred_root)
    if data_layout is not None:
        DATA_LAYOUT = data_layout
    if num_draws is not None:
        NUM_DRAWS = num_draws

    if version_tag is not None:
        VERSION_TAG = version_tag
    sim_label_v1, _ = parse_version_tag(VERSION_TAG)
    DATA_DIR = effective_pred_root(PRED_ROOT, sim_label_v1)
    SIM_FOR_MAPS = resolve_sim_for_maps(VERSION_TAG)

    if version_tag_2 is not None:
        VERSION_TAG_2 = version_tag_2
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    if radial_profile_dir is not None:
        RADIAL_PROFILE_DATA_DIR = Path(radial_profile_dir)
    if draws_dir is not None:
        DRAWS_DATA_DIR = Path(draws_dir)
    if draws_file_pattern is not None:
        DRAWS_FILE_PATTERN = draws_file_pattern
    if draws_slice is not None:
        DRAWS_SLICE = parse_draws_slice(draws_slice)
    if draws_output_label is not None:
        DRAWS_OUTPUT_LABEL = draws_output_label
    if draws_dir_2 is not None:
        DRAWS_DATA_DIR_2 = Path(draws_dir_2)
    if draws_file_pattern_2 is not None:
        DRAWS_FILE_PATTERN_2 = draws_file_pattern_2
    if draws_slice_2 is not None:
        DRAWS_SLICE_2 = parse_draws_slice(draws_slice_2)
    if draws_output_label_2 is not None:
        DRAWS_OUTPUT_LABEL_2 = draws_output_label_2
    if num_draws_2 is not None:
        NUM_DRAWS_2 = num_draws_2
    if output_dir is not None:
        OUTPUT_DIR = Path(output_dir)
    if galaxy_idx_list is not None:
        GALAXY_IDX_LIST = galaxy_idx_list

    radial_dir = resolve_radial_profile_dir(version_tag=VERSION_TAG)
    draws_dir_resolved = resolve_draws_data_dir(version_tag=VERSION_TAG)
    draw_slice_resolved = parse_draws_slice(DRAWS_SLICE)
    draws_pattern_v1 = effective_draws_file_pattern(
        VERSION_TAG, NUM_DRAWS, draws_file_pattern
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Version: {VERSION_TAG}")
    if VERSION_TAG_2:
        print(f"Version 2 (overlay): {VERSION_TAG_2}")
    print(f"Radial profile dir: {radial_dir}")
    print(f"Draws dir: {draws_dir_resolved}")
    print(f"Draws pattern: {draws_pattern_v1}")
    print(f"Draws slice: [{format_draws_slice(draw_slice_resolved)}]")
    if DRAWS_OUTPUT_LABEL:
        print(f"Draws output label (PNG): {DRAWS_OUTPUT_LABEL}")
    if VERSION_TAG_2:
        print(f"Draws v2 pattern: {DRAWS_FILE_PATTERN_2 or DRAWS_FILE_PATTERN}")
        print(f"Draws v2 slice: [{format_draws_slice(parse_draws_slice(DRAWS_SLICE_2))}]")
        if DRAWS_OUTPUT_LABEL_2:
            print(f"Draws v2 output label (PNG): {DRAWS_OUTPUT_LABEL_2}")
        if NUM_DRAWS_2:
            print(f"Draws v2 num_draws: {NUM_DRAWS_2}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Pred root: {PRED_ROOT}")
    print(f"Data layout: {DATA_LAYOUT}")
    print(f"SIM_FOR_MAPS: {SIM_FOR_MAPS}")

    has_radial_npz_v1 = radial_profiles_exist(PRED_ROOT, VERSION_TAG, GALAXY_IDX_LIST)
    plot_pred_vs_true = PLOT_PRED_VS_TRUE
    if plot_pred_vs_true and not has_radial_npz_v1:
        print("Disabling PLOT_PRED_VS_TRUE (no radial profile npz for version 1)")
        plot_pred_vs_true = False

    # Every setting below changes the cached numbers, so every one of them is part of the
    # key. A cache entry built for one figure set can then never be served to another --
    # which matters most for the combiner settings, where the arrays have identical shapes
    # and a mismatch would show up only as a quietly different published number.
    cache = None
    if cache_mode != "off":
        cache = figure_cache.FigureCache(
            cache_dir or stars2dm_paths.CACHE_ROOT,
            cache_mode,
            version=VERSION_TAG,
            version2=VERSION_TAG_2,
            pattern=draws_pattern_v1,
            num_draws=NUM_DRAWS,
            galaxies=None if GALAXY_IDX_LIST is None else f"{GALAXY_IDX_LIST[0]}:{GALAXY_IDX_LIST[-1] + 1}:{len(GALAXY_IDX_LIST)}",
            draw_slice=draw_slice_resolved,
            ensemble=_pdl.ENSEMBLE_COMBINE,
            profile_agg=_pdl.PROFILE_AGG,
            denominator=DEVIATION_DENOMINATOR,
        )
        cache.require()

    draw_profiles = None
    all_draws = None

    if cache is not None and cache.readable:
        print(f"Reading reduced figure data: {cache.describe()}")
        all_structured_data, draw_profiles = cache.load_profiles()
        all_structured_data = {k: np.asarray(v) for k, v in all_structured_data.items()}
        if needs_draws_profiles():
            # Only the galaxies that get a pixel-level panel of their own are cached. Others
            # resolve to None in _draws_row_for_galaxy, which already prints a skip notice,
            # so a partial set degrades to fewer panels rather than to a crash.
            cached_ids = cache.cached_galaxy_ids()
            if cached_ids:
                all_draws = {
                    'galaxy_ids': np.array(cached_ids),
                    'x_pred_draws': np.stack([cache.load_galaxy_draws(g) for g in cached_ids]),
                }
                print(f"Per-galaxy draws available from cache for galaxies {cached_ids}")
            else:
                all_draws = {'galaxy_ids': np.array([], dtype=int),
                             'x_pred_draws': np.empty((0, 0, 0, 0), dtype=np.float32)}
    else:
        start_time = time.time()
        all_structured_data = load_structured_data_for_version(
            VERSION_TAG,
            GALAXY_IDX_LIST,
        )
        print(
            f"Loaded {len(all_structured_data['galaxy_ids'])} galaxies "
            f"in {time.time() - start_time:.2f} s"
        )

        if needs_draws_profiles():
            start_time = time.time()
            galaxy_ids_v1 = all_structured_data['galaxy_ids'].tolist()
            all_draws = load_pred_data_structured(
                draws_dir_resolved,
                idx_list=galaxy_ids_v1,
                file_pattern=draws_pattern_v1,
                draw_slice=draw_slice_resolved,
                version_tag=VERSION_TAG,
                num_draws=NUM_DRAWS,
            )
            all_draws = align_draws_to_structured(all_draws, all_structured_data)
            print(
                f"Loaded pred draws for {len(all_draws['galaxy_ids'])} galaxies "
                f"in {time.time() - start_time:.2f} s"
            )
            if COMPUTE_DRAWS_PROFILES or any([
                PLOT_PRED_VS_TRUE_MEDIAN_CC,
                PLOT_PRED_VS_TRUE_MEDIAN_CC_CENTER,
                PLOT_PRED_VS_TRUE_MEDIAN_CC_ERRORBARS,
                PLOT_STD_PRED_VS_TRUE_MEDIAN_CC,
                PLOT_DEVIATION_DISTRIBUTION,
            ]):
                draw_profiles = compute_and_derive_draw_profiles(all_draws['x_pred_draws'])

        if cache is not None and cache.writable:
            cache.save_profiles(all_structured_data, draw_profiles)
            if all_draws is not None and len(all_draws['galaxy_ids']):
                for galaxy_idx in PER_GALAXY_INDICES:
                    row = _draws_row_for_galaxy(all_draws, galaxy_idx)
                    if row is not None:
                        cache.save_galaxy_draws(galaxy_idx, all_draws['x_pred_draws'][row])
            print(f"Wrote reduced figure data: {cache.describe()}")

    overlay = None
    overlay_cache = None
    if VERSION_TAG_2 and cache_mode != "off":
        overlay_cache = figure_cache.FigureCache(
            cache_dir or stars2dm_paths.CACHE_ROOT,
            cache_mode,
            kind="overlay",
            version=VERSION_TAG_2,
            primary=VERSION_TAG,
            pattern=DRAWS_FILE_PATTERN_2,
            num_draws=NUM_DRAWS_2,
            draw_slice=DRAWS_SLICE_2,
            ensemble=_pdl.ENSEMBLE_COMBINE,
            profile_agg=_pdl.PROFILE_AGG,
            denominator=DEVIATION_DENOMINATOR,
        )
        overlay_cache.require()
    if VERSION_TAG_2:
        data_dir_v2 = None
        if DATA_LAYOUT != "per_galaxy":
            sim_label_v2, _ = parse_version_tag(VERSION_TAG_2)
            data_dir_v2 = (
                Path(data_dir_2)
                if data_dir_2 is not None
                else effective_pred_root(PRED_ROOT, sim_label_v2)
            )
        radial_centers_v1 = all_structured_data['radial_centers'][0]
        overlay = load_second_version_bundle(
            data_dir_v2,
            all_structured_data['galaxy_ids'].tolist(),
            needs_draws_profiles(),
            version_tag=VERSION_TAG_2,
            pred_root=PRED_ROOT,
            num_draws=NUM_DRAWS_2,
            draws_dir=DRAWS_DATA_DIR_2,
            file_pattern=DRAWS_FILE_PATTERN_2,
            draw_slice=DRAWS_SLICE_2,
            radial_centers_fallback=radial_centers_v1,
            cache=overlay_cache,
        )

    if plot_pred_vs_true:
        fig, _ = plot_multiple_radial_centers(
            RADIAL_INDICES_MIXED,
            all_structured_data,
            overlay=overlay,
            highlight_galaxy_ids=HIGHLIGHT_GALAXY_IDS,
        )
        save_figure(fig, output_path("rhoDM_predict_vs_true"))

    if PLOT_PRED_VS_TRUE_MEDIAN_CC:
        for style in DREAMS_ALL_STYLES:
            DREAMS_ALL_STYLE = style
            fig, _ = plot_multiple_radial_centers(
                RADIAL_INDICES_MIXED,
                all_structured_data,
                xpred_to_plot=draw_profiles['combined_profile'],
                xpred_cc_to_plot=deviation_denominator(draw_profiles),
                axis_llim=AXIS_LLIM,
                axis_ulim=AXIS_ULIM,
                overlay=overlay,
                highlight_galaxy_ids=HIGHLIGHT_GALAXY_IDS,
            )
            save_figure(
                fig,
                output_path(
                    f"rhoDM_predict_vs_true_{combine_label()}_cc_perpixel_std"
                    f"{dreams_all_suffix()}"
                ),
            )

    if PLOT_PRED_VS_TRUE_MEDIAN_CC_CENTER:
        fig, _ = plot_multiple_radial_centers(
            RADIAL_INDICES_CENTER,
            all_structured_data,
            xpred_to_plot=draw_profiles['combined_profile'],
            xpred_cc_to_plot=deviation_denominator(draw_profiles),
            overlay=overlay,
            highlight_galaxy_ids=HIGHLIGHT_GALAXY_IDS,
        )
        save_figure(fig, output_path(f"rhoDM_predict_vs_true_{combine_label()}_cc_perpixel_std_center{dreams_all_suffix()}"))

    if PLOT_PRED_VS_TRUE_MEDIAN_CC_ERRORBARS:
        fig, _ = plot_multiple_radial_centers(
            RADIAL_INDICES_MIXED,
            all_structured_data,
            xpred_to_plot=draw_profiles['combined_profile'],
            xpred_cc_to_plot=deviation_denominator(draw_profiles),
            downsample_factor=1,
            do_error_bars=True,
            uerr=draw_profiles['uerr_perpixel_agg'],
            lerr=draw_profiles['lerr_perpixel_agg'],
            overlay=overlay,
            highlight_galaxy_ids=HIGHLIGHT_GALAXY_IDS,
        )
        save_figure(
            fig,
            output_path(f"rhoDM_predict_vs_true_{combine_label()}_cc_perpixel_std_downsample1_errorbars{dreams_all_suffix()}"),
        )

    if PLOT_STD_PRED_VS_TRUE:
        fig, _ = plot_multiple_radial_centers_scatter(
            RADIAL_INDICES_MIXED, all_structured_data, overlay=overlay
        )
        save_figure(fig, output_path("rhoDM_std_predict_vs_true"))

    if PLOT_STD_PRED_VS_TRUE_MEDIAN_CC:
        fig, _ = plot_multiple_radial_centers_scatter(
            RADIAL_INDICES_MIXED,
            all_structured_data,
            xpred_to_plot=draw_profiles['combined_profile'],
            xpred_cc_to_plot=deviation_denominator(draw_profiles),
            min_val_fixed=0.0,
            max_val_fixed=0.2,
            overlay=overlay,
        )
        save_figure(fig, output_path(f"rhoDM_std_predict_vs_true_{combine_label()}_cc_perpixel_std{dreams_all_suffix()}"))

    if PLOT_DEVIATION_DISTRIBUTION:
        fig, _ = plot_deviation_distribution(
            RADIAL_INDICES_MIXED,
            all_structured_data,
            xpred_to_plot=draw_profiles['combined_profile'],
            xpred_cc_to_plot=deviation_denominator(draw_profiles),
            overlay=overlay,
            highlight_galaxy_ids=HIGHLIGHT_GALAXY_IDS,
        )
        save_figure(fig, output_path(f"deviation_distribution_{combine_label()}_cc_perpixel_std"))

    overlay_draws = overlay.get('all_draws') if overlay is not None else None

    if PLOT_PER_GALAXY_DEVIATION:
        plot_per_galaxy_deviation(all_draws, GALAXY_IDX_LIST, overlay_draws=overlay_draws)

    if PLOT_PER_GALAXY_DRAW_PANELS:
        plot_per_galaxy_draw_panels(all_draws, GALAXY_IDX_LIST, overlay_draws=overlay_draws)

    if PLOT_PER_GALAXY_PRIOR_POSTERIOR:
        plot_per_galaxy_prior_posterior(all_draws, GALAXY_IDX_LIST, overlay_draws=overlay_draws)

    print("Done.")


def parse_galaxy_indices(spec):
    if spec is None:
        return None
    if ':' in spec:
        start, end = spec.split(':', 1)
        return np.arange(int(start), int(end))
    return np.array([int(x) for x in spec.split(',')])


def run_path_checks(pred_root=None):
    root = Path(pred_root or PRED_ROOT)
    print(f"OOD path resolution check (pred_root={root})")
    ok = True
    for label, path, exists in check_ood_path_examples(root):
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}: {path}")
        if not exists and label.endswith("draws"):
            ok = False
    return ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate radial profile comparison figures from saved npz data.'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Prediction version tag (default: VERSION_TAG in script config)',
    )
    parser.add_argument(
        '--version2',
        type=str,
        default=None,
        help='Second version tag overlaid on scatter/deviation plots',
    )
    parser.add_argument(
        '--data-dir-2',
        type=str,
        default=None,
        help='Override data directory for second version',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output figures (default: script directory)',
    )
    parser.add_argument(
        '--per-galaxy-indices',
        type=str,
        default=None,
        help=(
            'Comma-separated galaxy IDs to draw the per-galaxy example figures for '
            '(Figures 1-3). These are galaxy IDs, not rows into the loaded draws, so the '
            'loaded subset no longer has to start at 0.'
        ),
    )
    parser.add_argument(
        '--ensemble-combine',
        type=str,
        choices=['mean', 'median'],
        default=None,
        help=(
            'How the stochastic draws are collapsed into the fiducial prediction '
            '(default: mean). Use median to reproduce the pre-2026-08 figures.'
        ),
    )
    parser.add_argument(
        '--profile-agg',
        type=str,
        choices=['mean', 'median'],
        default=None,
        help='How pixels are combined inside an annulus (default: mean).',
    )
    parser.add_argument(
        '--dreams-all-style',
        type=str,
        default=None,
        help=(
            'How the train/val DREAMS galaxies appear behind the held-out ones in the '
            'predicted-vs-true figure: points, contours or hide (default: points). '
            'Accepts a comma-separated list, e.g. "points,contours,hide", to emit every '
            'variant from a single data load.'
        ),
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Override base data directory (default: derived from --version)',
    )
    parser.add_argument(
        '--radial-profile-dir',
        type=str,
        default=None,
        help='Directory with *_radial_profile.npz (default: base data dir)',
    )
    parser.add_argument(
        '--draws-dir',
        type=str,
        default=None,
        help='Directory with pred draws npz files (default: base data dir)',
    )
    parser.add_argument(
        '--draws-pattern',
        type=str,
        default=None,
        help='Glob pattern for draws files (default: DRAWS_FILE_PATTERN in config)',
    )
    parser.add_argument(
        '--draws-slice',
        type=str,
        default=None,
        help='Draw index slice start:end applied per galaxy file (e.g. 1:17)',
    )
    parser.add_argument(
        '--draws-output-label',
        type=str,
        default=None,
        help='Tag appended to output PNG names only (not used for file discovery)',
    )
    parser.add_argument(
        '--galaxy-indices',
        type=str,
        default=None,
        help='Galaxy index subset as start:end or comma-separated list (e.g. 0:2)',
    )
    parser.add_argument(
        '--check-paths',
        action='store_true',
        help='Print resolved DREAMS/TNG50/FIRE path examples and exit',
    )
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help=(
            'Quick run: galaxies 0:2 (unless --galaxy-indices set) and write PNGs to '
            'training_res/temp/radial_profile_figures/'
        ),
    )
    parser.add_argument(
        '--pred-root',
        type=str,
        default=None,
        help='Base pred directory with galaxy_XXXX subdirs (default: DREAMS SB5 pred root)',
    )
    parser.add_argument(
        '--data-layout',
        type=str,
        choices=['per_galaxy', 'flat'],
        default='per_galaxy',
        help='per_galaxy reads galaxy_XXXX/; flat uses legacy single directory',
    )
    parser.add_argument(
        '--num-draws',
        type=int,
        default=None,
        help='Number of draws in npz filename (default 10, or 17/50 for sweeps)',
    )
    parser.add_argument(
        '--draws-dir-2',
        type=str,
        default=None,
        help='Draws directory for VERSION_TAG_2 (default: OOD subdir under pred root)',
    )
    parser.add_argument(
        '--draws-pattern-2',
        type=str,
        default=None,
        help='Glob pattern for VERSION_TAG_2 draws files',
    )
    parser.add_argument(
        '--draws-slice-2',
        type=str,
        default=None,
        help='Draw slice start:end for VERSION_TAG_2 (e.g. 0:64)',
    )
    parser.add_argument(
        '--draws-output-label-2',
        type=str,
        default=None,
        help='PNG name tag for VERSION_TAG_2 overlay (not used for file discovery)',
    )
    parser.add_argument(
        '--num-draws-2',
        type=int,
        default=None,
        help='Number of draws in VERSION_TAG_2 npz filename',
    )
    parser.add_argument(
        '--hide-r2',
        action='store_true',
        help='Omit R^2 from scatter-panel text boxes',
    )
    parser.add_argument(
        '--deviation-denominator',
        type=str,
        default='profile',
        choices=['profile', 'perpixel'],
        help=(
            "Denominator for profile-level Delta residuals. 'profile' (default) uses "
            'the draw-to-draw spread of the annulus-averaged profile, which is the '
            'uncertainty matched to a profile-level residual. "perpixel" reproduces '
            'the legacy behavior (per-pixel (P84-P16)/2 medianed over the annulus), '
            'which inflates the denominator by ~1.8x at these radii'
        ),
    )
    parser.add_argument(
        '--deviation-hist-xlim',
        type=float,
        nargs=2,
        default=None,
        metavar=('LO', 'HI'),
        help=(
            'Plotted x-range of the Delta histograms (default -5 5). Widen it for the '
            'parameter-sweep runs, whose distributions are shifted far off zero with a '
            'long one-sided tail: the quoted mean must lie inside the range shown'
        ),
    )
    parser.add_argument(
        '--highlight-test-galaxies',
        action='store_true',
        help=(
            'On single-version (no --version2) predict-vs-true panels, highlight the '
            'held-out test-split galaxies (v2-overlay marker style) and show dual R^2 '
            '(test-only + all galaxies) in the panel text box'
        ),
    )
    parser.add_argument(
        '--test-galaxy-range',
        type=str,
        default='0:102',
        help=(
            'Galaxy ID range start:end counted as the held-out test split for '
            '--highlight-test-galaxies (default 0:102, matching DREAMSDataModule\'s '
            'test_idx=range(n_test) convention for the standard 820/102/102 split)'
        ),
    )

    figure_cache.add_cache_arguments(parser, default_root=str(stars2dm_paths.CACHE_ROOT))

    args = parser.parse_args()

    if args.pred_root is not None:
        PRED_ROOT = Path(args.pred_root)

    if args.check_paths:
        run_path_checks(args.pred_root)
        raise SystemExit(0)

    output_dir = args.output_dir
    galaxy_indices = args.galaxy_indices
    if args.smoke_test:
        if output_dir is None:
            output_dir = str(SMOKE_TEST_OUTPUT_DIR)
        if galaxy_indices is None:
            galaxy_indices = '0:2'

    main(
        data_dir=args.data_dir,
        output_dir=output_dir,
        galaxy_idx_list=parse_galaxy_indices(galaxy_indices),
        version_tag=args.version,
        version_tag_2=args.version2,
        data_dir_2=args.data_dir_2,
        radial_profile_dir=args.radial_profile_dir,
        draws_dir=args.draws_dir,
        draws_file_pattern=args.draws_pattern,
        draws_slice=args.draws_slice,
        draws_output_label=args.draws_output_label,
        draws_dir_2=args.draws_dir_2,
        draws_file_pattern_2=args.draws_pattern_2,
        draws_slice_2=args.draws_slice_2,
        draws_output_label_2=args.draws_output_label_2,
        num_draws_2=args.num_draws_2,
        pred_root=args.pred_root,
        data_layout=args.data_layout,
        num_draws=args.num_draws,
        show_r2=not args.hide_r2,
        deviation_denominator_mode=args.deviation_denominator,
        deviation_hist_xlim=args.deviation_hist_xlim,
        highlight_test_galaxies=args.highlight_test_galaxies,
        test_galaxy_range=args.test_galaxy_range,
        ensemble_combine=args.ensemble_combine,
        profile_agg=args.profile_agg,
        dreams_all_style=args.dreams_all_style,
        per_galaxy_indices=args.per_galaxy_indices,
        cache_dir=args.cache_dir,
        cache_mode=args.cache_mode,
    )
