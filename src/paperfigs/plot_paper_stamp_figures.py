#!/usr/bin/env python3
"""
Stamp-grid figures for the paper: a grid of randomly sampled test galaxies, each
showing the same single-galaxy diagnostic panel (residual Delta map, histogram of
Delta, prior/posterior radial profile, radial/azimuthal dependence of the residual,
the input stellar-density condition map, the true dark-matter map, or the predicted
dark-matter map).

Reuses the per-galaxy building blocks already defined in plot_radial_profile_figures.py
(map loading, radial-annulus geometry, prior/posterior radial-profile helper) rather
than duplicating them; this script only adds the random-sampling rule and the
grid-of-galaxies layout.

Test-galaxy sampling rule
--------------------------
- Plain DREAMS version tags (no sim prefix, e.g. "v91", "v56_heun100"): sampled only
  from the test split, galaxy indices [0, n_test). n_test depends on when the model
  was trained: v91 onward uses the standard 80/10/10 split (n_test=102); earlier
  versions used 860/78/86 (n_test=86). See docs/final_model_pipeline.md, Section 1,
  for the provenance of this split-size change.
- "TNG50" / "FIREm12" version tags (out-of-domain sims): no train/val/test split
  exists for these, so any galaxy with available prediction draws is eligible.
- A second version tag (as used for v1/v2 overlays elsewhere in this codebase) is not
  applicable here -- the stamp grid is a single-version diagnostic figure, and the
  sampling pool is determined solely by the (single) version tag passed in.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

from analyze_anisotropy import (
    DEFAULT_R_MAX_KPC,
    DEFAULT_R_MIN_KPC,
    M_VALUES,
    PAPER_M,
    annulus_fourier_geometry,
    circular_spread_deg,
    fourier_moments,
    galaxy_statistics,
    linear_sigma,
)

import figure_cache
import pred_data_loaders as _pdl
import stars2dm_paths
from pred_data_loaders import (
    DEFAULT_PRED_ROOT,
    combine_draws,
    default_draws_file_pattern,
    discover_galaxy_indices,
    load_pred_data_structured as load_pred_per_galaxy,
    parse_version_tag,
    set_combiners,
)

from paper_style import (
    FOV_HALF_KPC,
    anisotropy_alpha,
    apply_paper_style,
    mark_fov_limit,
    no_grid,
    save_figure as save_paper_figure,
    style_azimuthal_axis,
    symmetric_ylim,
    to_degrees,
)

from plot_radial_profile_figures import (
    COND_MAP_VMAX,
    COND_MAP_VMIN,
    DEVIATION_HIST_XLIM,
    PER_GALAXY_SUMMARY_N_AZIMUTHAL,
    PER_GALAXY_SUMMARY_N_RINGS,
    PER_GALAXY_V1_COLOR,
    RHO_MAP_VMAX,
    RHO_MAP_VMIN,
    build_radial_annulus_geometry,
    compute_radial_profile_from_map,
    compute_radial_profiles_from_map_stack,
    load_sim_maps,
    resolve_sim_for_maps,
)

# ---------------------------------------------------------------------------
# DREAMS test-split rule (see module docstring)
# ---------------------------------------------------------------------------

DREAMS_TEST_SPLIT_VERSION_THRESHOLD = 91
DREAMS_TEST_N_LEGACY = 86
DREAMS_TEST_N_CURRENT = 102

_VERSION_NUMBER_RE = re.compile(r'^v(\d+)')

# The original eight: everything built from the residual map, the input/output maps, or
# the radial profile. These run on the 10-draw ensemble (see the STAGE=support call in
# submit_paper_figures_pdf.sh).
RESIDUAL_PANEL_TYPES = (
    'deviation_map',
    'deviation_hist',
    'deviation_radial',
    'deviation_azimuthal',
    'prior_posterior',
    'input_stellar_map',
    'truth_dm_map',
    'pred_dm_map',
)

# Stamp-scale versions of the two panels of the main text's per-galaxy anisotropy figure.
# There is deliberately no stamp counterpart of the POPULATION anisotropy figure: both of
# its panels are already aggregates over galaxies, and their per-galaxy content is either
# a monotone remapping of the phase panel below (cos 2 dpsi_2) or the difference between
# the two curves of the A_2 panel below (Delta^{A2}) -- a grid of either would repeat
# what these two already show.
#
# These must run on the SAME 100-draw ensemble as the figures they mirror: Delta^{A2}'s
# denominator is a ddof=0 ensemble std (biased low by x0.92 at N=10 against x0.99 at
# N=100) and a 16-84th percentile band is meaningless from ten samples.
ANISOTROPY_PANEL_TYPES = (
    'anisotropy_A2',
    'anisotropy_phase',
)

PANEL_TYPES = RESIDUAL_PANEL_TYPES + ANISOTROPY_PANEL_TYPES

# Panels built from the azimuthal Fourier moments rather than from the residual map.
_ANISOTROPY_PANELS = frozenset(ANISOTROPY_PANEL_TYPES)

# Named groups accepted by --panel-type, so one invocation can select the set that shares
# an ensemble size. 'all' keeps its literal meaning and therefore mixes the two.
PANEL_TYPE_GROUPS = {
    'all': PANEL_TYPES,
    'residual': RESIDUAL_PANEL_TYPES,
    'anisotropy': ANISOTROPY_PANEL_TYPES,
}

OUTPUT_DIR = Path(__file__).resolve().parent / 'stamp'

# Map-panel image extent, matching the grid the maps were built on (see the radial_loss_*
# / grid entries in the BaryonBridge configs: xmin/xmax/ymin/ymax = -32..32 kpc). Passing
# this as imshow(extent=...) puts the map axes in kpc rather than pixel index, so the
# spatial scale is readable off the figure.
MAP_EXTENT_KPC = (-32.0, 32.0, -32.0, 32.0)

# The stamp grids print at about 0.7 of their drawn size -- a much gentler downscale than
# the scatter figures -- so they carry their own colorbar-label size rather than importing
# AXIS_LABEL_FONTSIZE, which is tuned for a scale near 0.5 and would print oversized here.
STAMP_CBAR_LABEL_FONTSIZE = 14

# Anisotropy phase panels are binned residuals against radius, like deviation_radial and
# deviation_azimuthal in this same grid, which are black. Mirrors
# plot_anisotropy_figures.PHASE_MARKER_COLOR so Figure 5 and its stamp grid match.
PHASE_MARKER_COLOR = 'black'

# (xlabel, ylabel) per panel type; applied to edge panels only by _apply_edge_labels.
_PANEL_AXIS_LABELS = {
    'deviation_map': ('$x$ [kpc]', '$y$ [kpc]'),
    'deviation_hist': (r'Residual [$\Delta$]', 'Probability'),
    'deviation_radial': ('Radius [kpc]', r'Residual [$\Delta$]'),
    'deviation_azimuthal': ('Azimuthal angle [deg]', r'Residual [$\Delta$]'),
    'prior_posterior': ('Radius [kpc]', r'$\log\,\rho_{\rm DM}$'),
    'input_stellar_map': ('$x$ [kpc]', '$y$ [kpc]'),
    'truth_dm_map': ('$x$ [kpc]', '$y$ [kpc]'),
    'pred_dm_map': ('$x$ [kpc]', '$y$ [kpc]'),
    'anisotropy_A2': ('Radius [kpc]', '$A_2$'),
    'anisotropy_phase': ('Radius [kpc]', r'$\Delta\psi_2$ [deg]'),
}


# Panel types whose axes genuinely span the same range in every panel (the maps all use
# MAP_EXTENT_KPC). Only for these is it safe to drop interior tick labels -- the line and
# histogram panels are autoscaled per galaxy, so blanking their tick labels would imply a
# shared scale that does not exist.
_SHARED_RANGE_PANELS = frozenset(
    {'deviation_map', 'input_stellar_map', 'truth_dm_map', 'pred_dm_map'}
)


def _apply_edge_labels(ax, i, n_cols, n_rows, n_galaxies, panel_type, fontsize=14):
    """Put the axis labels on the left column and bottom edge of the grid only.

    The bottom-edge test is 'nothing is drawn below me' rather than 'I am in the last
    row', so a partially filled final row still gets its x labels.

    Tick labels are left alone except for the shared-range map panels, where interior
    ones are dropped to cut clutter (every map covers the same kpc extent, so the edge
    ticks apply to all of them).
    """
    xlabel, ylabel = _PANEL_AXIS_LABELS[panel_type]
    is_left = (i % n_cols == 0)
    is_bottom = (i + n_cols >= n_galaxies)

    if is_left:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if is_bottom:
        ax.set_xlabel(xlabel, fontsize=fontsize)

    if panel_type in _SHARED_RANGE_PANELS:
        if not is_left:
            ax.tick_params(labelleft=False)
        if not is_bottom:
            ax.tick_params(labelbottom=False)


def dreams_model_version_number(version_tag):
    """Parse the integer after the leading 'v' in a plain DREAMS version tag.

    e.g. 'v91' -> 91, 'v56_heun100' -> 56. Raises for OOD-prefixed tags (TNG50_*,
    FIREm12_*, ...), which don't have a train/val/test split to look up.
    """
    sim_label, model_version = parse_version_tag(version_tag)
    if sim_label is not None:
        raise ValueError(
            f"{version_tag!r} is an OOD tag (sim={sim_label!r}); only plain DREAMS "
            f"version tags (e.g. 'v91') have a version number for the split rule"
        )
    match = _VERSION_NUMBER_RE.match(model_version)
    if not match:
        raise ValueError(
            f"Could not parse a leading 'vNN' version number from {version_tag!r}"
        )
    return int(match.group(1))


def dreams_test_galaxy_count(version_tag):
    """Test-split size (n_test) for a plain DREAMS version tag; test = [0, n_test)."""
    version_number = dreams_model_version_number(version_tag)
    if version_number >= DREAMS_TEST_SPLIT_VERSION_THRESHOLD:
        return DREAMS_TEST_N_CURRENT
    return DREAMS_TEST_N_LEGACY


def eligible_galaxy_pool(
    version_tag,
    pred_root=DEFAULT_PRED_ROOT,
    num_draws=10,
    file_pattern=None,
    max_galaxy_id=None,
):
    """Galaxy indices the stamp grid may sample from, per the rule in the module docstring."""
    sim_label, _ = parse_version_tag(version_tag)
    pattern = file_pattern or default_draws_file_pattern(version_tag, num_draws)
    available = discover_galaxy_indices(
        pred_root, version_tag, num_draws=num_draws, file_pattern=pattern
    )
    if not available:
        raise FileNotFoundError(
            f"No draw files found for version={version_tag!r} pattern={pattern!r} "
            f"under {pred_root}"
        )

    if sim_label is None:
        n_test = dreams_test_galaxy_count(version_tag)
        pool = [g for g in available if g < n_test]
    elif sim_label in ("TNG50", "FIREm12"):
        pool = list(available)
    else:
        raise ValueError(
            f"No stamp-sampling rule defined for sim_label={sim_label!r} "
            f"(supported: plain DREAMS, TNG50, FIREm12)"
        )

    if max_galaxy_id is not None:
        pool = [g for g in pool if g < max_galaxy_id]
    if not pool:
        raise ValueError(
            f"No eligible galaxies for version={version_tag!r} after applying the "
            f"sampling rule (available={len(available)})"
        )
    return sorted(pool)


def sample_stamp_galaxy_ids(
    version_tag,
    n_galaxies,
    seed=42,
    pred_root=DEFAULT_PRED_ROOT,
    num_draws=10,
    file_pattern=None,
    max_galaxy_id=None,
):
    pool = eligible_galaxy_pool(
        version_tag,
        pred_root=pred_root,
        num_draws=num_draws,
        file_pattern=file_pattern,
        max_galaxy_id=max_galaxy_id,
    )
    if len(pool) < n_galaxies:
        raise ValueError(
            f"Requested {n_galaxies} galaxies but only {len(pool)} eligible "
            f"for version={version_tag!r}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(pool), size=n_galaxies, replace=False)
    return sorted(int(g) for g in chosen)


# ---------------------------------------------------------------------------
# Per-panel-type drawing (compact, stamp-scale versions of the single-galaxy panels)
# ---------------------------------------------------------------------------


def _style_map_axes(ax, max_ticks=3):
    """Sparse kpc ticks for a stamp-sized map panel (labels added only on edge panels)."""
    no_grid(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks))
    ax.tick_params(labelsize=12, direction='in', color='0.5')


def _panel_deviation_map(ax, draws_row, true_map_g, vmin=-5.0, vmax=5.0):
    x_pred_median = combine_draws(draws_row, axis=0)
    x_pred_std = np.std(draws_row, axis=0)
    deviation_map = (x_pred_median - true_map_g) / x_pred_std
    im = ax.imshow(
        deviation_map, cmap='bwr', vmin=vmin, vmax=vmax, aspect='equal', origin='lower',
        extent=MAP_EXTENT_KPC,
    )
    _style_map_axes(ax)
    return im, deviation_map


def _panel_deviation_hist(ax, draws_row, true_map_g, xlim=DEVIATION_HIST_XLIM, n_bins=40):
    x_pred_median = combine_draws(draws_row, axis=0)
    x_pred_std = np.std(draws_row, axis=0)
    deviation_map = (x_pred_median - true_map_g) / x_pred_std
    flat = deviation_map.flatten()

    hist_bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    ax.hist(flat, bins=hist_bins, density=True, color='steelblue', alpha=0.7)
    mean_sigma = float(np.mean(flat))
    std_sigma = float(np.std(flat))
    ax.axvline(mean_sigma, color='r', linestyle='--', linewidth=1.2)

    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    ax.plot(x, y, 'k--', linewidth=1.2)

    ax.set_xlim(xlim)
    ax.tick_params(labelsize=12, direction='in')
    return mean_sigma, std_sigma


def _panel_input_stellar_map(ax, cond_map_g, vmin=COND_MAP_VMIN, vmax=COND_MAP_VMAX):
    im = ax.imshow(
        cond_map_g, cmap='gray_r', vmin=vmin, vmax=vmax, aspect='equal', origin='lower',
        extent=MAP_EXTENT_KPC,
    )
    _style_map_axes(ax)
    return im


def _panel_truth_dm_map(ax, true_map_g, vmin=RHO_MAP_VMIN, vmax=RHO_MAP_VMAX):
    im = ax.imshow(
        true_map_g, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal', origin='lower',
        extent=MAP_EXTENT_KPC,
    )
    _style_map_axes(ax)
    return im


def _panel_pred_dm_map(ax, draws_row, vmin=RHO_MAP_VMIN, vmax=RHO_MAP_VMAX):
    x_pred_median = combine_draws(draws_row, axis=0)
    im = ax.imshow(
        x_pred_median, cmap='viridis', vmin=vmin, vmax=vmax, aspect='equal', origin='lower',
        extent=MAP_EXTENT_KPC,
    )
    _style_map_axes(ax)
    return im


def _radial_azimuthal_deviation_stats(
    deviation_map,
    n_rings=PER_GALAXY_SUMMARY_N_RINGS,
    n_azimuthal=PER_GALAXY_SUMMARY_N_AZIMUTHAL,
    xmin=-32, xmax=32, ymin=-32, ymax=32,
):
    """Mean +/- std of deviation_map binned by radius only and by azimuthal angle only.

    Mirrors the radial/azimuthal marginals computed in
    plot_radial_profile_figures.compute_deviation_summary, without the joint r x theta
    binning (not needed for a single-curve stamp panel) or the plotting/saving side
    effects of that function.
    """
    resolution = deviation_map.shape[0]
    r_max = float(np.sqrt(xmax**2 + ymax**2))
    num_rbins = min(n_rings, int(resolution // 4))
    num_thetabins = min(n_azimuthal, 8)

    radial_edges = np.linspace(0.0, r_max, num_rbins + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    theta_edges = np.linspace(-np.pi, np.pi, num_thetabins + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")
    r_coords = np.sqrt(x_coords**2 + y_coords**2)
    theta_coords = np.arctan2(y_coords, x_coords)

    radial_mean = np.zeros(num_rbins)
    radial_std = np.zeros(num_rbins)
    for i in range(num_rbins):
        r1, r2 = radial_edges[i], radial_edges[i + 1]
        in_bin = (
            (r_coords >= r1) & (r_coords < r2) if i < num_rbins - 1
            else (r_coords >= r1) & (r_coords <= r2)
        )
        radial_mean[i] = np.mean(deviation_map[in_bin])
        radial_std[i] = np.std(deviation_map[in_bin])

    theta_mean = np.zeros(num_thetabins)
    theta_std = np.zeros(num_thetabins)
    for i in range(num_thetabins):
        t1, t2 = theta_edges[i], theta_edges[i + 1]
        in_bin = (
            (theta_coords >= t1) & (theta_coords < t2) if i < num_thetabins - 1
            else (theta_coords >= t1) & (theta_coords <= t2)
        )
        theta_mean[i] = np.mean(deviation_map[in_bin])
        theta_std[i] = np.std(deviation_map[in_bin])

    return radial_centers, radial_mean, radial_std, theta_centers, theta_mean, theta_std


def _panel_deviation_radial(ax, draws_row, true_map_g):
    x_pred_median = combine_draws(draws_row, axis=0)
    x_pred_std = np.std(draws_row, axis=0)
    deviation_map = (x_pred_median - true_map_g) / x_pred_std
    radial_centers, radial_mean, radial_std, *_ = _radial_azimuthal_deviation_stats(deviation_map)
    ax.axhline(0, color='0.7', linestyle='--', linewidth=1.0)
    ax.errorbar(
        radial_centers, radial_mean, yerr=radial_std,
        fmt='o', color='black', elinewidth=1.2, capsize=3, markersize=3,
    )
    ax.tick_params(labelsize=12, direction='in')
    symmetric_ylim(ax)
    mark_fov_limit(ax)


def _panel_deviation_azimuthal(ax, draws_row, true_map_g):
    x_pred_median = combine_draws(draws_row, axis=0)
    x_pred_std = np.std(draws_row, axis=0)
    deviation_map = (x_pred_median - true_map_g) / x_pred_std
    *_, theta_centers, theta_mean, theta_std = _radial_azimuthal_deviation_stats(deviation_map)
    ax.axhline(0, color='0.7', linestyle='--', linewidth=1.0)
    ax.errorbar(
        to_degrees(theta_centers), theta_mean, yerr=theta_std,
        fmt='o', color='black', elinewidth=1.2, capsize=3, markersize=3,
    )
    ax.tick_params(labelsize=12, direction='in')
    style_azimuthal_axis(ax)
    symmetric_ylim(ax)


def _anisotropy_stats(draws_row, true_map_g, geometry):
    """
    Fourier moments and every derived statistic for one galaxy, at stamp scale.

    Routed through analyze_anisotropy.galaxy_statistics so a number read off a stamp panel
    is the same number that module wrote to its CSVs -- these panels are a second
    rendering of that metric, not a second implementation of it. The per-draw amplitudes
    are returned alongside, because the percentile band needs the full ensemble.
    """
    c_true = fourier_moments(linear_sigma(true_map_g), geometry)
    c_draws = fourier_moments(linear_sigma(draws_row), geometry)
    stats = galaxy_statistics(c_true, c_draws, m_values=geometry["m_values"],
                              paper_m=PAPER_M)
    mode_index = int(np.flatnonzero(geometry["m_values"] == PAPER_M)[0])
    return stats, np.abs(c_draws[:, :, mode_index]), c_draws[:, :, mode_index]


def _style_anisotropy_axes(ax):
    """Figure 4's x frame: 0 to the inscribed-circle radius, inner cut shaded.

    The metric is only defined over DEFAULT_R_MIN_KPC..DEFAULT_R_MAX_KPC, so the curves
    stop short of the frame at both ends. Shading the excluded inner annuli rather than
    cropping to the data keeps every panel on the same x scale as the figure these grids
    corroborate, and shows that the curve starts where it does by choice.
    """
    ax.set_xlim(0.0, FOV_HALF_KPC)
    ax.axvspan(0.0, DEFAULT_R_MIN_KPC, color='0.55', alpha=0.16, linewidth=0, zorder=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=12, direction='in')
    return ax


def _panel_anisotropy_A2(ax, draws_row, true_map_g, geometry):
    """Stamp version of the top panel of the main-text anisotropy figure."""
    stats, amp_draws, _ = _anisotropy_stats(draws_row, true_map_g, geometry)
    radii = geometry["radial_centers"]
    mode_index = int(np.flatnonzero(geometry["m_values"] == PAPER_M)[0])
    lo, hi = np.percentile(amp_draws, [16.0, 84.0], axis=0)

    ax.fill_between(radii, lo, hi, color=PER_GALAXY_V1_COLOR, alpha=0.30, linewidth=0)
    ax.plot(radii, amp_draws.mean(axis=0), color=PER_GALAXY_V1_COLOR, linewidth=1.6)
    ax.plot(radii, stats["amp_true"][:, mode_index], color='black', linewidth=1.6)
    ax.set_ylim(bottom=0.0)
    _style_anisotropy_axes(ax)


def _panel_anisotropy_phase(ax, draws_row, true_map_g, geometry):
    """Stamp version of the bottom panel of the main-text anisotropy figure."""
    stats, _, c_draws_m = _anisotropy_stats(draws_row, true_map_g, geometry)
    radii = geometry["radial_centers"]
    mode_index = int(np.flatnonzero(geometry["m_values"] == PAPER_M)[0])
    amp_true = stats["amp_true"][:, mode_index]

    ax.axhline(0, color='0.7', linestyle='--', linewidth=1.0)
    # Opacity scaled by the TRUE amplitude, as in the main-text panel: a large offset in a
    # nearly axisymmetric annulus is not a failure and must not read as one. Black markers,
    # matching the other binned-residual panels of this grid, and the error bar fades with
    # its own marker so a discounted point does not shout through its uncertainty.
    spread = circular_spread_deg(c_draws_m)
    alphas = anisotropy_alpha(amp_true)
    for radius, offset, err, alpha in zip(radii, stats["dpsi_deg"], spread, alphas):
        ax.errorbar(radius, offset, yerr=err, fmt='none', ecolor=PHASE_MARKER_COLOR,
                    alpha=float(alpha) * 0.55, elinewidth=1.2, capsize=2)
        ax.plot(radius, offset, marker='o', markersize=3.5, color=PHASE_MARKER_COLOR,
                alpha=float(alpha), linestyle='none')
    ax.set_ylim(-90.0, 90.0)
    ax.set_yticks([-90, -45, 0, 45, 90])
    _style_anisotropy_axes(ax)






def _panel_prior_posterior(ax, radial_centers, true_profile, prior_mean, posterior_profiles):
    posterior_median = combine_draws(posterior_profiles, axis=0, nan_safe=True)
    posterior_lo = np.nanpercentile(posterior_profiles, 16, axis=0)
    posterior_hi = np.nanpercentile(posterior_profiles, 84, axis=0)

    ax.plot(radial_centers, prior_mean, color='0.5', linestyle='--', linewidth=1.0)
    ax.fill_between(
        radial_centers, posterior_lo, posterior_hi,
        color=PER_GALAXY_V1_COLOR, alpha=0.30, linewidth=0,
    )
    ax.plot(radial_centers, posterior_median, color=PER_GALAXY_V1_COLOR, linewidth=1.6)
    ax.plot(radial_centers, true_profile, color='black', linewidth=1.6)
    ax.tick_params(labelsize=12, direction='in')


def _panel_type_label(panel_type):
    return {
        'deviation_map': r'Residual [$\Delta$] map',
        'deviation_hist': r'Residual [$\Delta$] histogram',
        'deviation_radial': r'Radial dependence of residual [$\Delta$]',
        'deviation_azimuthal': r'Azimuthal dependence of residual [$\Delta$]',
        'prior_posterior': 'Prior / posterior radial profile',
        'input_stellar_map': r'Input stellar-density map',
        'truth_dm_map': r'True dark-matter map',
        'pred_dm_map': r'Predicted dark-matter map (median)',
        'anisotropy_A2': r'Elongation amplitude $A_2$',
        'anisotropy_phase': r'Major-axis offset $\Delta\psi_2$',
    }[panel_type]


_MAP_PANEL_COLORBAR_LABELS = {
    'input_stellar_map': r'log $\rho_{\mathrm{star}}$',
    'truth_dm_map': r'log $\rho_{\mathrm{DM}}$',
    'pred_dm_map': r'log $\rho_{\mathrm{DM}}$',
}


# ---------------------------------------------------------------------------
# Grid figure
# ---------------------------------------------------------------------------


def plot_galaxy_stamp_grid(
    version_tag,
    panel_type,
    n_galaxies=16,
    n_cols=4,
    seed=42,
    pred_root=DEFAULT_PRED_ROOT,
    num_draws=10,
    file_pattern=None,
    draw_slice=None,
    channel=0,
    galaxy_ids=None,
    panel_size=2.6,
    save_path=None,
    save_dpi=200,
    show_suptitle=False,
    cache_dir=None,
    cache_mode="auto",
):
    """Grid of stamp-sized panels, one per randomly sampled galaxy, all the same panel_type.

    galaxy_ids: explicit override (skips random sampling entirely) -- e.g. to reuse
    the same galaxies across panel_type calls for a paper figure set.

    show_suptitle: off by default -- journal style puts the figure description in the
    caption, not in an in-figure title. Set True (CLI: --suptitle) for standalone
    diagnostic use, where the sim/version/panel-type label is useful on its own.
    """
    if panel_type not in PANEL_TYPES:
        raise ValueError(f"Unknown panel_type={panel_type!r}; choose from {PANEL_TYPES}")

    sim_for_maps = resolve_sim_for_maps(version_tag)
    x_map_true, cond_map_true = load_sim_maps(sim_for_maps)

    if galaxy_ids is None:
        galaxy_ids = sample_stamp_galaxy_ids(
            version_tag,
            n_galaxies,
            seed=seed,
            pred_root=pred_root,
            num_draws=num_draws,
            file_pattern=file_pattern,
            max_galaxy_id=x_map_true.shape[0],
        )
    else:
        galaxy_ids = sorted(int(g) for g in galaxy_ids)
        n_galaxies = len(galaxy_ids)

    pattern = file_pattern or default_draws_file_pattern(version_tag, num_draws)

    # The stamp grids read draws for 16 galaxies and no more, so the cache for them is
    # simply those galaxies' draws. Sampling is seeded and depends on the galaxy pool, so
    # the resolved ids go into the key: a cache built for one seed cannot answer another.
    cache = None
    if cache_mode != "off":
        cache = figure_cache.FigureCache(
            cache_dir or stars2dm_paths.CACHE_ROOT,
            cache_mode,
            kind="stamp",
            version=version_tag,
            pattern=pattern,
            num_draws=num_draws,
            galaxies=",".join(str(int(g)) for g in galaxy_ids),
            draw_slice=draw_slice,
            channel=channel,
            ensemble=_pdl.ENSEMBLE_COMBINE,
            profile_agg=_pdl.PROFILE_AGG,
        )
        cache.require()

    cached_ids = cache.cached_galaxy_ids() if cache is not None else []
    if cache is not None and cached_ids == sorted(int(g) for g in galaxy_ids):
        print(f"Reading stamp draws from cache: {cache.describe()}")
        draws_ids = np.array(cached_ids)
        draws_arr = np.stack([cache.load_galaxy_draws(g) for g in cached_ids])
    else:
        draws_data = load_pred_per_galaxy(
            pred_root,
            version_tag,
            num_draws=num_draws,
            galaxy_indices=galaxy_ids,
            draw_slice=draw_slice,
            channel=channel,
            file_pattern=pattern,
        )
        draws_ids = draws_data['galaxy_ids']
        draws_arr = draws_data['x_pred_draws']
        if cache is not None and cache.writable:
            for row, gid in enumerate(draws_ids):
                cache.save_galaxy_draws(gid, draws_arr[row])
            # A stamp entry has no population arrays, but meta.json is what makes the
            # directory self-describing, so write an empty profiles entry to produce it.
            cache.save_profiles({'galaxy_ids': np.asarray(draws_ids)}, None)
            print(f"Wrote stamp draws to cache: {cache.describe()}")

    if panel_type in _ANISOTROPY_PANELS:
        # Geometry only, so it is built once here rather than per galaxy per draw.
        fourier_geometry = annulus_fourier_geometry(
            draws_arr.shape[-1], DEFAULT_R_MIN_KPC, DEFAULT_R_MAX_KPC, m_values=M_VALUES
        )

    if panel_type == 'prior_posterior':
        resolution = draws_arr.shape[-1]
        _, radial_centers, mask = build_radial_annulus_geometry(resolution)
        n_galaxies_maps = x_map_true.shape[0]
        all_true_profiles = np.array([
            compute_radial_profile_from_map(x_map_true[i], mask)
            for i in range(n_galaxies_maps)
        ])

    n_rows = int(np.ceil(n_galaxies / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    im_for_cbar = None
    for i, gid in enumerate(draws_ids):
        ax = axes_flat[i]
        draws_row = draws_arr[i]

        if panel_type == 'deviation_map':
            im, _ = _panel_deviation_map(ax, draws_row, x_map_true[gid])
            im_for_cbar = im
        elif panel_type == 'deviation_hist':
            _panel_deviation_hist(ax, draws_row, x_map_true[gid])
        elif panel_type == 'deviation_radial':
            _panel_deviation_radial(ax, draws_row, x_map_true[gid])
        elif panel_type == 'deviation_azimuthal':
            _panel_deviation_azimuthal(ax, draws_row, x_map_true[gid])
        elif panel_type == 'input_stellar_map':
            im = _panel_input_stellar_map(ax, cond_map_true[gid])
            im_for_cbar = im
        elif panel_type == 'truth_dm_map':
            im = _panel_truth_dm_map(ax, x_map_true[gid])
            im_for_cbar = im
        elif panel_type == 'pred_dm_map':
            im = _panel_pred_dm_map(ax, draws_row)
            im_for_cbar = im
        elif panel_type == 'anisotropy_A2':
            _panel_anisotropy_A2(ax, draws_row, x_map_true[gid], fourier_geometry)
        elif panel_type == 'anisotropy_phase':
            _panel_anisotropy_phase(ax, draws_row, x_map_true[gid], fourier_geometry)
        elif panel_type == 'prior_posterior':
            true_profile = all_true_profiles[gid]
            other_mask = np.ones(n_galaxies_maps, dtype=bool)
            other_mask[gid] = False
            prior_mean = np.nanmean(all_true_profiles[other_mask], axis=0)
            posterior_profiles = compute_radial_profiles_from_map_stack(draws_row, mask)
            _panel_prior_posterior(ax, radial_centers, true_profile, prior_mean, posterior_profiles)
        else:
            # Reached only if a name was added to PANEL_TYPES without a branch here. It
            # used to fall into the prior/posterior path and die on an undefined name
            # several lines later; say so at the point of failure instead.
            raise ValueError(f"No drawing branch for panel_type={panel_type!r}")

        _apply_edge_labels(ax, i, n_cols, n_rows, len(draws_ids), panel_type)

        ax.text(
            0.04, 0.96, f'gal {gid}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, pad=0.2),
        )

    for j in range(n_galaxies, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if panel_type in _ANISOTROPY_PANELS:
        fig.tight_layout()

    if panel_type == 'deviation_map' and im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02
        )
        cbar.set_label(r'Residual [$\Delta$]', fontsize=STAMP_CBAR_LABEL_FONTSIZE)
    elif panel_type in _MAP_PANEL_COLORBAR_LABELS and im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar, ax=axes_flat.tolist(), orientation='vertical', fraction=0.02, pad=0.02
        )
        cbar.set_label(_MAP_PANEL_COLORBAR_LABELS[panel_type], fontsize=STAMP_CBAR_LABEL_FONTSIZE)
    elif panel_type == 'prior_posterior':
        legend_handles = [
            Line2D([0], [0], color='black', lw=1.6, label='True'),
            Line2D([0], [0], color=PER_GALAXY_V1_COLOR, lw=1.6, label='Posterior median'),
            Patch(facecolor=PER_GALAXY_V1_COLOR, alpha=0.30, label='Posterior 16-84%'),
            Line2D([0], [0], color='0.5', lw=1.0, ls='--', label='Prior mean'),
        ]
        # Reserve a band at the top for the legend and let tight_layout fill the rest
        # with the axes. Without this the default subplot top margin -- previously
        # occupied by the suptitle -- is saved as a white gap above the panels, and
        # simply nudging the legend anchor down instead makes it overlap the top row.
        legend_frac = 0.35 / (panel_size * n_rows)
        fig.tight_layout(rect=[0, 0, 1, 1 - legend_frac])
        fig.legend(
            handles=legend_handles, loc='upper center', ncol=4, fontsize=10,
            bbox_to_anchor=(0.5, 1.0), frameon=True,
        )

    if show_suptitle:
        sim_label, model_version = parse_version_tag(version_tag)
        sim_desc = sim_label or 'DREAMS'
        fig.suptitle(
            f'{sim_desc} {model_version} — {_panel_type_label(panel_type)} (n={n_galaxies})',
            y=1.0 + 0.55 / (panel_size * n_rows),
            fontsize=13,
        )

    if save_path is not None:
        save_paper_figure(fig, save_path, dpi=save_dpi)
    else:
        plt.show()

    return fig, axes, list(draws_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_galaxy_id_list(spec):
    if spec is None:
        return None
    return [int(x) for x in spec.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate stamp-grid figures: a grid of randomly sampled test galaxies, '
            'each showing the same single-galaxy diagnostic panel.'
        )
    )
    parser.add_argument(
        '--version', type=str, required=True,
        help="Version tag, e.g. 'v91', 'TNG50_v91', 'FIREm12_v91'",
    )
    parser.add_argument(
        '--panel-type', type=str, default='all',
        choices=list(PANEL_TYPES) + sorted(PANEL_TYPE_GROUPS),
        help=(
            "Which panel to draw for each galaxy. A group name generates every panel type "
            "in it as separate figures: 'residual' is the original eight (10-draw "
            "ensemble), 'anisotropy' the four Fourier panels (100-draw), 'all' both."
        ),
    )
    parser.add_argument('--n-galaxies', type=int, default=16)
    parser.add_argument('--n-cols', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for galaxy sampling')
    parser.add_argument('--pred-root', type=str, default=str(DEFAULT_PRED_ROOT))
    parser.add_argument('--num-draws', type=int, default=10)
    parser.add_argument(
        '--draws-pattern', type=str, default=None,
        help='Glob pattern for draws files (default: derived from --version/--num-draws)',
    )
    parser.add_argument(
        '--galaxy-ids', type=str, default=None,
        help='Comma-separated explicit galaxy IDs; overrides random sampling entirely',
    )
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    parser.add_argument('--save-dpi', type=int, default=200)
    parser.add_argument(
        '--suptitle', action='store_true',
        help='Draw the in-figure title (off by default; journal style puts it in the caption)',
    )
    parser.add_argument(
        '--ensemble-combine', type=str, choices=['mean', 'median'], default=None,
        help=(
            'How the stochastic draws are collapsed into the fiducial prediction '
            '(default: mean). Use median to reproduce the pre-2026-08 figures.'
        ),
    )
    parser.add_argument(
        '--profile-agg', type=str, choices=['mean', 'median'], default=None,
        help='How pixels are combined inside an annulus (default: mean).',
    )
    figure_cache.add_cache_arguments(parser, default_root=str(stars2dm_paths.CACHE_ROOT))

    args = parser.parse_args()

    apply_paper_style()
    combine, agg = set_combiners(ensemble=args.ensemble_combine, profile=args.profile_agg)
    print(f"ensemble combine: {combine} | annulus aggregation: {agg}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    galaxy_ids = parse_galaxy_id_list(args.galaxy_ids)

    panel_types = list(PANEL_TYPE_GROUPS.get(args.panel_type, (args.panel_type,)))
    for panel_type in panel_types:
        save_path = output_dir / f"{args.version}_stamp_{panel_type}_n{args.n_galaxies}.png"
        plot_galaxy_stamp_grid(
            args.version,
            panel_type,
            n_galaxies=args.n_galaxies,
            n_cols=args.n_cols,
            seed=args.seed,
            pred_root=Path(args.pred_root),
            num_draws=args.num_draws,
            file_pattern=args.draws_pattern,
            galaxy_ids=galaxy_ids,
            save_path=save_path,
            save_dpi=args.save_dpi,
            show_suptitle=args.suptitle,
            cache_dir=args.cache_dir,
            cache_mode=args.cache_mode,
        )


if __name__ == '__main__':
    main()
