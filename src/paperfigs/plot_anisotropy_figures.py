#!/usr/bin/env python3
"""
Paper figures for the azimuthal anisotropy metric.

Reads only the CSVs and the small per-draw npz written by analyze_anisotropy.py, so
restyling a panel costs seconds and never re-reads the 6.7 GB of prediction draws.

Panels, each written as its own PDF so they stack in LaTeX the way Figure 2 already does:

  {prefix}_galaxy_{g}_anisotropy_A2.pdf          A_2(r), truth vs posterior ensemble
  {prefix}_galaxy_{g}_anisotropy_phase.pdf       major-axis offset dpsi_2(r)
  {prefix}_anisotropy_population_delta.pdf       Delta^{A2} vs radius, 102 held-out
  {prefix}_anisotropy_population_alignment.pdf   <cos 2 dpsi> vs radius, with the null
  {prefix}_with_{prefix2}_anisotropy_alignment_hist.pdf   per-galaxy alignment, v95 vs v97

  python plot_anisotropy_figures.py --data-dir anisotropy_v95 --version v95_heun100_n100 \
      --data-dir-2 anisotropy_v97 --version-2 v97_heun100_n100 --output-dir <figures_pdf>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analyze_anisotropy import PAPER_M, circular_spread_deg
from paper_style import (
    FOV_HALF_KPC,
    anisotropy_alpha,
    apply_paper_style,
    paper_grid,
    save_figure,
    symmetric_ylim,
)

# --- Stacked-panel geometry ------------------------------------------------------------
# Copied verbatim from plot_radial_profile_figures.py:232-237 rather than imported, so
# this module does not pull in that 3500-line script. These panels are placed one above
# another in LaTeX at a common \linewidth: they must share a figure size AND an explicit
# margin box, and must be saved with tight=False, or bbox_inches='tight' crops each file
# to its own content and the axes land at different x offsets. If the originals change,
# change these with them.
RESIDUAL_PANEL_FIGSIZE = (10, 5)
RESIDUAL_PANEL_MARGINS = dict(left=0.160, right=0.975, bottom=0.225, top=0.965)
RESIDUAL_PANEL_LABEL_FONTSIZE = 30
RESIDUAL_PANEL_TICK_FONTSIZE = 26
RESIDUAL_PANEL_LEGEND_FONTSIZE = 22
RESIDUAL_PANEL_TEXTBOX_FONTSIZE = 24

# Same truth/posterior colours as the prior-posterior figure (fig:dreams_profile_comp):
# black is the truth, orange the stochastic ensemble.
# Truth black, prediction orange, band at alpha 0.30 -- the convention the prior/posterior
# figure already uses (plot_radial_profile_figures.PER_GALAXY_V1_COLOR = 'C1'). This was
# "darkorange" (#ff8c00), close to but not the same as C1 (#ff7f0e).
TRUTH_COLOR = "black"
POSTERIOR_COLOR = "C1"
POSTERIOR_BAND_ALPHA = 0.30
# The phase panel is a binned residual against radius, like the radial and azimuthal
# panels of the per-galaxy residual figure, which are drawn in black. Black also makes the
# per-point opacity encoding far easier to read than a light colour does.
PHASE_MARKER_COLOR = "black"
# Paired-histogram colours, matching plot_radial_profile_figures.py:212-215.
V1_HIST_COLOR = "steelblue"
V2_HIST_FILL_COLOR = "cornflowerblue"
V2_HIST_EDGE_COLOR = "darkblue"
V2_HIST_ALPHA = 0.45

NULL_COLOR = "0.45"

# Suite palette, taken verbatim from plot_calibration_figures.py:95-97 so DREAMS, TNG50
# and FIRE keep the same colours they already carry in Figures 5 and 6 of the draft.
C_DREAMS = "#4878A6"
C_TNG = "#7FB2D5"
C_FIRE = "#1F3B60"
SUITE_COLORS = {"DREAMS": C_DREAMS, "TNG50": C_TNG, "FIRE": C_FIRE}
SUITE_MARKERS = {"DREAMS": "o", "TNG50": "s", "FIRE": "D"}


def new_panel():
    """A residual-panel-shaped axis with the shared margin box."""
    fig = plt.figure(figsize=RESIDUAL_PANEL_FIGSIZE)
    fig.subplots_adjust(**RESIDUAL_PANEL_MARGINS)
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=RESIDUAL_PANEL_TICK_FONTSIZE)
    return fig, ax


def legend_headroom(ax, n_entries, row_fraction=0.135, pad=0.05):
    """
    Grow the top of the y axis so an upper-corner legend clears the data.

    The panels are saved with tight=False and a fixed margin box, so matplotlib will not
    reflow anything to make room: a legend placed over a curve simply covers it. Sizing
    the headroom from the entry count is the only thing that works at a fixed geometry.
    """
    fraction = min(0.75, n_entries * row_fraction + pad)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, bottom + (top - bottom) / (1.0 - fraction))
    return ax


def finish_panel(ax, xlabel, ylabel, r_min_kpc=None):
    ax.set_xlabel(xlabel, fontsize=RESIDUAL_PANEL_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=RESIDUAL_PANEL_LABEL_FONTSIZE)
    ax.set_xlim(0.0, FOV_HALF_KPC)
    if r_min_kpc is not None:
        # The metric is undefined inside r_min: those annuli hold too few pixels for an
        # azimuthal angle to mean anything. Shade rather than crop, so the reader can see
        # that the curve starts where it does by choice.
        ax.axvspan(0.0, r_min_kpc, color="0.55", alpha=0.16, linewidth=0, zorder=0.5)
    paper_grid(ax, alpha=0.3)
    return ax


def version_prefix(version, version_2=None):
    if version_2:
        return f"{version}_with_{version_2}"
    return version


def load_bundle(data_dir, version):
    """The four CSVs and the optional per-draw npz for one version tag."""
    data_dir = Path(data_dir)
    bundle = {
        "radial": pd.read_csv(data_dir / f"anisotropy_radial_{version}.csv"),
        "scalars": pd.read_csv(data_dir / f"anisotropy_scalars_{version}.csv"),
        "null": pd.read_csv(data_dir / f"anisotropy_null_{version}.csv"),
    }
    population_path = data_dir / f"anisotropy_population_{version}.csv"
    bundle["population"] = pd.read_csv(population_path) if population_path.exists() else None
    curves_path = data_dir / f"anisotropy_curves_{version}.npz"
    bundle["curves"] = np.load(curves_path) if curves_path.exists() else None
    return bundle


def per_draw_amplitudes(curves, galaxy_id):
    """(A_2 per draw, c_2 per draw, c_2 truth, radii) for one featured galaxy."""
    m_values = curves["m_values"]
    mi = int(np.flatnonzero(m_values == PAPER_M)[0])
    c_draws = curves[f"galaxy_{galaxy_id}_c_draws"][:, :, mi]
    c_true = curves[f"galaxy_{galaxy_id}_c_true"][:, mi]
    return np.abs(c_draws), c_draws, c_true, curves["radial_centers"]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def plot_galaxy_amplitude(bundle, galaxy_id, output_dir, prefix, r_min_kpc, dpi):
    amp_draws, _, c_true, radii = per_draw_amplitudes(bundle["curves"], galaxy_id)
    lo, hi = np.percentile(amp_draws, [16.0, 84.0], axis=0)

    fig, ax = new_panel()
    ax.fill_between(
        radii, lo, hi, color=POSTERIOR_COLOR, alpha=POSTERIOR_BAND_ALPHA, linewidth=0,
        label="Posterior 16-84th pct",
    )
    ax.plot(radii, amp_draws.mean(axis=0), color=POSTERIOR_COLOR, linewidth=2.5,
            label="Posterior mean")
    ax.plot(radii, np.abs(c_true), color=TRUTH_COLOR, linewidth=2.5, label="Truth")
    finish_panel(ax, "$r$ [kpc]", "$A_2$", r_min_kpc=r_min_kpc)
    ax.set_ylim(bottom=0.0)
    legend_headroom(ax, 3)
    ax.legend(loc="upper left", fontsize=RESIDUAL_PANEL_LEGEND_FONTSIZE)
    return save_figure(
        fig, output_dir / f"{prefix}_galaxy_{galaxy_id}_anisotropy_A2.png",
        dpi=dpi, tight=False,
    )


def plot_galaxy_phase(bundle, galaxy_id, output_dir, prefix, r_min_kpc, dpi):
    amp_draws, c_draws, c_true, radii = per_draw_amplitudes(bundle["curves"], galaxy_id)
    c_pred_mean = c_draws.mean(axis=0)
    dpsi = np.degrees(np.angle(c_pred_mean * np.conj(c_true))) / PAPER_M
    spread = circular_spread_deg(c_draws)

    # Annuli that are nearly axisymmetric carry an essentially random angle, so each point
    # is faded according to the TRUE amplitude: a large offset at A_2 ~ 0 is not a failure,
    # and the figure should not let it read as one. The error bar is faded with its own
    # marker, so a discounted point does not shout through its uncertainty.
    alphas = anisotropy_alpha(np.abs(c_true))

    fig, ax = new_panel()
    ax.axhline(0.0, color="0.7", linestyle="--", linewidth=1.4)
    for r_value, offset, err, alpha in zip(radii, dpsi, spread, alphas):
        ax.errorbar(r_value, offset, yerr=err, fmt="none", ecolor=PHASE_MARKER_COLOR,
                    alpha=float(alpha) * 0.55, elinewidth=1.8, capsize=3)
        ax.plot(r_value, offset, marker="o", markersize=10, color=PHASE_MARKER_COLOR,
                alpha=float(alpha), linestyle="none")
    finish_panel(ax, "$r$ [kpc]", r"$\Delta\psi_2$ [deg]", r_min_kpc=r_min_kpc)
    ax.set_ylim(-90.0, 90.0)
    ax.set_yticks([-90, -45, 0, 45, 90])

    alignment = float(
        bundle["scalars"].loc[
            bundle["scalars"]["galaxy_id"] == galaxy_id, "alignment"
        ].iloc[0]
    )
    ax.text(
        0.03, 0.95,
        rf"$\mathcal{{A}} = {alignment:.2f}$",
        transform=ax.transAxes, fontsize=RESIDUAL_PANEL_TEXTBOX_FONTSIZE,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    return save_figure(
        fig, output_dir / f"{prefix}_galaxy_{galaxy_id}_anisotropy_phase.png",
        dpi=dpi, tight=False,
    )


def plot_population_delta(bundle, output_dir, prefix, r_min_kpc, dpi):
    population = bundle["population"]
    fig, ax = new_panel()
    ax.axhline(0.0, color="0.7", linestyle="--", linewidth=1.4)
    ax.errorbar(
        population["radius_kpc"], population["delta_A2_mean"],
        yerr=population["delta_A2_std"], fmt="o", color=POSTERIOR_COLOR,
        markersize=8, elinewidth=2.0, capsize=4,
    )
    finish_panel(ax, "$r$ [kpc]", r"$\Delta^{A_2}$", r_min_kpc=r_min_kpc)
    symmetric_ylim(ax, minimum=1.0)
    return save_figure(
        fig, output_dir / f"{prefix}_anisotropy_population_delta.png",
        dpi=dpi, tight=False,
    )


def plot_population_alignment(bundle, output_dir, prefix, r_min_kpc, dpi):
    population = bundle["population"]
    null_mean = float(bundle["null"]["alignment_null"].mean())

    fig, ax = new_panel()
    ax.axhline(0.0, color="0.7", linestyle="--", linewidth=1.4,
               label="No orientation information")
    ax.axhline(null_mean, color=NULL_COLOR, linestyle=":", linewidth=2.0,
               label="Mismatched-galaxy null")
    ax.errorbar(
        population["radius_kpc"], population["cos2dpsi_mean"],
        yerr=population["cos2dpsi_sem"], fmt="o", color=POSTERIOR_COLOR,
        markersize=8, elinewidth=2.0, capsize=4,
    )
    finish_panel(ax, "$r$ [kpc]", r"$\langle\cos 2\Delta\psi_2\rangle$",
                 r_min_kpc=r_min_kpc)
    legend_headroom(ax, 2)
    ax.legend(loc="upper right", fontsize=RESIDUAL_PANEL_LEGEND_FONTSIZE)
    return save_figure(
        fig, output_dir / f"{prefix}_anisotropy_population_alignment.png",
        dpi=dpi, tight=False,
    )


def plot_population_overlay(bundles, output_dir, prefix, r_min_kpc, dpi, panel):
    """
    One population panel with every suite overlaid.

    bundles: list of (label, bundle) in draw order. `panel` is 'alignment' or 'delta'.

    Small horizontal offsets keep the error bars of the three suites from overprinting at
    shared radii; the offset is cosmetic and a fraction of the 1.41 kpc annulus width.
    """
    fig, ax = new_panel()
    ax.axhline(0.0, color="0.7", linestyle="--", linewidth=1.4)

    n_suites = len(bundles)
    for k, (label, bundle) in enumerate(bundles):
        population = bundle["population"]
        if population is None:
            print(f"{label}: no population CSV, skipped")
            continue
        color = SUITE_COLORS.get(label, POSTERIOR_COLOR)
        offset = 0.30 * (k - 0.5 * (n_suites - 1))
        if panel == "alignment":
            y, yerr = population["cos2dpsi_mean"], population["cos2dpsi_sem"]
        else:
            # The population std over galaxies, as in the single-suite panel: this is the
            # spread of the residual, not the uncertainty on its mean.
            y, yerr = population["delta_A2_mean"], population["delta_A2_std"]
        ax.errorbar(
            population["radius_kpc"] + offset, y, yerr=yerr,
            fmt=SUITE_MARKERS.get(label, "o"), color=color, markersize=7,
            elinewidth=1.8, capsize=3, alpha=0.9,
            label=f"{label} ($N={bundle['scalars'].shape[0]}$)",
        )

    if panel == "alignment":
        ylabel = r"$\langle\cos 2\Delta\psi_2\rangle$"
    else:
        ylabel = r"$\Delta^{A_2}$"
    finish_panel(ax, "$r$ [kpc]", ylabel, r_min_kpc=r_min_kpc)
    # A legend of one row per suite would reserve more than half the panel and squash the
    # data into the bottom third, so it goes on one line above the points instead.
    if panel == "alignment":
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        legend_headroom(ax, 1, row_fraction=0.20)
    else:
        symmetric_ylim(ax, minimum=1.0)
        legend_headroom(ax, 1, row_fraction=0.16)
    ax.legend(loc="upper center", ncol=n_suites, columnspacing=1.4,
              handletextpad=0.4, fontsize=RESIDUAL_PANEL_LEGEND_FONTSIZE - 5)
    return save_figure(
        fig, output_dir / f"{prefix}_anisotropy_population_{panel}_suites.png",
        dpi=dpi, tight=False,
    )


def plot_alignment_hist(bundle, bundle_2, label, label_2, output_dir, prefix, dpi,
                        n_bins=18):
    values = bundle["scalars"]["alignment"].values
    edges = np.linspace(-1.0, 1.0, n_bins + 1)

    fig, ax = new_panel()
    ax.hist(values, bins=edges, density=True, color=V1_HIST_COLOR, label=label)
    if bundle_2 is not None:
        ax.hist(
            bundle_2["scalars"]["alignment"].values, bins=edges, density=True,
            color=V2_HIST_FILL_COLOR, edgecolor=V2_HIST_EDGE_COLOR, alpha=V2_HIST_ALPHA,
            label=label_2,
        )
    ax.axvline(0.0, color="0.4", linestyle="--", linewidth=2.0,
               label="No orientation information")
    ax.hist(
        bundle["null"]["alignment_null"].values, bins=edges, density=True,
        histtype="step", color=NULL_COLOR, linewidth=2.0, linestyle=":",
        label="Mismatched-galaxy null",
    )
    finish_panel(ax, r"$\mathcal{A}$", "Normalized counts")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(bottom=0.0)
    legend_headroom(ax, 4 if bundle_2 is not None else 3)
    ax.legend(loc="upper left", fontsize=RESIDUAL_PANEL_LEGEND_FONTSIZE - 3)
    return save_figure(
        fig, output_dir / f"{prefix}_anisotropy_alignment_hist.png", dpi=dpi, tight=False,
    )


def report(bundle, label):
    """Print the numbers the draft prose has to quote, so they are in the job log."""
    scalars = bundle["scalars"]
    null = bundle["null"]
    print(f"\n=== {label}: {len(scalars)} galaxies ===")
    print(f"  alignment      median {scalars['alignment'].median():+.4f}   "
          f"mean {scalars['alignment'].mean():+.4f}   "
          f"frac > 0 {float((scalars['alignment'] > 0).mean()):.3f}")
    print(f"  amp-weighted   median {scalars['alignment_ampweighted'].median():+.4f}")
    print(f"  null           median {null['alignment_null'].median():+.4f}   "
          f"mean {null['alignment_null'].mean():+.4f}")
    print(f"  Delta^A2       mean {scalars['mean_delta_A2'].mean():+.3f}   "
          f"width {scalars['mean_delta_A2'].std(ddof=0):.3f}")
    print(f"  coherent frac  median {scalars['coherent_fraction'].median():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default="anisotropy_v95")
    parser.add_argument("--version", type=str, default="v95_heun100_n100")
    parser.add_argument("--data-dir-2", type=str, default=None,
                        help="Second version, drawn as the overlay in the histogram.")
    parser.add_argument("--version-2", type=str, default=None)
    parser.add_argument("--label", type=str, default=None,
                        help="Legend label for the primary version. Default: --version.")
    parser.add_argument("--label-2", type=str, default=None)
    parser.add_argument("--suite", action="append", default=[], metavar="LABEL:DIR:TAG",
                        help=("Add a suite to the population overlay panels, e.g. "
                              "'TNG50:anisotropy_TNG50_v95:TNG50_v95_heun100_n100'. "
                              "Repeatable. The primary --version is always first."))
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--galaxy", type=int, default=19,
                        help="Featured galaxy for the per-galaxy panels.")
    parser.add_argument("--r-min-kpc", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    apply_paper_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.data_dir, args.version)
    bundle_2 = None
    if args.data_dir_2 and args.version_2:
        bundle_2 = load_bundle(args.data_dir_2, args.version_2)

    prefix = args.version
    report(bundle, args.version)

    if bundle["curves"] is None:
        print(f"No per-draw npz in {args.data_dir}; per-galaxy panels skipped.")
    else:
        plot_galaxy_amplitude(bundle, args.galaxy, output_dir, prefix, args.r_min_kpc,
                              args.dpi)
        plot_galaxy_phase(bundle, args.galaxy, output_dir, prefix, args.r_min_kpc,
                          args.dpi)

    if bundle["population"] is None:
        print("No population CSV; population panels skipped.")
    else:
        plot_population_delta(bundle, output_dir, prefix, args.r_min_kpc, args.dpi)
        plot_population_alignment(bundle, output_dir, prefix, args.r_min_kpc, args.dpi)

    if args.suite:
        overlay_bundles = [("DREAMS", bundle)]
        for spec in args.suite:
            label, data_dir, tag = spec.split(":", 2)
            extra = load_bundle(data_dir, tag)
            report(extra, f"{label} ({tag})")
            overlay_bundles.append((label, extra))
        for panel in ("alignment", "delta"):
            plot_population_overlay(
                overlay_bundles, output_dir, prefix, args.r_min_kpc, args.dpi, panel
            )

    if bundle_2 is not None:
        report(bundle_2, args.version_2)
        plot_alignment_hist(
            bundle, bundle_2,
            args.label or args.version,
            args.label_2 or args.version_2,
            output_dir, version_prefix(args.version, args.version_2), args.dpi,
        )
    else:
        plot_alignment_hist(
            bundle, None, args.label or args.version, None, output_dir, prefix, args.dpi
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
