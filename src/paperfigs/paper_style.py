#!/usr/bin/env python3
"""Shared matplotlib styling for the paper figures.

The figure scripts grew independently and set no rcParams at all, so tick direction,
grids, fonts and DPI differ from panel to panel. Everything headed for the draft now
routes through here instead.

Usage:

    from paper_style import apply_paper_style, save_figure, no_grid, paper_grid

    apply_paper_style()          # once, at the top of main()
    ...
    save_figure(fig, out_path)   # writes .pdf (and a .png preview)

Two things are deliberately *not* global:

- ``bbox_inches='tight'`` is a per-call argument, not an rcParam. Panels that get
  stacked in LaTeX and expected to line up must NOT be tight-cropped, because tight
  cropping trims each file to its own content and the axes then land at different
  offsets once scaled to a common width.
- ``axes.grid`` is on globally, which is wrong for image panels. Call ``no_grid(ax)``
  on every ``imshow`` axis, and ``paper_grid(ax, ...)`` rather than ``ax.grid(True, ...)``
  on data panels that want their own alpha, so PAPER_GRID=0 still reaches them.
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Half-width of the 64 x 64 kpc field of view, and the largest radius any pixel in it
# reaches (the corners). Annuli between the two are only partially sampled.
FOV_HALF_KPC = 32.0
FOV_CORNER_KPC = float(np.sqrt(2.0) * FOV_HALF_KPC)   # 45.2548...

# Legend label for the N(0,1) reference curve. Must stay identical everywhere it is
# drawn so the same curve never acquires two different names across figures.
UNIT_NORMAL_LABEL = 'Unit normal'

# Whether the grid is drawn at all. Set PAPER_GRID=0 in the environment to build the
# no-grid variant of the whole figure set; the sbatch driver flips it and sends that run
# to its own output directory. Read once at import, so a single process is internally
# consistent.
GRID_ON = os.environ.get('PAPER_GRID', '1').strip().lower() not in ('0', 'false', 'no', 'off')

PAPER_RC = {
    # Serif for text AND math, so mathtext labels match their surroundings and the whole
    # figure matches the Times-like body font of the AASTeX class. STIXGeneral is the
    # Times metric-clone that ships with matplotlib and is the family the 'stix' mathtext
    # set is drawn from, so text and math stay on the same face.
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',

    # Ticks inward on all four sides, with minors. Majors are twice the minors so the two
    # stay distinguishable at print size.
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 8.0,
    'ytick.major.size': 8.0,
    'xtick.minor.size': 4.0,
    'ytick.minor.size': 4.0,

    # Baseline tick-label size. Most panels override this with an explicit labelsize, so
    # this only reaches the axes nobody styled by hand.
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,

    # Grid under the data, unless switched off for the no-grid variant.
    'axes.grid': GRID_ON,
    'axes.axisbelow': True,
    'grid.color': '0.75',
    'grid.linestyle': '-',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.35,

    'axes.linewidth': 1.0,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.6',

    # Embed real fonts rather than Type 3, which some journals reject.
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'figure.dpi': 110,
}


def apply_paper_style():
    """Install the paper rcParams. Idempotent; safe to call more than once."""
    mpl.rcParams.update(PAPER_RC)


def no_grid(ax):
    """Turn the grid off for an image panel (imshow), where it would overlay the data."""
    ax.grid(False)
    return ax


def paper_grid(ax, **kwargs):
    """Turn the grid on for a data panel, honouring the global GRID_ON switch.

    Several panels want a grid at their own alpha rather than the rcParam default, and a
    bare ``ax.grid(True, alpha=...)`` would re-enable it in the no-grid build. Route those
    calls through here instead.
    """
    if GRID_ON:
        ax.grid(True, **kwargs)
    else:
        ax.grid(False)
    return ax


def save_figure(fig, path, dpi=None, tight=True, close=True, png_preview=True):
    """Save ``fig`` as PDF, plus an optional PNG preview alongside.

    ``path`` may carry any extension (existing callers pass ``.png``); it is replaced.
    The PDF is the deliverable. The PNG goes in a ``png_preview/`` subdirectory so the
    figure folder handed to Overleaf stays pure PDF, while the result can still be
    eyeballed without a PDF viewer.

    ``tight=False`` preserves the exact figure geometry, which is what panels stacked in
    LaTeX need in order to align.
    """
    path = Path(path)
    stem = path.with_suffix('')
    kwargs = {'dpi': dpi if dpi is not None else mpl.rcParams['savefig.dpi']}
    if tight:
        kwargs['bbox_inches'] = 'tight'

    pdf_path = stem.with_suffix('.pdf')
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, **kwargs)
    print(f"Saved {pdf_path}")

    if png_preview:
        png_path = pdf_path.parent / 'png_preview' / (pdf_path.stem + '.png')
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, **kwargs)

    if close:
        plt.close(fig)
    return pdf_path


def symmetric_ylim(ax, pad=0.08, minimum=None):
    """Make the y-axis symmetric about zero.

    Residual panels are read against zero, so an axis running -1.5 to +1.0 makes an
    unbiased galaxy look biased. Called on every binned-Delta panel.
    """
    lo, hi = ax.get_ylim()
    half = max(abs(lo), abs(hi))
    if minimum is not None:
        half = max(half, minimum)
    half *= (1.0 + pad)
    ax.set_ylim(-half, half)
    return ax


def mark_fov_limit(ax, label=False, color='0.55'):
    """Mark where annuli stop being fully contained in the square field of view.

    A circle of radius FOV_HALF_KPC is the largest one inscribed in the 64 x 64 kpc
    frame. Beyond it only the corners contribute, so those bins average over a shrinking
    and increasingly anisotropic set of pixels.
    """
    ax.axvline(FOV_HALF_KPC, color=color, linestyle=':', linewidth=1.4, zorder=1.5)
    ax.axvspan(
        FOV_HALF_KPC, FOV_CORNER_KPC,
        color=color, alpha=0.16, linewidth=0, zorder=0.5,
        label='Outside image square' if label else None,
    )
    return ax


# Opacity encoding for the anisotropy phase panels. The major-axis angle of a nearly
# axisymmetric annulus is ill-defined, so those points are faded rather than dropped; the
# reader needs to see at a glance which points carry weight.
#
# Two things make the naive mapping (alpha proportional to A_2 / max A_2) hard to read:
# the A_2 values inside one galaxy rarely reach zero, so the alpha range is compressed
# into roughly 0.4-1.0, and alpha differences in that upper band are perceptually small.
# Stretching min-to-max recovers the full range, and the panels use black markers, where
# a given alpha step is far more visible than it is in a light colour.
ANISOTROPY_ALPHA_MIN = 0.15


def anisotropy_alpha(amplitudes, alpha_min=ANISOTROPY_ALPHA_MIN):
    """Per-point opacity from a set of amplitudes, stretched over the full [alpha_min, 1].

    A flat input maps to fully opaque rather than to alpha_min: if every annulus is
    equally anisotropic there is no reason to fade any of them.
    """
    a = np.asarray(amplitudes, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.ones_like(a)
    return alpha_min + (1.0 - alpha_min) * (a - lo) / (hi - lo)


def to_degrees(theta_radians):
    """Azimuthal bin centers in radians -> degrees."""
    return np.asarray(theta_radians) * 180.0 / np.pi


def style_azimuthal_axis(ax, xlabel_fontsize=None):
    """Put an azimuthal-angle axis on a -180..180 degree scale with 90-degree ticks."""
    ax.set_xlim(-180.0, 180.0)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticks(np.arange(-180, 181, 45), minor=True)
    if xlabel_fontsize is not None:
        ax.set_xlabel('Azimuthal angle [deg]', fontsize=xlabel_fontsize)
    return ax
