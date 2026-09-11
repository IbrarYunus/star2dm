"""Compact main-text figures: calibration of the profile-level residual, and the
supernova-feedback parameter sweep.

Produces two 2x2 panels.

  calibration_2x2 : rows = raw / calibrated, columns = DREAMS test split / OOD.
      Shows that the raw stochastic ensemble is over-confident about the
      azimuthally averaged profile, and that the correction fixes it.

  sn1_sweep_2x2   : rows = e_w low / high, columns = inner / outer radius, all
      calibrated. Shows the sweep response and that it grows with radius.

The DREAMS calibration is LEAVE-ONE-OUT over the held-out split: the factor
applied to galaxy i is derived from the other 101 test galaxies only. No galaxy
contributes to its own correction, and no training galaxy is ever used, so the
calibrated panel is neither circular nor contaminated by seen data.

The true profile uses agg="mean" throughout, matching what the per-galaxy
radial_profile.npz files store and what build_structured_from_sim_maps now does.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import figure_cache
import stars2dm_paths
from paper_style import apply_paper_style, save_figure as save_paper_figure  # noqa: E402
from pred_data_loaders import (  # noqa: E402
    NORM_DICT,
    aggregate_pixels,
    build_radial_annulus_geometry,
    combine_draws,
    set_combiners,
)

RES = 128
N_TEST = 102
ROOT = str(stars2dm_paths.PRED_ROOT)
FIG_DIR = str(stars2dm_paths.DATA_ROOT / "figures")

DREAMS_TRUE = str(stars2dm_paths.DREAMS_RHODM_MAP)
TNG_TRUE = str(stars2dm_paths.TNG_RHODM_MAP)
FIRE_TRUE = str(stars2dm_paths.FIRE_RHODM_MAP)

MEAN_RHO = NORM_DICT["2D"]["rhoDM"][RES]["mean"]
STD_RHO = NORM_DICT["2D"]["rhoDM"][RES]["std"]

# Fiducial ensembles use 100 draws. At n=10 the ddof=0 std of the draws is biased low by
# sqrt((n-1)/n)*c4(n) = 0.923, which inflates every Delta built from it: the same estimator
# applied to a PERFECTLY calibrated Gaussian ensemble returns k = 1.27 at n=10 against 1.03 at
# n=100 (profile_calibration_audit.py::simulate_k). Measured k over the held-out split falls
# from 1.48 -> 2.13 to 1.28 -> 1.93 across r = 3.5 -> 24.7 kpc once n=100 removes that floor.
#
# DREAMS and TNG50 move together and are overridable from the CLI. FIRE does not: it has no
# counterpart to the DREAMS SN1/SN2/BHFF parametrization, so its ensemble marginalizes over
# those under a flat prior (100 configs x 1 draw, Om/s8 left at FIRE's true values) rather than
# sampling at fixed theta. Its spread therefore contains theta ignorance that DREAMS' and
# TNG50's do not, which is why the DREAMS-derived k is an UPPER BOUND on the correction FIRE
# needs rather than the right divisor for it.
DEFAULT_NDRAWS = 100
DEFAULT_SUFFIX = "heun100_n100"
FIRE_NDRAWS = 100
FIRE_SUFFIX = "heun100_FIREm12_flat_prior_baryon_100"

# The two suites order the suffix differently around "_pred_draws_"; these are written out
# rather than resolved through pred_data_loaders.draws_path() because that helper tries several
# candidate names and silently accepts whichever exists first. For a published figure the file
# that was read has to be unambiguous -- a silent fallback to the n=10 set is exactly the
# failure this whole exercise is about.
def dreams_path(g, ndraws=DEFAULT_NDRAWS, suffix=DEFAULT_SUFFIX):
    return f"{ROOT}/galaxy_{g:04d}/galaxy_{g}_v95_pred_draws_{ndraws}_{suffix}.npz"


def tng_path(g, ndraws=DEFAULT_NDRAWS, suffix=DEFAULT_SUFFIX):
    return (f"{ROOT}/TNG50/galaxy_{g:04d}/"
            f"galaxy_{g}_TNG50_v95_{suffix}_pred_draws_{ndraws}.npz")


def fire_path(g, ndraws=FIRE_NDRAWS, suffix=FIRE_SUFFIX):
    return (f"{ROOT}/FIREm12/galaxy_{g:04d}/"
            f"galaxy_{g}_FIREm12_v95_pred_draws_{ndraws}_{suffix}.npz")

XLIM = (-5.0, 5.0)
NBINS = 24
C_DREAMS = "#4878A6"
C_TNG = "#7FB2D5"
C_FIRE = "#1F3B60"
# Prints at a scale near 0.65, so these are ~10 pt and ~8.5 pt on the page.
AXLABEL_FS = 15
TICK_FS = 13
BOX_FS = 12

_, RADIAL_CENTERS, MASK = build_radial_annulus_geometry(RES)


def bin_index(r_kpc):
    return int(np.argmin(np.abs(RADIAL_CENTERS - r_kpc)))


def profile_delta(draws, true_map, bins):
    """Profile-level Delta per radial bin for one galaxy, plus the raw sigma.

    Numerator and denominator use the same annulus aggregation as the true profile;
    before 2026-08 the predicted side used the median over pixels while the true side
    used the mean, so the two halves of Delta were not defined the same way.
    """
    pred_med = combine_draws(draws, axis=0)
    out_d, out_s = [], []
    for bi in bins:
        m = MASK[bi]
        pred_prof = aggregate_pixels(pred_med[m])
        true_prof = np.mean(true_map[m])
        sig_prof = np.std(aggregate_pixels(draws[:, m], axis=1))
        out_d.append((pred_prof - true_prof) / sig_prof)
        out_s.append(sig_prof)
    return np.array(out_d), np.array(out_s)


def collect(true_path, path_fn, gal_ids, bins, n_expected=None, label=""):
    """Gather profile-level Delta for each galaxy that has a draws file.

    Missing galaxies are skipped rather than raised on, because the OOD suites are not
    contiguous. `n_expected` guards the case that actually matters: the draw count changed
    under us and a figure got built from a mix of n=10 and n=100 ensembles, whose estimator
    floors differ (1.27 vs 1.03) and which therefore cannot be compared in the same panel.
    """
    true_maps = np.load(true_path)
    deltas, ids = [], []
    for g in gal_ids:
        fp = path_fn(g)
        if not os.path.exists(fp):
            continue
        data = np.load(fp)
        draws = data["draws"]
        data.close()
        if draws.ndim == 4:
            draws = draws[:, 0]
        if n_expected is not None and draws.shape[0] != n_expected:
            raise ValueError(
                f"{label or 'sample'} galaxy {g}: expected {n_expected} draws, "
                f"found {draws.shape[0]} in {fp}"
            )
        draws = draws * STD_RHO + MEAN_RHO
        d, _ = profile_delta(draws, true_maps[g], bins)
        deltas.append(d)
        ids.append(g)
    if not deltas:
        raise FileNotFoundError(
            f"{label or 'sample'}: no draws files found (first path tried: "
            f"{path_fn(next(iter(gal_ids)))})"
        )
    return np.array(deltas), np.array(ids)


# Every panel in this module is built from collect()'s output, which is one profile-level
# Delta per galaxy per annulus -- kilobytes, from gigabytes of draws. So that is what gets
# cached, keyed by everything that changes it.
_CACHE_ROOT = None
_CACHE_MODE = "off"


def cached_collect(key, true_path, path_fn, gal_ids, bins, n_expected=None, label=""):
    cache = None
    if _CACHE_MODE != "off":
        cache = figure_cache.FigureCache(
            _CACHE_ROOT or stars2dm_paths.CACHE_ROOT, _CACHE_MODE,
            kind="calibration", sample=key, true_map=Path(true_path).name,
            n_galaxies=len(list(gal_ids)), n_expected=n_expected,
            bins=f"{len(bins)}:{bins[0]:.6f}:{bins[-1]:.6f}",
        )
        cache.require()
    if cache is not None and cache.readable:
        data, _ = cache.load_profiles()
        print(f"  {label or key}: from cache ({data['deltas'].shape[0]} galaxies)")
        return data["deltas"], data["ids"]
    deltas, ids = collect(true_path, path_fn, gal_ids, bins,
                          n_expected=n_expected, label=label)
    if cache is not None and cache.writable:
        cache.save_profiles({"deltas": deltas, "ids": ids}, None)
    return deltas, ids


def loo_calibrate(delta):
    """Leave-one-out calibration: galaxy i is divided by sigma of the others."""
    n = delta.shape[0]
    out = np.empty_like(delta)
    for i in range(n):
        others = np.delete(delta, i, axis=0)
        out[i] = delta[i] / np.std(others, axis=0)
    return out


def draw_hist(ax, vals, color, label, alpha=0.75, xlim=None):
    xlim = xlim or XLIM
    bins = np.linspace(xlim[0], xlim[1], NBINS + 1)
    ax.hist(vals, bins=bins, density=True, color=color, alpha=alpha,
            edgecolor="white", linewidth=0.6, label=label, zorder=2)


def draw_rug(ax, vals, color, label):
    """Individual objects, for samples too small to histogram honestly (FIRE, n=8)."""
    # Sits below the legend box in the upper corner, which would otherwise
    # obscure the rightmost points.
    y = ax.get_ylim()[1]
    ax.plot(vals, np.full_like(vals, 0.58 * y), "D", ms=7, color=color,
            mec="white", mew=0.8, label=label, zorder=4, clip_on=False)
    ax.vlines(vals, 0.54 * y, 0.62 * y, color=color, lw=1.0, alpha=0.5, zorder=3)
    ax.axvline(np.mean(vals), color=color, lw=1.8, ls="-", alpha=0.85, zorder=3)


def unit_gaussian(ax, xlim=None):
    xlim = xlim or XLIM
    x = np.linspace(xlim[0], xlim[1], 400)
    ax.plot(x, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi), "k--", lw=1.8,
            zorder=3, label="Unit normal")


def style(ax, title_lines, show_xlabel, show_ylabel, xlim=None):
    ax.set_xlim(*(xlim or XLIM))
    ax.axvline(0.0, color="0.35", lw=1.0, ls=":", zorder=1)
    ax.text(0.03, 0.97, "\n".join(title_lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=BOX_FS,
            bbox=dict(boxstyle="round", fc="white", ec="0.4", alpha=0.95))
    ax.tick_params(labelsize=TICK_FS)
    if show_xlabel:
        ax.set_xlabel(r"Residual [$\Delta$]", fontsize=AXLABEL_FS)
    else:
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("Probability", fontsize=AXLABEL_FS)
    else:
        ax.tick_params(labelleft=False)


def stat(v):
    return f"{np.mean(v):+.2f}$\\pm${np.std(v):.2f}"


def make_calibration_figure(r_kpc, outfile, ndraws=DEFAULT_NDRAWS,
                            suffix=DEFAULT_SUFFIX):
    bi = [bin_index(r_kpc)]
    r_lab = RADIAL_CENTERS[bi[0]]

    d_dreams, _ = cached_collect(
        "dreams_test",
        DREAMS_TRUE, lambda g: dreams_path(g, ndraws, suffix),
        range(N_TEST), bi, n_expected=ndraws, label="DREAMS")
    d_tng, _ = cached_collect(
        "tng50_ood",
        TNG_TRUE, lambda g: tng_path(g, ndraws, suffix),
        range(400), bi, n_expected=ndraws, label="TNG50")
    d_fire, _ = cached_collect(
        "fire_ood",
        FIRE_TRUE, fire_path,
        range(8), bi, n_expected=FIRE_NDRAWS, label="FIRE")

    print(f"  DREAMS test {d_dreams.shape[0]}, TNG50 {d_tng.shape[0]}, "
          f"FIRE {d_fire.shape[0]}  (n_draws: {ndraws}/{ndraws}/{FIRE_NDRAWS})")

    # DREAMS: leave-one-out. OOD: divided by k from the full held-out split
    # (OOD galaxies are not in that set, so this is already out-of-sample).
    d_dreams_cal = loo_calibrate(d_dreams)
    k = np.std(d_dreams, axis=0)
    d_tng_cal, d_fire_cal = d_tng / k, d_fire / k
    print(f"  k({r_lab:.1f} kpc) = {k[0]:.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)

    for row, (dr, tn, fr, tag) in enumerate([
        (d_dreams, d_tng, d_fire, "raw"),
        (d_dreams_cal, d_tng_cal, d_fire_cal, "calibrated"),
    ]):
        ax = axes[row][0]
        draw_hist(ax, dr[:, 0], C_DREAMS, f"DREAMS held-out ($N={dr.shape[0]}$)")
        unit_gaussian(ax)
        style(ax, [f"$r={r_lab:.1f}$ kpc, {tag}",
                   rf"$\Delta_{{\rm DREAMS}}=${stat(dr[:, 0])}"],
              row == 1, True)
        if row == 0:
            ax.legend(loc="upper right", fontsize=BOX_FS - 1, framealpha=0.9)

        ax = axes[row][1]
        draw_hist(ax, tn[:, 0], C_TNG, f"TNG50 ($N={tn.shape[0]}$)")
        unit_gaussian(ax)
        # FIRE has only 8 galaxies; a histogram would imply a density it cannot
        # support, so each galaxy is drawn individually with the sample mean.
        draw_rug(ax, fr[:, 0], C_FIRE, f"FIRE ($N={fr.shape[0]}$), individual")
        style(ax, [f"$r={r_lab:.1f}$ kpc, {tag}",
                   rf"$\Delta_{{\rm TNG50}}=${stat(tn[:, 0])}",
                   rf"$\Delta_{{\rm FIRE}}=${stat(fr[:, 0])}"],
              row == 1, False)
        if row == 0:
            ax.legend(loc="upper right", fontsize=BOX_FS - 1, framealpha=0.9)

    fig.tight_layout()
    save_paper_figure(fig, outfile, dpi=200)
    plt.close(fig)
    print(f"  saved {outfile}")


SWEEP_XLIM = (-7.0, 9.0)


def make_sweep_figure(r_inner, r_outer, outfile, ndraws=DEFAULT_NDRAWS,
                      suffix=DEFAULT_SUFFIX):
    bins = [bin_index(r_inner), bin_index(r_outer)]
    r_lab = [RADIAL_CENTERS[b] for b in bins]

    # k comes from the fiducial n=100 ensemble even though the sweep draws below stay at 50.
    # That is deliberate: k is the calibration factor being measured and must be free of the
    # n=10/n=50 estimator floor, whereas the sweep panels report residual MEANS, where the
    # ddof=0 bias at n=50 is 0.986 -- under 2%, against 165 GPU-h to re-run them at 100.
    d_fid, _ = cached_collect(
        "sweep_fiducial",
        DREAMS_TRUE, lambda g: dreams_path(g, ndraws, suffix),
        range(N_TEST), bins, n_expected=ndraws, label="DREAMS fiducial")
    k = np.std(d_fid, axis=0)
    print(f"  k = {k}  (from n={ndraws} fiducial ensemble)")

    true_maps = np.load(DREAMS_TRUE)
    lo, hi = [], []
    for g in range(1024):
        fp = (f"{ROOT}/galaxy_{g:04d}/galaxy_{g}_v95_pred_draws_50_"
              f"heun100_sn1_sweep_5.npz")
        if not os.path.exists(fp):
            continue
        data = np.load(fp)
        draws = data["draws"]
        data.close()
        if draws.ndim == 4:
            draws = draws[:, 0]
        draws = draws * STD_RHO + MEAN_RHO
        lo.append(profile_delta(draws[0:10], true_maps[g], bins)[0])
        hi.append(profile_delta(draws[40:50], true_maps[g], bins)[0])
    lo, hi = np.array(lo) / k, np.array(hi) / k
    print(f"  sweep galaxies: {lo.shape[0]}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True)
    rows = [(lo, r"$\bar{e}_w=0.25$ (weak)"), (hi, r"$\bar{e}_w=4$ (strong)")]
    for row, (vals, lab) in enumerate(rows):
        for col in range(2):
            ax = axes[row][col]
            draw_hist(ax, vals[:, col], C_DREAMS, lab, xlim=SWEEP_XLIM)
            unit_gaussian(ax, xlim=SWEEP_XLIM)
            style(ax, [f"{lab}, $r={r_lab[col]:.1f}$ kpc",
                       rf"$\Delta=${stat(vals[:, col])}"],
                  row == 1, col == 0, xlim=SWEEP_XLIM)
    fig.tight_layout()
    save_paper_figure(fig, outfile, dpi=200)
    plt.close(fig)
    print(f"  saved {outfile}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=3.5)
    ap.add_argument("--sweep-inner", type=float, default=3.5)
    ap.add_argument("--sweep-outer", type=float, default=24.7)
    ap.add_argument("--outdir", type=str, default=FIG_DIR)
    ap.add_argument("--num-draws", type=int, default=DEFAULT_NDRAWS,
                    help="Draws per DREAMS/TNG50 ensemble. FIRE is unaffected: its "
                         "flat-prior ensemble is a different quantity (see module header).")
    ap.add_argument("--draws-suffix", type=str, default=DEFAULT_SUFFIX,
                    help="Output-suffix of the draws files, e.g. heun100_n100 for the "
                         "fiducial n=100 set or heun100 for the older 10-draw set.")
    ap.add_argument("--ensemble-combine", type=str, choices=["mean", "median"], default=None,
                    help="How the stochastic draws collapse to the fiducial prediction "
                         "(default: mean).")
    figure_cache.add_cache_arguments(ap, default_root=str(stars2dm_paths.CACHE_ROOT))
    ap.add_argument("--profile-agg", type=str, choices=["mean", "median"], default=None,
                    help="How pixels are combined inside an annulus (default: mean).")
    args = ap.parse_args()

    # Module scope (this block is under `if __name__`), so a plain assignment rebinds the
    # module globals that cached_collect reads. No `global` statement -- it is invalid here.
    _CACHE_ROOT, _CACHE_MODE = args.cache_dir, args.cache_mode

    apply_paper_style()
    combine, agg = set_combiners(ensemble=args.ensemble_combine, profile=args.profile_agg)
    print(f"ensemble combine: {combine} | annulus aggregation: {agg}")

    os.makedirs(args.outdir, exist_ok=True)
    print("calibration figure:")
    make_calibration_figure(
        args.radius, f"{args.outdir}/v95_heun100_calibration_2x2.png",
        args.num_draws, args.draws_suffix)
    print("sweep figure:")
    make_sweep_figure(
        args.sweep_inner, args.sweep_outer,
        f"{args.outdir}/v95_heun100_sn1_sweep_2x2.png",
        args.num_draws, args.draws_suffix)
