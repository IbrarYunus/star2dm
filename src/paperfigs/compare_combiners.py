#!/usr/bin/env python3
"""Measure what switching the combiner from median to mean actually costs.

Two independent choices are being changed at once:

  ENSEMBLE_COMBINE  median -> mean over the stochastic draws (the fiducial prediction)
  PROFILE_AGG       median -> mean over the pixels of an annulus

Both move the numerator of the profile-level Delta, hence R^2, hence every number quoted
in the Results. Before rewriting any of that prose we need to know whether the shift is
cosmetic or substantive, so this reports the old and new values side by side over the
102 held-out DREAMS galaxies at the six radii used throughout the paper.

Run:
    python compare_combiners.py                    # DREAMS held-out, n=100
    python compare_combiners.py --n-galaxies 10    # quick check
"""

import argparse

import numpy as np

from plot_calibration_figures import (
    DREAMS_TRUE,
    MASK,
    MEAN_RHO,
    RADIAL_CENTERS,
    STD_RHO,
    dreams_path,
)

# The six radial bins plotted throughout the paper (RADIAL_INDICES_MIXED in
# plot_radial_profile_figures.py): r = 3.5, 6.4, 9.2, 20.5, 24.7, 40.3 kpc.
PAPER_BINS = [2, 4, 6, 14, 17, 28]


def r_squared(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-galaxies', type=int, default=102)
    ap.add_argument('--ndraws', type=int, default=100)
    args = ap.parse_args()

    true_maps = np.load(DREAMS_TRUE)
    n_bins = len(PAPER_BINS)

    truth = np.full((args.n_galaxies, n_bins), np.nan)
    pred_mean = np.full((args.n_galaxies, n_bins), np.nan)
    pred_median = np.full((args.n_galaxies, n_bins), np.nan)
    delta_mean = np.full((args.n_galaxies, n_bins), np.nan)
    delta_median = np.full((args.n_galaxies, n_bins), np.nan)

    for row, g in enumerate(range(args.n_galaxies)):
        data = np.load(dreams_path(g, ndraws=args.ndraws))
        draws = data['draws']
        data.close()
        if draws.ndim == 4:
            draws = draws[:, 0]
        if draws.shape[0] != args.ndraws:
            raise ValueError(f"galaxy {g}: expected {args.ndraws} draws, got {draws.shape[0]}")
        draws = draws * STD_RHO + MEAN_RHO

        per_pixel_mean = np.mean(draws, axis=0)
        per_pixel_median = np.median(draws, axis=0)

        for col, bi in enumerate(PAPER_BINS):
            m = MASK[bi]
            truth[row, col] = np.mean(true_maps[g][m])
            pred_mean[row, col] = np.mean(per_pixel_mean[m])
            pred_median[row, col] = np.median(per_pixel_median[m])
            sig_mean = np.std(np.mean(draws[:, m], axis=1))
            sig_median = np.std(np.median(draws[:, m], axis=1))
            delta_mean[row, col] = (pred_mean[row, col] - truth[row, col]) / sig_mean
            delta_median[row, col] = (pred_median[row, col] - truth[row, col]) / sig_median

        if (row + 1) % 20 == 0:
            print(f"  ...{row + 1}/{args.n_galaxies} galaxies", flush=True)

    print(f"\nDREAMS held-out, N={args.n_galaxies}, {args.ndraws} draws")
    print("mean/mean = new convention; median/median = the pre-2026-08 one\n")

    hdr = f"{'r [kpc]':>9} {'d(pred) dex':>12} {'max|d| dex':>11} " \
          f"{'R2 mean':>8} {'R2 med':>8} {'dR2':>7} " \
          f"{'Delta mean':>18} {'Delta median':>18}"
    print(hdr)
    print('-' * len(hdr))
    worst_r2 = 0.0
    worst_dex = 0.0
    for col, bi in enumerate(PAPER_BINS):
        diff = pred_mean[:, col] - pred_median[:, col]
        r2m = r_squared(truth[:, col], pred_mean[:, col])
        r2d = r_squared(truth[:, col], pred_median[:, col])
        worst_r2 = max(worst_r2, abs(r2m - r2d))
        worst_dex = max(worst_dex, float(np.max(np.abs(diff))))
        print(
            f"{RADIAL_CENTERS[bi]:9.1f} {np.mean(diff):+12.4f} {np.max(np.abs(diff)):11.4f} "
            f"{r2m:8.3f} {r2d:8.3f} {r2m - r2d:+7.3f} "
            f"{np.mean(delta_mean[:, col]):+9.2f}+/-{np.std(delta_mean[:, col]):<7.2f}"
            f"{np.mean(delta_median[:, col]):+9.2f}+/-{np.std(delta_median[:, col]):<7.2f}"
        )

    print(f"\nlargest |R2(mean) - R2(median)| over the six radii: {worst_r2:.4f}")
    print(f"largest per-galaxy |mean - median| profile shift:   {worst_dex:.4f} dex")


if __name__ == '__main__':
    main()
