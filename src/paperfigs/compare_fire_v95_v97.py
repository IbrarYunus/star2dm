"""
FIRE (m12) out-of-domain comparison between v95 (theta-conditioned) and v97 (no-theta ablation).

v95's FIRE predictions marginalize over a log-uniform prior on (SN1, SN2, BHFF): 100 configs x
1 draw per galaxy.  v97 ignores theta entirely (condition_on_parameters=false), so its FIRE
predictions are 10 i.i.d. stochastic draws at a single (unused) theta.  This script quantifies
how much of v95's FIRE predictive scatter comes from that prior marginalization rather than from
the interpolant's own stochasticity, and whether averaging over the prior shifts the central
prediction away from the fiducial-theta one.

Method: the flat prior is log-uniform and centered on theta = (1, 1, 1) in log10 space, so a
quadratic response surface in log10(theta) fit across the 100 draws gives, per pixel,
  - fit at log10(theta) = 0            -> the fiducial-theta prediction (= intercept coefficient)
  - mean of the 100 draws              -> the prior-marginalized prediction (OLS: = mean of fit)
  - explained / residual variance      -> theta-induced vs purely stochastic scatter
The Jensen gap (marginalized minus fiducial) is the prior-induced shift of the central prediction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import stars2dm_paths
from pred_data_loaders import build_radial_annulus_geometry, denormalize_draws

PRED_ROOT = stars2dm_paths.PRED_ROOT
FIRE_TRUE_MAPS = stars2dm_paths.FIRE_RHODM_MAP
DREAMS_TRUE_MAPS = stars2dm_paths.DREAMS_RHODM_MAP
FIRE_PARAM_CSV = (
    PRED_ROOT / "FIREm12" / "param_config_used_heun100_FIREm12_flat_prior_baryon_100.csv"
)

N_FIRE = 8
DREAMS_TEST = range(102)
SWEPT = ["SN1", "SN2", "BHFF"]

# Same bin grouping as report_radial_bias_metrics.py
GROUPS = {
    "inner": list(range(0, 7)),
    "mid": list(range(7, 14)),
    "outer": list(range(14, 32)),
    "all": list(range(0, 32)),
}


def fire_draws_path(gal, tag, n_draws=10):
    """v95's FIRE run is the 100-config flat-prior sweep; v97's is n_draws i.i.d. draws."""
    gdir = PRED_ROOT / "FIREm12" / f"galaxy_{gal:04d}"
    if tag == "v95":
        return gdir / (
            f"galaxy_{gal}_FIREm12_v95_pred_draws_100_heun100_"
            "FIREm12_flat_prior_baryon_100.npz"
        )
    suffix = "heun100" if n_draws == 10 else f"heun100_n{n_draws}"
    return gdir / f"galaxy_{gal}_FIREm12_{tag}_{suffix}_pred_draws_{n_draws}.npz"


def dreams_draws_path(gal, tag, n_draws):
    gdir = PRED_ROOT / f"galaxy_{gal:04d}"
    if n_draws == 100:
        return gdir / f"galaxy_{gal}_{tag}_pred_draws_100_heun100_n100.npz"
    return gdir / f"galaxy_{gal}_{tag}_pred_draws_10_heun100.npz"


def load_draws(path, tag):
    draws = np.load(path)["draws"]
    if draws.ndim == 4:
        draws = draws[:, 0]
    return denormalize_draws(draws.astype(np.float64), 128, tag)


def quadratic_design(theta):
    """[1, l, l^2, cross-terms] in log10(theta); fiducial theta=(1,1,1) maps to the intercept."""
    l = np.log10(theta)
    cols = [np.ones(len(l))]
    cols += [l[:, i] for i in range(3)]
    cols += [l[:, i] ** 2 for i in range(3)]
    cols += [l[:, i] * l[:, j] for i, j in [(0, 1), (0, 2), (1, 2)]]
    return np.column_stack(cols)


def bin_stat(values_2d, mask, group_bins, agg=np.median):
    """Aggregate a per-pixel map over the pixels of a radial-bin group."""
    sel = mask[group_bins].any(axis=0)
    return agg(values_2d[sel])


def group_pixels(mask, group_bins):
    return mask[group_bins].any(axis=0)


def accumulate_accuracy(acc, key, pred_map, true_map, mask):
    for gname, bins in GROUPS.items():
        sel = group_pixels(mask, bins)
        resid = pred_map[sel] - true_map[sel]
        rec = acc.setdefault((key, gname), {"resid": [], "true": []})
        rec["resid"].append(resid)
        rec["true"].append(true_map[sel])


def finalize_accuracy(acc):
    rows = []
    for (key, gname), rec in acc.items():
        resid = np.concatenate(rec["resid"])
        true = np.concatenate(rec["true"])
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        rows.append(
            {
                "config": key,
                "radial_group": gname,
                "median_bias_dex": np.median(resid),
                "mean_bias_dex": resid.mean(),
                "mae_dex": np.mean(np.abs(resid)),
                "rmse_dex": np.sqrt(np.mean(resid**2)),
                "r2_pixel": 1.0 - ss_res / ss_tot,
                "n_pixels": resid.size,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(Path(__file__).parent / "fire_v95_v97_compare"))
    ap.add_argument("--v97-fire-draws", type=int, default=10,
                    help="draw count of the v97 FIRE run to use (10 or 100)")
    ap.add_argument("--n-dreams", type=int, default=102,
                    help="number of DREAMS test galaxies used for the in-domain scatter reference")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, radial_centers, mask = build_radial_annulus_geometry(128)

    theta_df = pd.read_csv(FIRE_PARAM_CSV)
    theta = theta_df[SWEPT].to_numpy()
    X = quadratic_design(theta)
    n_draw, n_coef = X.shape
    pinv_X = np.linalg.pinv(X)
    print(f"FIRE flat prior: {n_draw} configs, quadratic design with {n_coef} coefficients")
    print(f"v97 FIRE draws per galaxy: {args.v97_fire_draws}")
    print("prior log10 midpoint per param:", np.round(np.log10(theta).mean(axis=0), 4))
    print(
        "prior ranges:",
        {p: (round(theta[:, i].min(), 3), round(theta[:, i].max(), 3)) for i, p in enumerate(SWEPT)},
    )

    fire_true = np.load(FIRE_TRUE_MAPS)
    dreams_true = np.load(DREAMS_TRUE_MAPS)

    acc = {}
    scatter_rows = []
    shift_rows = []

    # ---------------- FIRE ----------------
    for gal in range(N_FIRE):
        true_map = fire_true[gal].astype(np.float64)

        d95 = load_draws(fire_draws_path(gal, "v95"), "v95")
        Y = d95.reshape(n_draw, -1)
        coef = pinv_X @ Y
        fitted = X @ coef
        resid = Y - fitted
        ss_res = np.sum(resid**2, axis=0)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
        var_resid = ss_res / (n_draw - n_coef)          # purely stochastic (theta removed)
        var_total = ss_tot / (n_draw - 1)               # prior-marginalized posterior width
        var_theta = np.clip(var_total - var_resid, 0.0, None)

        fid_map = coef[0].reshape(128, 128)             # fit at theta = (1,1,1)
        marg_mean_map = Y.mean(axis=0).reshape(128, 128)
        marg_med_map = np.median(d95, axis=0)

        accumulate_accuracy(acc, "v95_FIRE_marg_median", marg_med_map, true_map, mask)
        accumulate_accuracy(acc, "v95_FIRE_marg_mean", marg_mean_map, true_map, mask)
        accumulate_accuracy(acc, "v95_FIRE_fiducial", fid_map, true_map, mask)

        d97 = load_draws(fire_draws_path(gal, "v97", args.v97_fire_draws), "v97")
        accumulate_accuracy(acc, "v97_FIRE_median", np.median(d97, axis=0), true_map, mask)
        var97 = d97.var(axis=0, ddof=1)

        for gname, bins in GROUPS.items():
            sel = group_pixels(mask, bins)
            scatter_rows.append(
                {
                    "galaxy": gal,
                    "radial_group": gname,
                    "sigma95_total": np.sqrt(np.median(var_total.reshape(128, 128)[sel])),
                    "sigma95_stoch": np.sqrt(np.median(var_resid.reshape(128, 128)[sel])),
                    "sigma95_theta": np.sqrt(np.median(var_theta.reshape(128, 128)[sel])),
                    "sigma97_total": np.sqrt(np.median(var97[sel])),
                }
            )
            shift_rows.append(
                {
                    "galaxy": gal,
                    "radial_group": gname,
                    "jensen_mean_minus_fid": np.median((marg_mean_map - fid_map)[sel]),
                    "jensen_median_minus_fid": np.median((marg_med_map - fid_map)[sel]),
                }
            )

    # -------------- DREAMS in-domain reference (test split, fixed true theta) --------------
    dreams_var = {tag: {g: [] for g in GROUPS} for tag in ("v95", "v97")}
    for tag in ("v95", "v97"):
        for gal in list(DREAMS_TEST)[: args.n_dreams]:
            d = load_draws(dreams_draws_path(gal, tag, 100), tag)
            v = d.var(axis=0, ddof=1)
            accumulate_accuracy(
                acc, f"{tag}_DREAMStest_median", np.median(d, axis=0),
                dreams_true[gal].astype(np.float64), mask,
            )
            for gname, bins in GROUPS.items():
                dreams_var[tag][gname].append(np.median(v[group_pixels(mask, bins)]))

    acc_df = finalize_accuracy(acc).sort_values(["radial_group", "config"])
    scatter_df = pd.DataFrame(scatter_rows)
    shift_df = pd.DataFrame(shift_rows)

    scat = scatter_df.groupby("radial_group").median(numeric_only=True).drop(columns="galaxy")
    scat["theta_frac_of_var"] = scat["sigma95_theta"] ** 2 / scat["sigma95_total"] ** 2
    scat["inflation_v95"] = scat["sigma95_total"] / scat["sigma95_stoch"]
    for tag in ("v95", "v97"):
        scat[f"sigma_{tag}_DREAMS"] = [
            np.median(dreams_var[tag][g]) ** 0.5 for g in scat.index
        ]
    shift = shift_df.groupby("radial_group").median(numeric_only=True).drop(columns="galaxy")

    order = ["inner", "mid", "outer", "all"]
    acc_df.to_csv(out_dir / "fire_accuracy.csv", index=False)
    scat.loc[order].to_csv(out_dir / "fire_scatter_decomposition.csv")
    shift.loc[order].to_csv(out_dir / "fire_prior_mean_shift.csv")
    scatter_df.to_csv(out_dir / "fire_scatter_per_galaxy.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n=== Point-prediction accuracy (per-pixel, pooled over galaxies) ===")
    print(acc_df.set_index(["radial_group", "config"]).loc[order].round(4).to_string())
    print("\n=== v95 FIRE predictive-scatter decomposition (dex, median over pixels/galaxies) ===")
    print(scat.loc[order].round(4).to_string())
    print("\n=== Prior-marginalization shift of the central prediction (dex) ===")
    print(shift.loc[order].round(4).to_string())
    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
