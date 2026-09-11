#!/usr/bin/env python3
"""
Azimuthal anisotropy metric for the DREAMS MW-zoom BaryonBridge runs.

The two metrics already in the paper are azimuthally symmetric by construction:
Delta_{g,b} is the m=0 moment of the density field in each annulus, and the pixel-level
Delta_i pools over every pixel.  Neither can distinguish a model that has learned a very
good CONDITIONAL 1D profile from one that has also learned where the mass sits at fixed
radius.  This module measures the second thing.

The operator, applied identically to the truth map and to every draw, in LINEAR surface
density Sigma (the .npy and the denormalized draws both hold log10(Sigma + 1)):

  c_m(b) = SUM_{p in b} Sigma_p exp(i m theta_p) / SUM_{p in b} Sigma_p
  A_m(b) = |c_m(b)|                dimensionless elongation amplitude
  psi_m(b) = arg(c_m(b)) / m       major-axis position angle

The sum runs over pixels at their exact theta_p -- there is no azimuthal binning, so
nothing aliases into m >= 4 the way an 8-bin histogram would.  m=2 is the standard
bar-strength / elongation amplitude A_2/A_0, the same quantity
training_prep/bar_strength_analysis.py:88-91 measures from particles.

Two readouts, each deliberately parallel to a metric already in the draft:

  Delta^{A2}_{g,b} = ( mean_j A_2^(j)(b) - A_2^true(b) ) / std_j[ A_2^(j)(b) ]

      Mirrors Delta_{g,b} exactly: ensemble mean in the numerator, draw-to-draw spread of
      the same statistic in the denominator.

  alignment(g)     = (1/N_b) SUM_b Re[ cbar_2 conj(c_2^true) ] / (|cbar_2| |c_2^true|)
                   = (1/N_b) SUM_b cos( 2 [psibar_2(b) - psi_2^true(b)] )
      where cbar_2(b) = mean_j c_2^(j)(b)

      +1 aligned, 0 random, -1 perpendicular.  Its expectation is EXACTLY 0 for any
      predictor carrying no azimuthal information, which is the whole point: no amount of
      skill at the 1D profile can produce a nonzero value.  Formed from the complex
      moments rather than from angles, so there is no phase unwrapping anywhere.

Radial range.  Only annuli fully inscribed in the square field of view are used, and the
innermost ones are dropped:

  edges[b] >= --r-min-kpc (2 kpc)  and  edges[b+1] <= --r-max-kpc (32 kpc)
  -> bins 2..21, twenty annuli spanning 2.83 - 31.11 kpc, 126 to 1078 pixels each.

Beyond 32 kpc an annulus samples only the image corners at theta = +/-45, +/-135 deg,
which is a pure geometric m=4 that leaks into m=2; this is the same cut the paper's
Figure 2 already shades.  Inside ~2 kpc the annuli hold too few pixels for theta to mean
anything, and 2 kpc is the model's own radial_loss_r_min_kpc.

Geometric leakage.  Truth and draws live on the same pixel grid and the same annulus
masks, so a quadrupole of the discretized mask would be a COMMON offset to c_2^true and
c_2^pred and could fake an alignment out of nothing.  It cannot, and the reason is exact
rather than empirical: the mask is invariant under a 90 degree rotation, and pairing each
pixel with its image under that rotation gives

  SUM over the 4 rotations of exp(i m [theta + k pi/2])  =  exp(i m theta) SUM_k i^(m k)

which vanishes unless m is a multiple of 4.  So m = 1, 2 and 3 are clean to machine
precision and only m = 4 is contaminated.  `selftest` confirms this numerically (measured:
A_2 = 8e-17, A_4 = 5e-2 on an exactly axisymmetric input) -- which is why m=2 is the mode
the paper reports and m=4 is carried only as a diagnostic.

`analyze` additionally writes a shuffled-pairing null (each galaxy's draws scored against
a different galaxy's truth) alongside the real numbers, as a data-driven confirmation on
the actual maps rather than on synthetic ones.

Subcommands
  selftest    synthetic-map validation of the operator (no sim data, seconds)
  crosscheck  map-based A_2 vs the particle-based peak_bar_strength_DM, 1024 galaxies
  analyze     the full metric over a set of galaxies

  python analyze_anisotropy.py selftest
  python analyze_anisotropy.py crosscheck
  python analyze_anisotropy.py analyze --version v95 --num-draws 100 \
      --draws-suffix heun100_n100 --galaxy-indices 0:102 --output-dir anisotropy_v95
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import stars2dm_paths
from pred_data_loaders import (
    DEFAULT_PRED_ROOT,
    build_radial_annulus_geometry,
    denormalize_draws,
    discover_galaxy_indices,
    extract_draws_from_npz,
    parse_version_tag,
    resolve_draws_filepath,
    split_label,
)

# The DREAMS MW-zoom truth stack, (1024, 128, 128) float64 holding log10(Sigma_DM + 1).
# Same literal paths as plot_radial_profile_figures.MAP_PATHS[*]["rhoDM"]; copied rather
# than imported so this module does not pull in that 3500-line script.  All three suites
# are imaged on the same fixed +/-32 kpc grid ("csv_fov_z0p350_0p350"), which is what lets
# one annulus geometry serve all of them.
TRUTH_MAP_PATHS = {
    "DREAMS": stars2dm_paths.DREAMS_RHODM_MAP,
    "TNG50": stars2dm_paths.TNG_RHODM_MAP,
    "FIREm12": stars2dm_paths.FIRE_RHODM_MAP,
}
TRUTH_MAP_PATH = TRUTH_MAP_PATHS["DREAMS"]


def truth_map_path(version_tag):
    """The truth stack for whichever suite this version tag names."""
    sim_label, _ = parse_version_tag(version_tag)
    return TRUTH_MAP_PATHS[sim_label or "DREAMS"]

# Particle-based bar strength for the same 1024 galaxies, measured in 3D from the
# snapshots by training_prep/bar_strength_analysis.py.  Independent of the maps, so it is
# the right thing to validate the map operator against.
BAR_STRENGTH_CSV = stars2dm_paths.BAR_STRENGTH_CSV
# bar_strength_analysis.py:233 bins particles over Rbins = np.arange(0, 10, ann_size), so
# its peak is a peak over r < 10 kpc.  The map-based maximum has to be taken over the same
# range or the crosscheck compares two different radial windows.
CROSSCHECK_R_MAX_KPC = 10.0

FOV_HALF_KPC = 32.0
DEFAULT_R_MIN_KPC = 2.0
DEFAULT_R_MAX_KPC = 32.0
DEFAULT_RESOLUTION = 128

# Modes computed.  m=2 is the one that goes in the paper; m=1 (lopsidedness), 3 and 4 are
# free once the moments exist and are carried in the CSVs as secondaries.
M_VALUES = (1, 2, 3, 4)
PAPER_M = 2

NULL_SEED = 42
BOOTSTRAP_N = 500



# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------

def annulus_fourier_geometry(
    resolution=DEFAULT_RESOLUTION,
    r_min_kpc=DEFAULT_R_MIN_KPC,
    r_max_kpc=DEFAULT_R_MAX_KPC,
    fov_half_kpc=FOV_HALF_KPC,
    m_values=M_VALUES,
):
    """
    Flat pixel indices and precomputed exp(i m theta) weights per kept annulus.

    Geometry only, so it is built once and reused for every galaxy and every draw.  The
    annuli come from build_radial_annulus_geometry so the radial binning is bit-identical
    to the one the radial-profile metric uses; this only selects a subset of them and
    attaches the azimuthal phases.
    """
    radial_edges, radial_centers, mask = build_radial_annulus_geometry(
        resolution, -fov_half_kpc, fov_half_kpc, -fov_half_kpc, fov_half_kpc
    )
    keep = np.flatnonzero(
        (radial_edges[:-1] >= r_min_kpc) & (radial_edges[1:] <= r_max_kpc)
    )
    if keep.size == 0:
        raise ValueError(
            f"No annuli survive r_min={r_min_kpc} r_max={r_max_kpc} at "
            f"resolution {resolution}."
        )

    # Same pixel-centre grid and same arctan2(y, x) convention as
    # plot_radial_profile_figures.compute_deviation_summary:2550-2556, so theta=0 lies
    # along +x and increases towards +y -- the convention the Figure 2 caption states.
    edges = np.linspace(-fov_half_kpc, fov_half_kpc, resolution + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_coords, y_coords = np.meshgrid(centers, centers, indexing="ij")
    theta_flat = np.arctan2(y_coords, x_coords).ravel()

    m_array = np.asarray(m_values, dtype=np.float64)
    pixel_idx = []
    phase_weights = []
    for b in keep:
        idx = np.flatnonzero(mask[b].ravel())
        pixel_idx.append(idx)
        # (n_m, n_pix); conj taken at use time via the matmul below.
        phase_weights.append(np.exp(1j * m_array[:, None] * theta_flat[idx][None, :]))

    return {
        "resolution": resolution,
        "bin_indices": keep,
        "radial_edges": radial_edges,
        "radial_centers": radial_centers[keep],
        "pixel_idx": pixel_idx,
        "phase_weights": phase_weights,
        "pixel_counts": np.array([idx.size for idx in pixel_idx]),
        "m_values": np.asarray(m_values, dtype=int),
    }


def fourier_moments(sigma, geometry):
    """
    Normalized azimuthal Fourier moments c_m.

    sigma: (..., resolution, resolution) in LINEAR surface density.
    Returns complex (..., n_bins, n_m).

    Vectorized over the leading axes, so all 100 draws of a galaxy go through in one pass
    per annulus rather than 100 python-level iterations.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    resolution = geometry["resolution"]
    lead_shape = sigma.shape[:-2]
    flat = sigma.reshape(-1, resolution * resolution)

    n_maps = flat.shape[0]
    n_bins = len(geometry["pixel_idx"])
    n_m = geometry["m_values"].size
    out = np.empty((n_maps, n_bins, n_m), dtype=np.complex128)

    for b, (idx, weights) in enumerate(
        zip(geometry["pixel_idx"], geometry["phase_weights"])
    ):
        vals = flat[:, idx]
        total = vals.sum(axis=1)
        # The DM surface density has a measured floor of 10**4.51 Msun/kpc^2 and no empty
        # pixels, so an annulus total of zero means the map is not what this module thinks
        # it is.  Fail rather than emit a silent nan.
        if not np.all(total > 0):
            raise ValueError(
                f"Non-positive annulus total in bin {b} "
                f"(radius {geometry['radial_centers'][b]:.2f} kpc): the map is not a "
                "positive surface density."
            )
        out[:, b, :] = (vals @ weights.T) / total[:, None]

    return out.reshape(lead_shape + (n_bins, n_m))


def linear_sigma(log_map):
    """log10(Sigma + 1) -> Sigma, the inverse of training_prep/get_2D_maps.py:730."""
    return 10.0 ** np.asarray(log_map, dtype=np.float64) - 1.0


def alignment_terms(c_pred_mean, c_true):
    """
    Per-annulus cos(m * dpsi) and the wrapped angle offset, from the complex moments.

    cos(m dpsi) = Re[cbar conj(c_true)] / (|cbar| |c_true|) and
    dpsi = arg(cbar conj(c_true)) / m.  Both are exact for any m and need no unwrapping.
    """
    cross = c_pred_mean * np.conj(c_true)
    denom = np.abs(c_pred_mean) * np.abs(c_true)
    if not np.all(denom > 0):
        raise ValueError("Zero Fourier amplitude: cannot define a position angle.")
    return np.real(cross) / denom, np.angle(cross), cross


def circular_spread_deg(c_draws_m, paper_m=PAPER_M):
    """
    Draw-to-draw scatter of the major-axis angle, in degrees.

    The angle lives on a circle, so an ordinary standard deviation is meaningless near
    the wrap. This is the circular version: R = |mean of the unit phasors|, spread =
    sqrt(-2 ln R) in the m*psi angle, divided by m to convert back to psi.
    """
    unit = c_draws_m / np.abs(c_draws_m)
    resultant = np.clip(np.abs(unit.mean(axis=0)), 1e-12, 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(resultant))) / paper_m


def galaxy_statistics(c_true, c_draws, m_values=M_VALUES, paper_m=PAPER_M):
    """
    Every per-annulus and scalar statistic for one galaxy.

    c_true:  (n_bins, n_m) complex, from the truth map.
    c_draws: (n_draws, n_bins, n_m) complex, one entry per stochastic draw.
    """
    m_values = np.asarray(m_values, dtype=int)
    mi = int(np.flatnonzero(m_values == paper_m)[0])

    amp_true = np.abs(c_true)                      # (n_bins, n_m)
    amp_draws = np.abs(c_draws)                    # (n_draws, n_bins, n_m)
    amp_pred_mean = amp_draws.mean(axis=0)
    # ddof=0, matching np.std(draws, axis=0) in compute_deviation_map and the ensemble
    # spread the paper's Delta_{g,b} denominator uses.
    amp_pred_std = amp_draws.std(axis=0)

    # The coherent moment: the ensemble mean of the complex c_m, whose modulus is <= the
    # mean modulus.  |cbar| / mean|c| is the fraction of the predicted elongation that
    # points the same way in every draw -- the same coherent/incoherent split
    # profile_calibration_audit.mode_decomposition applies to the pixel residuals.
    c_pred_mean = c_draws.mean(axis=0)             # (n_bins, n_m)
    amp_coherent = np.abs(c_pred_mean)

    if not np.all(amp_pred_std > 0):
        raise ValueError(
            "Zero draw-to-draw spread in A_m: the ensemble is degenerate, so "
            "Delta^{A2} has no denominator."
        )
    delta_amp = (amp_pred_mean - amp_true) / amp_pred_std

    cos_align, dpsi, cross = alignment_terms(
        c_pred_mean[:, mi], c_true[:, mi]
    )
    psi_true = np.angle(c_true[:, mi]) / paper_m
    psi_pred = np.angle(c_pred_mean[:, mi]) / paper_m

    # Amplitude-weighted alignment: the same cosine with weight |cbar||c_true|, i.e. the
    # normalized real part of the summed cross moment.  Annuli that are nearly
    # axisymmetric carry an essentially random angle and should not count equally.
    weights = np.abs(c_pred_mean[:, mi]) * np.abs(c_true[:, mi])
    alignment_weighted = float(np.sum(np.real(cross)) / np.sum(weights))

    return {
        "amp_true": amp_true,
        "amp_pred_mean": amp_pred_mean,
        "amp_pred_std": amp_pred_std,
        "amp_coherent": amp_coherent,
        "delta_amp": delta_amp,
        "cos_align": cos_align,
        "dpsi_deg": np.degrees(dpsi) / paper_m,
        "psi_true_deg": np.degrees(psi_true),
        "psi_pred_deg": np.degrees(psi_pred),
        "c_true_m": c_true[:, mi],
        "c_pred_mean_m": c_pred_mean[:, mi],
        "alignment": float(np.mean(cos_align)),
        "alignment_ampweighted": alignment_weighted,
    }


def pairwise_alignment(c_pred_mean_m, c_true_m):
    """Scalar alignment between one galaxy's coherent moment and a truth moment."""
    cos_align, _, cross = alignment_terms(c_pred_mean_m, c_true_m)
    weights = np.abs(c_pred_mean_m) * np.abs(c_true_m)
    return float(np.mean(cos_align)), float(np.sum(np.real(cross)) / np.sum(weights))


def derangement(n, seed=NULL_SEED):
    """
    A random permutation with no fixed point.

    Shuffling and rolling the shuffled order by one guarantees partner[i] != i, which a
    bare rng.permutation does not.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    partner = np.empty(n, dtype=int)
    partner[order] = np.roll(order, 1)
    return partner


def bootstrap_median_ci(values, n_boot=BOOTSTRAP_N, alpha=0.05, seed=NULL_SEED):
    """Median and its percentile bootstrap CI. Same estimator as analyze_central_bias."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.sort(
        [np.median(rng.choice(values, size=values.size, replace=True))
         for _ in range(n_boot)]
    )
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return float(np.median(values)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Synthetic maps, for the selftest
# ---------------------------------------------------------------------------

def pixel_grid(resolution=DEFAULT_RESOLUTION, fov_half_kpc=FOV_HALF_KPC):
    """Pixel-centre x and y in kpc, matching annulus_fourier_geometry's grid."""
    edges = np.linspace(-fov_half_kpc, fov_half_kpc, resolution + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.meshgrid(centers, centers, indexing="ij")


def elliptical_profile(
    axis_ratio=1.0,
    position_angle_deg=0.0,
    core_kpc=5.0,
    amplitude=1.0e7,
    resolution=DEFAULT_RESOLUTION,
    fov_half_kpc=FOV_HALF_KPC,
):
    """
    A cored profile whose isodensity contours are ellipses of axis ratio `axis_ratio`
    with the major axis at `position_angle_deg` from +x.

    axis_ratio=1 gives an exactly axisymmetric map, which is what the geometric-leakage
    floor is measured on.
    """
    x_coords, y_coords = pixel_grid(resolution, fov_half_kpc)
    angle = np.radians(position_angle_deg)
    x_major = x_coords * np.cos(angle) + y_coords * np.sin(angle)
    y_minor = -x_coords * np.sin(angle) + y_coords * np.cos(angle)
    ellipse_r = np.sqrt(x_major ** 2 + (y_minor / axis_ratio) ** 2)
    return amplitude / (1.0 + (ellipse_r / core_kpc) ** 2)


def wrap_angle_deg(delta_deg, period_deg=180.0):
    """Wrap an angle difference into (-period/2, +period/2]."""
    half = period_deg / 2.0
    return (np.asarray(delta_deg) + half) % period_deg - half


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_selftest(args):
    """
    Validate the operator on synthetic maps.

    Four claims:
      1. an exactly axisymmetric map gives A_2 at the pixelization floor (this IS the
         geometric-leakage number the analysis rests on),
      2. the recovered major-axis angle tracks an injected one,
      3. A_2 grows monotonically as the injected axis ratio shrinks,
      4. mismatched galaxies show no alignment, so the common pixel grid is not itself
         manufacturing one.
    """
    geometry = annulus_fourier_geometry(
        args.resolution, args.r_min_kpc, args.r_max_kpc, m_values=M_VALUES
    )
    mi = int(np.flatnonzero(geometry["m_values"] == PAPER_M)[0])
    n_bins = len(geometry["pixel_idx"])
    print(f"geometry: {n_bins} annuli, "
          f"r = {geometry['radial_edges'][geometry['bin_indices'][0]]:.2f} - "
          f"{geometry['radial_edges'][geometry['bin_indices'][-1] + 1]:.2f} kpc, "
          f"{geometry['pixel_counts'].min()} - {geometry['pixel_counts'].max()} pixels "
          "per annulus")

    # --- 1. geometric-leakage floor -----------------------------------------------
    axisymmetric = elliptical_profile(
        axis_ratio=1.0, resolution=args.resolution, fov_half_kpc=FOV_HALF_KPC
    )
    c_axi = fourier_moments(axisymmetric, geometry)
    leakage = np.abs(c_axi[:, mi])
    print("\naxisymmetric input (geometric leakage floor):")
    print(f"  max A_2 over annuli   {leakage.max():.3e}   (expect < 1e-3)")
    print(f"  max A_1 over annuli   {np.abs(c_axi[:, 0]).max():.3e}")
    print(f"  max A_4 over annuli   {np.abs(c_axi[:, 3]).max():.3e}")
    ok_floor = bool(leakage.max() < 1e-3)

    # --- 2. position-angle recovery -----------------------------------------------
    print("\ninjected position angle vs recovered (amplitude-weighted over annuli):")
    angle_errors = []
    for position_angle in (0.0, 30.0, 60.0, 120.0):
        elongated = elliptical_profile(
            axis_ratio=0.7,
            position_angle_deg=position_angle,
            resolution=args.resolution,
        )
        c_ell = fourier_moments(elongated, geometry)[:, mi]
        # One angle for the whole map: the argument of the amplitude-weighted mean
        # moment, which is how the analysis combines annuli.
        recovered = np.degrees(np.angle(c_ell.sum())) / PAPER_M
        error = float(np.abs(wrap_angle_deg(recovered - position_angle)))
        angle_errors.append(error)
        print(f"  psi_0 = {position_angle:6.1f} deg   recovered {recovered:7.2f} deg"
              f"   error {error:.3f} deg")
    ok_angle = bool(max(angle_errors) < 2.0)

    # --- 3. amplitude monotonic in axis ratio -------------------------------------
    print("\nmean A_2 over annuli vs injected axis ratio:")
    amplitudes = []
    for axis_ratio in (0.9, 0.7, 0.5):
        elongated = elliptical_profile(
            axis_ratio=axis_ratio, position_angle_deg=25.0, resolution=args.resolution
        )
        mean_amp = float(np.mean(np.abs(fourier_moments(elongated, geometry)[:, mi])))
        amplitudes.append(mean_amp)
        print(f"  q = {axis_ratio:.1f}   A_2 = {mean_amp:.4f}")
    ok_monotonic = bool(amplitudes[0] < amplitudes[1] < amplitudes[2])

    # --- 4. mismatched-pair null --------------------------------------------------
    # Score each synthetic galaxy's OWN truth moment against a different galaxy's. A
    # perfect predictor paired with the wrong galaxy is the strongest possible version of
    # this test: anything left is the shared grid, not the model.
    rng = np.random.default_rng(NULL_SEED)
    n_synth = args.n_synthetic
    moments = []
    for _ in range(n_synth):
        synthetic = elliptical_profile(
            axis_ratio=float(rng.uniform(0.5, 0.95)),
            position_angle_deg=float(rng.uniform(0.0, 180.0)),
            resolution=args.resolution,
        )
        moments.append(fourier_moments(synthetic, geometry)[:, mi])
    moments = np.array(moments)
    partner = derangement(n_synth)
    self_scores = np.array(
        [pairwise_alignment(moments[i], moments[i])[0] for i in range(n_synth)]
    )
    null_scores = np.array(
        [pairwise_alignment(moments[i], moments[partner[i]])[0] for i in range(n_synth)]
    )
    # The null mean is itself a Monte Carlo estimate: for uniformly random relative
    # angles cos(2 dpsi) has mean 0 and variance 1/2, so the sample mean scatters by
    # ~0.7/sqrt(n) whatever the operator does. Test it against its OWN standard error,
    # not against a fixed constant that only happens to hold at one sample size.
    null_mean = float(null_scores.mean())
    null_sem = float(null_scores.std(ddof=1) / np.sqrt(n_synth))
    print(f"\nmismatched-pair null over {n_synth} synthetic galaxies:")
    print(f"  self-paired alignment     {self_scores.mean():+.4f}   (expect +1)")
    print(f"  mismatched alignment      {null_mean:+.4f} +/- {null_sem:.4f} (SEM)"
          f"   -> {abs(null_mean) / null_sem:.2f} sigma from zero")
    ok_null = bool(abs(null_mean) < 3.0 * null_sem and self_scores.mean() > 0.999)

    checks = {
        "axisymmetric map has A_2 at the pixelization floor": ok_floor,
        "recovered position angle tracks the injected one": ok_angle,
        "A_2 increases monotonically as the axis ratio shrinks": ok_monotonic,
        "mismatched galaxies show no alignment": ok_null,
    }
    print()
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


def cmd_crosscheck(args):
    """
    Map-based A_2 vs the particle-based peak_bar_strength_DM for the same 1024 galaxies.

    The particle measurement never touches the maps -- different code, different data
    product, 3D cylindrical annuli rather than a projected image -- so a strong rank
    correlation is evidence the map operator measures the physical quantity it claims to.
    They will not agree in absolute value: projection along z dilutes the amplitude.
    """
    truth_stack = np.load(TRUTH_MAP_PATH, mmap_mode="r")
    print(f"truth stack {TRUTH_MAP_PATH.name}: shape {truth_stack.shape}")

    geometry = annulus_fourier_geometry(
        args.resolution, args.r_min_kpc, CROSSCHECK_R_MAX_KPC, m_values=M_VALUES
    )
    mi = int(np.flatnonzero(geometry["m_values"] == PAPER_M)[0])
    print(f"crosscheck window: {len(geometry['pixel_idx'])} annuli over "
          f"r = {args.r_min_kpc:.2f} - {CROSSCHECK_R_MAX_KPC:.2f} kpc, matching "
          "bar_strength_analysis.py's Rbins = arange(0, 10)")

    n_galaxies = truth_stack.shape[0]
    peak_map = np.empty(n_galaxies)
    block = 128
    for start in range(0, n_galaxies, block):
        stop = min(start + block, n_galaxies)
        sigma = linear_sigma(truth_stack[start:stop])
        peak_map[start:stop] = np.abs(fourier_moments(sigma, geometry)[:, :, mi]).max(axis=1)

    frame = pd.read_csv(BAR_STRENGTH_CSV)
    merged = pd.DataFrame({
        "galaxy_id": np.arange(n_galaxies),
        "A2_map_peak": peak_map,
    }).merge(
        frame[["galaxy_id", "peak_bar_strength_DM", "peak_bar_strength_star"]],
        on="galaxy_id",
        how="inner",
    )
    valid = merged.dropna(subset=["A2_map_peak", "peak_bar_strength_DM"])
    rho_dm, p_dm = spearmanr(valid["A2_map_peak"], valid["peak_bar_strength_DM"])
    rho_star, p_star = spearmanr(valid["A2_map_peak"], valid["peak_bar_strength_star"])

    print(f"\n{len(valid)} galaxies matched")
    print(f"  map A_2 peak: median {valid['A2_map_peak'].median():.4f}, "
          f"5-95 pct {valid['A2_map_peak'].quantile(0.05):.4f} - "
          f"{valid['A2_map_peak'].quantile(0.95):.4f}")
    print(f"  particle DM peak: median {valid['peak_bar_strength_DM'].median():.4f}")
    print(f"\n  Spearman rho (map A_2 vs particle DM  ) = {rho_dm:+.3f}  (p = {p_dm:.2e})")
    print(f"  Spearman rho (map A_2 vs particle star) = {rho_star:+.3f}  (p = {p_star:.2e})")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "anisotropy_crosscheck.csv"
        merged.to_csv(out_path, index=False)
        print(f"\nwrote {out_path}")

    if rho_dm < 0.3:
        print("\nFAIL: the map-based A_2 does not track the independent particle-based")
        print("      bar strength. Check the theta convention and the linear-Sigma")
        print("      conversion before reading any result.")
        return 1
    print("\nPASS: map-based A_2 tracks the particle-based bar strength.")
    return 0


def resolve_galaxy_indices(args, pred_root):
    if args.galaxy_indices:
        if ":" in args.galaxy_indices:
            lo, hi = args.galaxy_indices.split(":")
            return list(range(int(lo), int(hi)))
        return [int(v) for v in args.galaxy_indices.split(",")]
    found = discover_galaxy_indices(
        pred_root, args.version, num_draws=args.num_draws,
        output_suffix=args.draws_suffix, file_pattern=args.draws_pattern,
    )
    return sorted(found)


def load_draws_linear(pred_root, galaxy_idx, version_tag, num_draws, suffix,
                      file_pattern=None):
    """One galaxy's draws: standardized -> log10(Sigma+1) -> linear Sigma."""
    path = resolve_draws_filepath(pred_root, galaxy_idx, version_tag, num_draws, suffix,
                                  file_pattern)
    # resolve_draws_filepath returns its first candidate when nothing matched, so the
    # caller has to test existence rather than trust a returned path.
    if path is None or not Path(path).exists():
        return None
    with np.load(path) as data:
        # slice(None), not None: extract_draws_from_npz indexes draws[draw_slice], and
        # numpy reads a literal None as np.newaxis.
        draws = extract_draws_from_npz(data, draw_slice=slice(None), channel=0)
    log_draws = denormalize_draws(
        np.asarray(draws, dtype=np.float64),
        resolution=draws.shape[-1],
        version_tag=version_tag,
    )
    return linear_sigma(log_draws)


def cmd_analyze(args):
    pred_root = Path(args.pred_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default the scored split to the one the suite actually has: "test" in domain,
    # "ood" for TNG50/FIREm12, which were never trained on.
    if args.population_split is None:
        sim_label, _ = parse_version_tag(args.version)
        args.population_split = "ood" if sim_label else "test"

    geometry = annulus_fourier_geometry(
        args.resolution, args.r_min_kpc, args.r_max_kpc, m_values=M_VALUES
    )
    m_values = geometry["m_values"]
    mi = int(np.flatnonzero(m_values == PAPER_M)[0])
    radial_centers = geometry["radial_centers"]
    bin_indices = geometry["bin_indices"]
    n_bins = len(radial_centers)
    print(f"version {args.version}: {n_bins} annuli over "
          f"r = {geometry['radial_edges'][bin_indices[0]]:.2f} - "
          f"{geometry['radial_edges'][bin_indices[-1] + 1]:.2f} kpc, "
          f"modes m = {[int(v) for v in m_values]}, scored on m = {PAPER_M}")

    map_path = truth_map_path(args.version)
    truth_stack = np.load(map_path, mmap_mode="r")
    print(f"truth stack {map_path.name}: {truth_stack.shape[0]} galaxies")
    galaxy_indices = resolve_galaxy_indices(args, pred_root)
    # An index past the end of the truth stack has no target to score against. TNG50 ships
    # 280 prediction directories against 264 maps, so this is a real case, not a guard.
    n_maps = truth_stack.shape[0]
    dropped = [g for g in galaxy_indices if g >= n_maps]
    if dropped:
        print(f"skipping {len(dropped)} galaxies with no truth map "
              f"(indices >= {n_maps}): {dropped[:6]}{' ...' if len(dropped) > 6 else ''}")
        galaxy_indices = [g for g in galaxy_indices if g < n_maps]
    featured = [int(v) for v in args.per_galaxy_indices.split(",")] if args.per_galaxy_indices else []
    print(f"{len(galaxy_indices)} galaxies: {galaxy_indices[:8]}"
          f"{' ...' if len(galaxy_indices) > 8 else ''}")

    radial_rows = []
    scalar_rows = []
    kept_ids = []
    c_true_all = []
    c_pred_all = []
    curves = {}

    for galaxy_idx in galaxy_indices:
        draws = load_draws_linear(
            pred_root, galaxy_idx, args.version, args.num_draws, args.draws_suffix,
            args.draws_pattern,
        )
        if draws is None:
            print(f"galaxy {galaxy_idx}: no draws found, skipped")
            continue

        sigma_true = linear_sigma(truth_stack[galaxy_idx])
        c_true = fourier_moments(sigma_true, geometry)
        c_draws = fourier_moments(draws, geometry)
        stats = galaxy_statistics(c_true, c_draws, m_values=m_values, paper_m=PAPER_M)

        split = split_label(galaxy_idx, args.version)
        kept_ids.append(galaxy_idx)
        c_true_all.append(stats["c_true_m"])
        c_pred_all.append(stats["c_pred_mean_m"])

        for j in range(n_bins):
            row = {
                "galaxy_id": galaxy_idx,
                "split": split,
                "radial_bin": int(bin_indices[j]),
                "radius_kpc": float(radial_centers[j]),
                "n_pixels": int(geometry["pixel_counts"][j]),
                "A2_true": float(stats["amp_true"][j, mi]),
                "A2_pred_mean": float(stats["amp_pred_mean"][j, mi]),
                "A2_pred_std": float(stats["amp_pred_std"][j, mi]),
                "A2_coherent": float(stats["amp_coherent"][j, mi]),
                "delta_A2": float(stats["delta_amp"][j, mi]),
                "psi2_true_deg": float(stats["psi_true_deg"][j]),
                "psi2_pred_deg": float(stats["psi_pred_deg"][j]),
                "dpsi2_deg": float(stats["dpsi_deg"][j]),
                "cos2dpsi": float(stats["cos_align"][j]),
            }
            # Secondary modes, free once the moments exist. m=1 is lopsidedness.
            for k, m in enumerate(m_values):
                if m == PAPER_M:
                    continue
                row[f"A{m}_true"] = float(stats["amp_true"][j, k])
                row[f"A{m}_pred_mean"] = float(stats["amp_pred_mean"][j, k])
                row[f"delta_A{m}"] = float(stats["delta_amp"][j, k])
            radial_rows.append(row)

        scalar_rows.append({
            "galaxy_id": galaxy_idx,
            "split": split,
            "alignment": stats["alignment"],
            "alignment_ampweighted": stats["alignment_ampweighted"],
            "mean_delta_A2": float(np.mean(stats["delta_amp"][:, mi])),
            "std_delta_A2": float(np.std(stats["delta_amp"][:, mi])),
            "A2_true_max": float(stats["amp_true"][:, mi].max()),
            "A2_true_mean": float(stats["amp_true"][:, mi].mean()),
            "A2_pred_max": float(stats["amp_pred_mean"][:, mi].max()),
            "A2_pred_mean": float(stats["amp_pred_mean"][:, mi].mean()),
            # <= 1 by Jensen; the fraction of the predicted elongation that points the
            # same way in every draw.
            "coherent_fraction": float(
                np.mean(stats["amp_coherent"][:, mi] / stats["amp_pred_mean"][:, mi])
            ),
        })

        if galaxy_idx in featured:
            curves[f"galaxy_{galaxy_idx}_c_draws"] = c_draws.astype(np.complex64)
            curves[f"galaxy_{galaxy_idx}_c_true"] = c_true.astype(np.complex128)

        print(f"galaxy {galaxy_idx} ({split}): alignment {stats['alignment']:+.3f}, "
              f"A_2 true {stats['amp_true'][:, mi].mean():.4f} vs pred "
              f"{stats['amp_pred_mean'][:, mi].mean():.4f}, "
              f"mean Delta^A2 {np.mean(stats['delta_amp'][:, mi]):+.2f}")

    if not scalar_rows:
        print("No galaxies produced results.")
        return 1

    radial_df = pd.DataFrame(radial_rows)
    scalar_df = pd.DataFrame(scalar_rows)
    # The version tag already carries the sim label for OOD suites (TNG50_v95), so the
    # output filenames separate the suites without any extra token.
    tag = args.version if not args.draws_suffix else f"{args.version}_{args.draws_suffix}"
    radial_df.to_csv(output_dir / f"anisotropy_radial_{tag}.csv", index=False)
    scalar_df.to_csv(output_dir / f"anisotropy_scalars_{tag}.csv", index=False)

    # --- shuffled-pairing null ---------------------------------------------------
    # Each galaxy's coherent predicted moment scored against a DIFFERENT galaxy's truth.
    # Nonzero here means the shared pixel grid and annulus masks are manufacturing an
    # alignment, and the real number has to be read against it.
    c_true_all = np.array(c_true_all)
    c_pred_all = np.array(c_pred_all)
    partner = derangement(len(kept_ids))
    null_rows = []
    for i, galaxy_idx in enumerate(kept_ids):
        plain, weighted = pairwise_alignment(c_pred_all[i], c_true_all[partner[i]])
        null_rows.append({
            "galaxy_id": galaxy_idx,
            "partner_galaxy_id": kept_ids[partner[i]],
            "alignment_null": plain,
            "alignment_ampweighted_null": weighted,
        })
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(output_dir / f"anisotropy_null_{tag}.csv", index=False)

    # --- population summary at fixed radius ---------------------------------------
    population = radial_df[radial_df["split"] == args.population_split]
    if population.empty:
        print(f"No galaxies in split '{args.population_split}'; population CSV skipped.")
        population_df = pd.DataFrame()
    else:
        pop_rows = []
        for radial_bin, group in population.groupby("radial_bin"):
            cos_median, cos_lo, cos_hi = bootstrap_median_ci(group["cos2dpsi"].values)
            delta_median, delta_lo, delta_hi = bootstrap_median_ci(group["delta_A2"].values)
            pop_rows.append({
                "radial_bin": int(radial_bin),
                "radius_kpc": float(group["radius_kpc"].iloc[0]),
                "n_galaxies": int(len(group)),
                "delta_A2_mean": float(group["delta_A2"].mean()),
                "delta_A2_std": float(group["delta_A2"].std(ddof=0)),
                "delta_A2_median": delta_median,
                "delta_A2_ci_lo": delta_lo,
                "delta_A2_ci_hi": delta_hi,
                "cos2dpsi_mean": float(group["cos2dpsi"].mean()),
                "cos2dpsi_std": float(group["cos2dpsi"].std(ddof=0)),
                "cos2dpsi_sem": float(
                    group["cos2dpsi"].std(ddof=0) / np.sqrt(len(group))
                ),
                "cos2dpsi_median": cos_median,
                "cos2dpsi_ci_lo": cos_lo,
                "cos2dpsi_ci_hi": cos_hi,
                "A2_true_median": float(group["A2_true"].median()),
                "A2_pred_median": float(group["A2_pred_mean"].median()),
            })
        population_df = pd.DataFrame(pop_rows)
        population_df.to_csv(output_dir / f"anisotropy_population_{tag}.csv", index=False)

    if curves:
        curves["radial_centers"] = radial_centers
        curves["radial_bin_indices"] = bin_indices
        curves["m_values"] = m_values
        curves["featured_galaxy_ids"] = np.array(sorted(curves_ids(curves)))
        np.savez_compressed(output_dir / f"anisotropy_curves_{tag}.npz", **curves)

    # --- summary ------------------------------------------------------------------
    scored = scalar_df[scalar_df["split"] == args.population_split]
    if scored.empty:
        scored = scalar_df
    null_scored = null_df[null_df["galaxy_id"].isin(scored["galaxy_id"])]
    print(f"\n=== {tag}: {len(scored)} galaxies in split '{args.population_split}' ===")
    print(f"alignment            median {scored['alignment'].median():+.4f}   "
          f"mean {scored['alignment'].mean():+.4f}   "
          f"frac > 0  {float((scored['alignment'] > 0).mean()):.3f}")
    print(f"alignment (amp-wtd)  median {scored['alignment_ampweighted'].median():+.4f}   "
          f"mean {scored['alignment_ampweighted'].mean():+.4f}")
    print(f"shuffled-pair NULL   median {null_scored['alignment_null'].median():+.4f}   "
          f"mean {null_scored['alignment_null'].mean():+.4f}   "
          f"(expect ~0; if not, quote the null-subtracted number)")
    print(f"Delta^A2             mean {scored['mean_delta_A2'].mean():+.3f}   "
          f"population width {scored['mean_delta_A2'].std(ddof=0):.3f}")
    print(f"coherent fraction    median {scored['coherent_fraction'].median():.3f}")
    print(f"\nwrote CSVs to {output_dir}")
    return 0


def curves_ids(curves):
    return {
        int(key.split("_")[1])
        for key in curves
        if key.startswith("galaxy_") and key.endswith("_c_true")
    }


COMMANDS = {
    "selftest": cmd_selftest,
    "crosscheck": cmd_crosscheck,
    "analyze": cmd_analyze,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--version", type=str, default="v95",
                        help="model version tag (v95, v97, ...)")
    parser.add_argument("--pred-root", type=str, default=str(DEFAULT_PRED_ROOT))
    parser.add_argument("--output-dir", type=str, default="anisotropy")
    parser.add_argument("--num-draws", type=int, default=100)
    parser.add_argument("--draws-suffix", type=str, default="heun100_n100")
    parser.add_argument("--galaxy-indices", type=str, default=None,
                        help="'start:end' or a comma list. Default: whatever is on disk.")
    parser.add_argument("--per-galaxy-indices", type=str, default="19",
                        help="Galaxies whose per-draw moments are saved for the figures.")
    parser.add_argument("--draws-pattern", type=str, default=None,
                        help=("Glob for the draws npz, for suites whose filenames the "
                              "default resolver does not build (e.g. the FIREm12 "
                              "flat-prior files)."))
    parser.add_argument("--population-split", type=str, default=None,
                        choices=["test", "val", "train", "ood"],
                        help="Default: 'test' in domain, 'ood' for TNG50/FIREm12.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--n-synthetic", type=int, default=256,
                        help="selftest only: synthetic galaxies for the null test.")
    parser.add_argument("--r-min-kpc", type=float, default=DEFAULT_R_MIN_KPC,
                        help="Drop annuli whose inner edge is inside this radius.")
    parser.add_argument("--r-max-kpc", type=float, default=DEFAULT_R_MAX_KPC,
                        help="Drop annuli not fully inscribed in the square field of view.")
    args = parser.parse_args()
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
