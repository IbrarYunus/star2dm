"""Shared loaders for per-galaxy prediction npz under DREAMS Pred tree."""

import fnmatch
import os
from pathlib import Path

import numpy as np
import stars2dm_paths
from baryonbridge.config import get_norm_profile, merged_norm_dict

NORM_DICT = {
    '2D': {
        'rhoDM': {
            128: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122},
            32: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122},
        },
    }
}

DEFAULT_PRED_ROOT = stars2dm_paths.PRED_ROOT
DEFAULT_THETA_PATH = stars2dm_paths.DREAMS_THETA_PATH

# The 820/102/102 split used by every model in this release: test [0, 102),
# val [102, 204), train [204, 1024). DREAMSDataModule.setup slices by index range, so
# these constants ARE the split -- there is no permutation file to read. Earlier runs of
# this project used 860/78/86 (TEST_IDX_END=86, TRAIN_IDX_START=164); applying those here
# would label galaxies 86-101 "val" when the model held them out as test, contaminating a
# test-set result rather than raising. If you retrain with different n_train/n_val/n_test,
# change these to match the config or every split label silently becomes wrong.
TEST_IDX_END = 102
VAL_IDX_START = 102
TRAIN_IDX_START = 204

PARAM_NAMES = ["Om", "s8", "SN1", "SN2", "BHFF"]

SWEEP_TAGS = [
    "sn1_sweep_5",
    "sn2_sweep_5",
    "bhff_sweep_5",
    "om_sweep_5",
    "s8_sweep_5",
]

OOD_SIM_LABELS = ("FIREm12", "FIREm11", "TNG100", "TNG50")


def parse_version_tag(version_tag):
    """Return (sim_label, model_version). sim_label is None for in-domain DREAMS."""
    if version_tag is None:
        return None, ""
    for sim in OOD_SIM_LABELS:
        if version_tag.startswith(sim):
            remainder = version_tag[len(sim) :].lstrip("_")
            model_version = remainder if remainder else version_tag
            return sim, model_version
    return None, version_tag


def effective_pred_root(pred_root, sim_label):
    pred_root = Path(pred_root)
    if sim_label:
        return pred_root / sim_label
    return pred_root


def rhodm_norm_constants(version_tag=None, resolution=128):
    """
    (mean, std) for denormalizing rhoDM draws, resolved per version tag.

    This module carries its own NORM_DICT whose rhoDM entry is the fixed +/-32 kpc one. The
    wide-redshift runs (csv_fov_z0p050_0p650) were trained and sampled against a DIFFERENT
    rhoDM normalization, so denormalizing their draws with the fixed-FOV constants shifts
    every predicted profile by ~0.75 dex. Worse, because surface density correlates with the
    per-galaxy FOV, a constant offset+scale error maps onto redshift and manufactures a
    spurious bias-vs-redshift trend -- which is exactly the quantity this study measures.

    Delegating to baryonbridge.config keeps one source of truth: NORM_PROFILE derives from
    GRID_SUFFIX on lookup, so a tag registered after import still resolves correctly, and
    unknown tags fall back to 'fixed32' (an empty overlay), leaving fixed-FOV runs bit-identical.
    """
    _, model_version = parse_version_tag(version_tag)

    # Guard, from a bug that was live for weeks. merged_norm_dict overlays with setdefault,
    # so a profile that introduces new field names (a surface-density variant defining
    # sigmaDM but not rhoDM) leaves the base rhoDM entry in place and the lookup below
    # SUCCEEDS with constants belonging to a different data variant. Getting both mean and
    # std wrong makes the induced error a function of the pixel's own value, and density
    # falls with radius, so the error maps onto RADIUS: +0.09 dex at the centre, -0.41 at
    # the FOV edge. It manufactured a -0.49 dex tilt that reads exactly like an outskirt
    # under-prediction -- the quantity the analysis measures -- and was reported as a real
    # result before it was found. Any profile that is non-empty and does not define rhoDM is
    # by construction describing channels that are not rhoDM, so answering with rhoDM
    # constants answers a different question. Raise instead.
    profile = get_norm_profile(model_version)
    if profile and 'rhoDM' not in profile.get('2D', {}):
        raise KeyError(
            f"Tag {model_version!r} resolves to a normalization profile that defines "
            f"{sorted(profile.get('2D', {}))} but not 'rhoDM'. Returning the base rhoDM "
            f"constants here would silently denormalize with another variant's numbers. "
            f"This release covers the MW rhoDM runs only; see docs/CAVEATS.md.")

    norm = merged_norm_dict(NORM_DICT, get_norm_profile(model_version))
    entry = norm['2D']['rhoDM'][resolution]
    return entry['mean'], entry['std']


def file_stem(galaxy_idx, sim_label, model_version):
    if sim_label:
        return f"galaxy_{galaxy_idx}_{sim_label}_{model_version}"
    return f"galaxy_{galaxy_idx}_{model_version}"


def default_draws_file_pattern(version_tag, num_draws):
    """Version-specific glob for per-galaxy draw npz files."""
    sim_label, model_version = parse_version_tag(version_tag)
    if sim_label:
        return f"galaxy_*_{sim_label}_{model_version}_pred_draws_{num_draws}*.npz"
    return f"galaxy_*_{model_version}_pred_draws_{num_draws}.npz"


def galaxy_dir(pred_root, galaxy_idx, sim_label=None):
    root = effective_pred_root(pred_root, sim_label)
    return root / f"galaxy_{galaxy_idx:04d}"


def _resolve_galaxy_dir(pred_root, galaxy_idx, version_tag):
    sim_label, _ = parse_version_tag(version_tag)
    return galaxy_dir(pred_root, galaxy_idx, sim_label)


def radial_profile_path(pred_root, galaxy_idx, version_tag, output_suffix=None):
    sim_label, model_version = parse_version_tag(version_tag)
    version = model_version
    if output_suffix:
        version = f"{model_version}_{output_suffix}"
    stem = file_stem(galaxy_idx, sim_label, version)
    return _resolve_galaxy_dir(pred_root, galaxy_idx, version_tag) / (
        f"{stem}_galaxy_{galaxy_idx}_radial_profile.npz"
    )


def draws_path(pred_root, galaxy_idx, version_tag, num_draws, output_suffix=None):
    sim_label, model_version = parse_version_tag(version_tag)
    gdir = _resolve_galaxy_dir(pred_root, galaxy_idx, version_tag)
    stem = file_stem(galaxy_idx, sim_label, model_version)
    candidates = []
    if output_suffix:
        candidates.append(gdir / f"{stem}_pred_draws_{num_draws}_{output_suffix}.npz")
        candidates.append(
            gdir / (
                f"galaxy_{galaxy_idx}_{model_version}_pred_draws_{num_draws}_{output_suffix}.npz"
            )
        )
        if sim_label:
            candidates.append(
                gdir / (
                    f"galaxy_{galaxy_idx}_{sim_label}_{model_version}_pred_draws_"
                    f"{num_draws}_{output_suffix}.npz"
                )
            )
        candidates.append(
            gdir / (
                f"galaxy_{galaxy_idx}_{model_version}_{output_suffix}_pred_draws_{num_draws}.npz"
            )
        )
        if sim_label:
            # batch_plot_pred_uncertainty_da.py writes the OOD suites with the suffix BEFORE
            # "_pred_draws_" and the sim label present, e.g.
            #   galaxy_7_TNG50_v95_heun100_n100_pred_draws_100.npz
            # which none of the orderings above match. Without this, every TNG50/FIRE lookup
            # falls through to candidates[0], the caller sees a missing file, and the suite is
            # silently skipped rather than erroring.
            candidates.append(
                gdir / (
                    f"galaxy_{galaxy_idx}_{sim_label}_{model_version}_{output_suffix}"
                    f"_pred_draws_{num_draws}.npz"
                )
            )
    candidates.append(gdir / f"{stem}_pred_draws_{num_draws}.npz")
    candidates.append(
        gdir / f"galaxy_{galaxy_idx}_{model_version}_pred_draws_{num_draws}.npz"
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_draws_filepath(
    pred_root,
    galaxy_idx,
    version_tag,
    num_draws=10,
    output_suffix=None,
    file_pattern=None,
):
    gdir = _resolve_galaxy_dir(pred_root, galaxy_idx, version_tag)
    sim_label, model_version = parse_version_tag(version_tag)
    stem = file_stem(galaxy_idx, sim_label, model_version)
    exact = draws_path(pred_root, galaxy_idx, version_tag, num_draws, output_suffix)
    if not gdir.exists():
        return exact
    if exact.exists():
        return exact

    if file_pattern:
        matches = sorted(
            p
            for p in gdir.iterdir()
            if p.is_file() and fnmatch.fnmatch(p.name, file_pattern)
        )
        stem_matches = [p for p in matches if p.name.startswith(stem)]
        if stem_matches:
            return stem_matches[0]
        if sim_label:
            tag_matches = [
                p
                for p in matches
                if f"_{sim_label}_{model_version}_pred_draws_" in p.name
            ]
            if tag_matches:
                return tag_matches[0]
        version_matches = [
            p for p in matches if f"_{model_version}_pred_draws_" in p.name
        ]
        if version_matches:
            return version_matches[0]
        if matches:
            return matches[0]
    return exact


def discover_galaxy_indices(
    pred_root,
    version_tag,
    num_draws=10,
    output_suffix=None,
    file_pattern=None,
):
    sim_label, _ = parse_version_tag(version_tag)
    scan_root = effective_pred_root(pred_root, sim_label)
    if not scan_root.exists():
        return []
    indices = []
    for entry in sorted(scan_root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("galaxy_"):
            continue
        galaxy_idx = int(entry.name.split("_")[1])
        path = resolve_draws_filepath(
            pred_root,
            galaxy_idx,
            version_tag,
            num_draws,
            output_suffix,
            file_pattern,
        )
        if path.exists():
            indices.append(galaxy_idx)
    return sorted(indices)


def radial_profiles_exist(
    pred_root,
    version_tag,
    galaxy_indices=None,
    output_suffix=None,
):
    sim_label, _ = parse_version_tag(version_tag)
    scan_root = effective_pred_root(pred_root, sim_label)
    if not scan_root.exists():
        return False
    if galaxy_indices is None:
        for entry in sorted(scan_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("galaxy_"):
                continue
            galaxy_idx = int(entry.name.split("_")[1])
            if radial_profile_path(pred_root, galaxy_idx, version_tag, output_suffix).exists():
                return True
        return False
    for galaxy_idx in galaxy_indices:
        if radial_profile_path(pred_root, galaxy_idx, version_tag, output_suffix).exists():
            return True
    return False


def build_structured_from_sim_maps(
    rho_map_path,
    galaxy_indices,
    radial_centers=None,
    resolution=128,
):
    """Build structured radial profiles from true rhoDM maps when npz profiles are absent."""
    rho_map_path = Path(rho_map_path)
    rho_maps = np.load(rho_map_path)
    n_maps = rho_maps.shape[0]
    _, default_centers, mask = build_radial_annulus_geometry(resolution)
    if radial_centers is None:
        radial_centers = default_centers
    num_bins = len(radial_centers)

    galaxy_ids = []
    radial_centers_all = []
    x1_profiles_all = []
    x1_stds_all = []
    nan_profile = np.full(num_bins, np.nan)
    skipped = []

    for galaxy_idx in galaxy_indices:
        if galaxy_idx < 0 or galaxy_idx >= n_maps:
            skipped.append(galaxy_idx)
            continue
        # agg="mean" to match what the per-galaxy radial_profile.npz files store, so a
        # suite that has no npz (e.g. FIREm12) is aggregated the same way as one that
        # does (DREAMS, TNG50). Previously this fallback used the median while the npz
        # path used the mean, which meant Fig. 6 compared TNG50 and FIRE under slightly
        # different definitions of the true profile (a ~0.04-0.06 dex offset at large
        # radius, against a ~0.09 dex profile-level uncertainty).
        profile = compute_radial_profile_from_map(rho_maps[galaxy_idx], mask, agg="mean")
        profile_std = compute_radial_profile_from_map(rho_maps[galaxy_idx], mask, agg="std")
        galaxy_ids.append(galaxy_idx)
        radial_centers_all.append(radial_centers)
        x1_profiles_all.append(profile)
        x1_stds_all.append(profile_std)

    if skipped:
        print(
            f"Warning: skipped {len(skipped)} galaxy indices outside "
            f"{rho_map_path.name} (n={n_maps}): {skipped[:5]}"
            + (" ..." if len(skipped) > 5 else "")
        )

    if not galaxy_ids:
        raise FileNotFoundError(
            f"No galaxies loaded from maps at {rho_map_path} "
            f"(n={n_maps}) for indices {galaxy_indices}"
        )

    sort_idx = np.argsort(galaxy_ids)
    n_gal = len(galaxy_ids)
    return {
        "galaxy_ids": np.array(galaxy_ids)[sort_idx],
        "radial_centers": np.array(radial_centers_all)[sort_idx],
        "x0_profiles": np.tile(nan_profile, (n_gal, 1))[sort_idx],
        "x1_profiles": np.array(x1_profiles_all)[sort_idx],
        "x_pred_profiles": np.tile(nan_profile, (n_gal, 1))[sort_idx],
        "x0_stds": np.tile(nan_profile, (n_gal, 1))[sort_idx],
        "x1_stds": np.array(x1_stds_all)[sort_idx],
        "x_pred_stds": np.tile(nan_profile, (n_gal, 1))[sort_idx],
    }


def parse_draws_slice(spec):
    if spec is None:
        return slice(None)
    if isinstance(spec, slice):
        return spec
    if isinstance(spec, tuple):
        if len(spec) == 2:
            return slice(spec[0], spec[1])
        raise ValueError(f"draw slice tuple must be (start, end), got {spec}")
    spec = str(spec).strip()
    if spec == '' or spec.lower() == 'none':
        return slice(None)
    if ':' not in spec:
        raise ValueError(f"draw slice must be start:end, got {spec!r}")
    start_str, end_str = spec.split(':', 1)
    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None
    return slice(start, end)


def check_ood_path_examples(pred_root=None):
    """Return (label, resolved_path, exists) for canonical DREAMS/TNG50/FIRE path layouts."""
    pred_root = Path(pred_root or DEFAULT_PRED_ROOT)
    checks = [
        ("DREAMS draws", draws_path(pred_root, 0, "v55", 10)),
        ("TNG50 draws", draws_path(pred_root, 0, "TNG50_v55", 10)),
        (
            "FIRE sweep draws",
            resolve_draws_filepath(
                pred_root,
                0,
                "FIREm12_v55",
                640,
                file_pattern="*pred_draws_640_FIREm12_flat_prior_baryon_64.npz",
            ),
        ),
        ("DREAMS radial", radial_profile_path(pred_root, 0, "v55")),
        ("TNG50 radial", radial_profile_path(pred_root, 0, "TNG50_v55")),
    ]
    return [(label, path, path.exists()) for label, path in checks]


def extract_draws_from_npz(data, draw_slice, channel=0):
    draws = data['draws']
    draws = draws[draw_slice]
    if draws.ndim == 4:
        draws = draws[:, channel]
    elif draws.ndim != 3:
        raise ValueError(f"Unexpected draws shape after slice: {draws.shape}")
    return draws


def load_theta_table(theta_path=None):
    import pandas as pd
    theta_path = Path(theta_path or DEFAULT_THETA_PATH)
    column_names = PARAM_NAMES
    params = pd.read_csv(
        theta_path,
        sep=" ",
        names=column_names,
        comment='#',
    )
    params["galaxy_id"] = np.arange(len(params))
    return params


def split_label(galaxy_idx, version_tag=None):
    """
    Which split a galaxy belongs to: "train" | "val" | "test" | "ood".

    TNG50 and FIREm12 were never trained on, so no split exists for them and every galaxy
    is scored: they are labelled "ood". Applying the DREAMS index ranges to them would
    silently call TNG50 galaxy 300 a "train" galaxy and drop it from the population.
    """
    sim_label, _ = parse_version_tag(version_tag)
    if sim_label:
        return "ood"
    if galaxy_idx < TEST_IDX_END:
        return "test"
    if galaxy_idx < TRAIN_IDX_START:
        return "val"
    return "train"


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


# ---------------------------------------------------------------------------
# Ensemble / annulus combination convention
# ---------------------------------------------------------------------------
# Two independent choices, both historically "median":
#
#   ENSEMBLE_COMBINE  how the stochastic draws collapse to the fiducial prediction
#   PROFILE_AGG       how pixels combine into one annulus value
#
# The median was chosen when the ensemble was 10 draws and one outlier could move a
# mean. At 100 draws the mean is the better estimator, and it is what Eq. 1 of the
# paper already uses to define sigma_pred; the annulus aggregation also has to be the
# mean for the profile-level Delta to have the same convention in numerator and
# denominator (the true profile has used the mean since 2026-07-31, the predicted one
# had not been followed up).
#
# Defined here because it is the one module every figure script imports. Set it through
# set_combiners() rather than assigning the names in an importing module, which would
# only rebind that module's own copy.
ENSEMBLE_COMBINE = "mean"
PROFILE_AGG = "mean"


def set_combiners(ensemble=None, profile=None):
    """Set the combination convention process-wide."""
    global ENSEMBLE_COMBINE, PROFILE_AGG
    if ensemble is not None:
        if ensemble not in ("mean", "median"):
            raise ValueError(f"Unknown ensemble combiner: {ensemble}")
        ENSEMBLE_COMBINE = ensemble
    if profile is not None:
        if profile not in ("mean", "median"):
            raise ValueError(f"Unknown profile aggregation: {profile}")
        PROFILE_AGG = profile
    return ENSEMBLE_COMBINE, PROFILE_AGG


def combine_label():
    """The word for the current ensemble combiner, for filenames and figure text."""
    return ENSEMBLE_COMBINE


def combine_draws(values, axis=0, nan_safe=False):
    """Collapse the stochastic ensemble along `axis` into the fiducial prediction."""
    if ENSEMBLE_COMBINE == "mean":
        return np.nanmean(values, axis=axis) if nan_safe else np.mean(values, axis=axis)
    return np.nanmedian(values, axis=axis) if nan_safe else np.median(values, axis=axis)


def aggregate_pixels(values, axis=None, nan_safe=False):
    """Combine pixels within an annulus into a single profile value."""
    if PROFILE_AGG == "mean":
        return np.nanmean(values, axis=axis) if nan_safe else np.mean(values, axis=axis)
    return np.nanmedian(values, axis=axis) if nan_safe else np.median(values, axis=axis)


def compute_radial_profile_from_map(map_2d, mask, agg="mean"):
    num_bins = mask.shape[0]
    profile = np.full(num_bins, np.nan)
    for j in range(num_bins):
        vals = map_2d[mask[j]]
        if vals.size > 0:
            if agg == "mean":
                profile[j] = np.mean(vals)
            elif agg == "median":
                profile[j] = np.median(vals)
            elif agg == "std":
                profile[j] = np.std(vals)
            else:
                raise ValueError(f"Unknown agg: {agg}")
    return profile


def compute_radial_profiles_from_draws(
    draws,
    mask,
    agg_pixel="median",
    agg_draw="median",
):
    """Compute radial profile per draw then aggregate across draws."""
    n_draws = draws.shape[0]
    num_bins = mask.shape[0]
    per_draw = np.full((n_draws, num_bins), np.nan)
    for d in range(n_draws):
        per_draw[d] = compute_radial_profile_from_map(draws[d], mask, agg=agg_pixel)

    if agg_draw == "mean":
        return np.nanmean(per_draw, axis=0)
    if agg_draw == "median":
        return np.nanmedian(per_draw, axis=0)
    raise ValueError(f"Unknown agg_draw: {agg_draw}")


def compute_median_median_profile(draws, mask):
    """Match plot_radial_profile_figures: median over pixels of per-pixel draw median."""
    num_bins = mask.shape[0]
    profile = np.full(num_bins, np.nan)
    for j in range(num_bins):
        draws_j = draws[:, mask[j]]
        profile[j] = np.median(np.median(draws_j, axis=0))
    return profile


def denormalize_draws(draws, resolution=128, version_tag=None, field=None):
    """
    Standardized draws -> log10 density.

    Every model in this release predicts rhoDM on the fixed +/-32 kpc grid, so `field` has
    exactly one valid value and is accepted only to reject the others loudly. A variant
    predicting a different channel (surface density, a different aperture) needs its own
    constants; silently reusing rhoDM's is the 2026-09 fall-through bug described in
    rhodm_norm_constants.
    """
    if field is not None and field != "rhoDM":
        raise ValueError(
            f"denormalize_draws got field={field!r}; this release ships rhoDM models only. "
            f"Denormalizing another channel with rhoDM constants would corrupt it silently.")
    mean_rho, std_rho = rhodm_norm_constants(version_tag, resolution)
    return draws * std_rho + mean_rho


def load_all_data_structured(
    pred_root,
    version_tag,
    galaxy_indices=None,
    layout="per_galaxy",
    flat_data_dir=None,
    # The suffix belongs to the run that WROTE these profiles, which is not always the
    # run being evaluated: the MW configs score against v47, whose files are un-suffixed,
    # while the dwarf runs are their own truth source and carry "heun100_n100". Default
    # None reproduces the historical un-suffixed lookup exactly.
    output_suffix=None,
):
    if layout == "flat":
        if flat_data_dir is None:
            raise ValueError("flat_data_dir required for flat layout")
        return _load_all_data_structured_flat(flat_data_dir, galaxy_indices)

    pred_root = Path(pred_root)
    if galaxy_indices is None:
        galaxy_indices = discover_galaxy_indices(
            pred_root, version_tag, num_draws=10
        )
        radial_indices = []
        for idx in range(1024):
            rp = radial_profile_path(pred_root, idx, version_tag, output_suffix)
            if rp.exists():
                radial_indices.append(idx)
        galaxy_indices = sorted(set(galaxy_indices) | set(radial_indices))

    galaxy_ids = []
    radial_centers_all = []
    x0_profiles_all = []
    x1_profiles_all = []
    x_pred_profiles_all = []
    x0_stds_all = []
    x1_stds_all = []
    x_pred_stds_all = []

    for galaxy_idx in galaxy_indices:
        filepath = radial_profile_path(pred_root, galaxy_idx, version_tag, output_suffix)
        if not filepath.exists():
            continue
        data = np.load(filepath)
        galaxy_ids.append(galaxy_idx)
        radial_centers_all.append(data['radial_centers'])
        x0_profiles_all.append(data['x0_radial_profile'])
        x1_profiles_all.append(data['x1_radial_profile'])
        x_pred_profiles_all.append(data['x_pred_radial_profile'])
        x0_stds_all.append(data['x0_radial_profile_std'])
        x1_stds_all.append(data['x1_radial_profile_std'])
        x_pred_stds_all.append(data['x_pred_radial_profile_std'])
        data.close()

    if not galaxy_ids:
        raise FileNotFoundError(
            f"No radial profiles found for version {version_tag} under {pred_root}"
        )

    sort_idx = np.argsort(galaxy_ids)
    return {
        'galaxy_ids': np.array(galaxy_ids)[sort_idx],
        'radial_centers': np.array(radial_centers_all)[sort_idx],
        'x0_profiles': np.array(x0_profiles_all)[sort_idx],
        'x1_profiles': np.array(x1_profiles_all)[sort_idx],
        'x_pred_profiles': np.array(x_pred_profiles_all)[sort_idx],
        'x0_stds': np.array(x0_stds_all)[sort_idx],
        'x1_stds': np.array(x1_stds_all)[sort_idx],
        'x_pred_stds': np.array(x_pred_stds_all)[sort_idx],
    }


def _load_all_data_structured_flat(data_dir, idx_list=None):
    npz_files = [
        f for f in os.listdir(data_dir) if f.endswith('radial_profile.npz')
    ]
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
    result = {
        'galaxy_ids': galaxy_ids[sort_idx],
        'radial_centers': np.array(radial_centers_all)[sort_idx],
        'x0_profiles': np.array(x0_profiles_all)[sort_idx],
        'x1_profiles': np.array(x1_profiles_all)[sort_idx],
        'x_pred_profiles': np.array(x_pred_profiles_all)[sort_idx],
        'x0_stds': np.array(x0_stds_all)[sort_idx],
        'x1_stds': np.array(x1_stds_all)[sort_idx],
        'x_pred_stds': np.array(x_pred_stds_all)[sort_idx],
    }
    if idx_list is not None:
        mask = np.isin(result['galaxy_ids'], idx_list)
        for key in result:
            result[key] = result[key][mask]
    return result


def load_pred_data_structured(
    pred_root,
    version_tag,
    num_draws=10,
    output_suffix=None,
    galaxy_indices=None,
    draw_slice=None,
    channel=0,
    layout="per_galaxy",
    flat_data_dir=None,
    file_pattern=None,
):
    if layout == "flat":
        return _load_pred_data_structured_flat(
            flat_data_dir,
            galaxy_indices,
            file_pattern=file_pattern,
            draw_slice=draw_slice,
            channel=channel,
            version_tag=version_tag,
        )

    pred_root = Path(pred_root)
    draw_slice = parse_draws_slice(draw_slice)
    if galaxy_indices is None:
        galaxy_indices = discover_galaxy_indices(
            pred_root, version_tag, num_draws, output_suffix, file_pattern
        )

    mean_rho, std_rho = rhodm_norm_constants(version_tag)

    galaxy_ids = []
    x_pred_draws_all = []

    for galaxy_idx in galaxy_indices:
        filepath = resolve_draws_filepath(
            pred_root,
            galaxy_idx,
            version_tag,
            num_draws,
            output_suffix,
            file_pattern,
        )
        if not filepath.exists():
            continue
        data = np.load(filepath)
        draws_temp = extract_draws_from_npz(data, draw_slice, channel)
        draws_temp = draws_temp * std_rho + mean_rho
        galaxy_ids.append(galaxy_idx)
        x_pred_draws_all.append(draws_temp)
        data.close()

    if not galaxy_ids:
        raise FileNotFoundError(
            f"No draw files for version={version_tag} pattern={file_pattern!r} "
            f"suffix={output_suffix} under {pred_root}"
        )

    sort_idx = np.argsort(galaxy_ids)
    return {
        'galaxy_ids': np.array(galaxy_ids)[sort_idx],
        'x_pred_draws': np.array(x_pred_draws_all)[sort_idx],
    }


def _load_pred_data_structured_flat(
    data_dir,
    idx_list=None,
    file_pattern='*pred_draws_10.npz',
    draw_slice=None,
    channel=0,
    version_tag=None,
):
    draw_slice = parse_draws_slice(draw_slice)
    npz_files = sorted(
        f for f in os.listdir(data_dir) if fnmatch.fnmatch(f, file_pattern)
    )
    mean_rho, std_rho = rhodm_norm_constants(version_tag)

    galaxy_ids = []
    x_pred_draws_all = []
    for filename in npz_files:
        filepath = os.path.join(data_dir, filename)
        data = np.load(filepath)
        galaxy_id = int(filename.split('_')[1])
        draws_temp = extract_draws_from_npz(data, draw_slice, channel)
        draws_temp = draws_temp * std_rho + mean_rho
        galaxy_ids.append(galaxy_id)
        x_pred_draws_all.append(draws_temp)
        data.close()

    sort_idx = np.argsort(galaxy_ids)
    galaxy_ids = np.array(galaxy_ids)[sort_idx]
    x_pred_draws_all = np.array(x_pred_draws_all)[sort_idx]
    if idx_list is not None:
        mask = np.isin(galaxy_ids, idx_list)
        galaxy_ids = galaxy_ids[mask]
        x_pred_draws_all = x_pred_draws_all[mask]

    return {
        'galaxy_ids': galaxy_ids,
        'x_pred_draws': x_pred_draws_all,
    }
