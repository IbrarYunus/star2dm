"""
Configuration constants for BaryonBridge project.

This module contains configuration dictionaries for different versions of the project,
including grid suffixes and resolution settings.
"""

# Grid suffix configuration for different versions
GRID_SUFFIX = {
    'v12': '',
    'v13': '_x-8_8_y-8_8_z-8_8',
    'v14': '_x-8_8_y-8_8_z-8_8',
    'v15': '_x-8_8_y-8_8_z-1_1',
    'v16': '_x-8_8_y-8_8_z-1_1',
    'v17': '_x-32_32_y-32_32_z-1_1',
    'v18': '_x-8_8_y-8_8_z-1_1',
    'v19': '_x-8_8_y-8_8_z-1_1',
    'v20': '_x-8_8_y-8_8_z-1_1',
    'v21': '_x-8_8_y-8_8_z-1_1',
    'v22': '_x-8_8_y-8_8_z-1_1',
    'v23': '_x-8_8_y-8_8_z-1_1',
    'v24': '_x-8_8_y-8_8_z-8_8',
    'v25': '_x-8_8_y-8_8_z-8_8',
    'v26': '_x-8_8_y-8_8_z-8_8',
    'v27': '_x-32_32_y-32_32_z-32_32',
    'v28': '_x-32_32_y-32_32_z-32_32',
    'v29': '_x-8_8_y-8_8_z-8_8',
    'v30': '_x-32_32_y-32_32_z-32_32',
    'v31': '_x-32_32_y-32_32_z-32_32',
    'v32': '_x-32_32_y-32_32_z-32_32',
    'v33': '_csv_fov_z0p300_0p400',
    'v34': '_csv_fov_z0p200_0p500',
    'v35': '_csv_fov_z0p050_0p650',
    'v36': '_csv_fov_z0p050_0p650',
    'v37': '_csv_fov_z0p050_0p650',
    'v38': '_csv_fov_z0p050_0p650',
    'v39': '_csv_fov_z0p050_0p650',
    'v40': '_csv_fov_z0p300_0p400',
    'v41': '_csv_fov_z0p200_0p500',
    'v42': '_csv_fov_z0p050_0p650',
    'v43': '_csv_fov_z0p050_0p650',
    'v44': '_csv_fov_z0p050_0p650',
    'v44TNG50': '_csv_fov_z0p050_0p450',
    'v45': '_csv_fov_z0p350_0p350',
    'v46': '_csv_fov_z0p300_0p400',
    'v47': '_csv_fov_z0p350_0p350',
    'v48': '_csv_fov_z0p050_0p650',
    'v49': '_csv_fov_z0p050_0p650',
    'v50': '_csv_fov_z0p350_0p350_TNG50stack',
    'v51': '_csv_fov_z0p350_0p350_200DREAMS_200TNG50_rand',
    'v51D': '_csv_fov_z0p350_0p350',
    'v51T': '_csv_fov_z0p350_0p350_TNG50stack',
    'v52': '_csv_fov_z0p350_0p350',
    'v53': '_csv_fov_z0p350_0p350',
    'v54': '_csv_fov_z0p350_0p350',
    'v55': '_csv_fov_z0p350_0p350',
    'v56': '_csv_fov_z0p350_0p350',
    'v57': '_csv_fov_z0p350_0p350',
}

# Sweep / future training versions (same 128 grid as v55)
for _v in range(58, 130):
    GRID_SUFFIX[f'v{_v}'] = '_csv_fov_z0p350_0p350'

# Resolution configuration for different versions
RESOLUTION = {
    'v12': 128,
    'v13': 128,
    'v14': 32,
    'v15': 32,
    'v16': 128,
    'v17': 128,
    'v18': 128,
    'v19': 128,
    'v20': 128,
    'v21': 128,
    'v22': 128,
    'v23': 128,
    'v24': 128,
    'v25': 128,
    'v26': 128,
    'v27': 128,
    'v28': 32,
    'v29': 32,
    'v30': 128,
    'v31': 128,
    'v32': 128,
    'v33': 128,
    'v34': 128,
    'v35': 128,
    'v36': 128,
    'v37': 128,
    'v38': 128,
    'v39': 128,
    'v40': 128,
    'v41': 128,
    'v42': 128,
    'v43': 128,
    'v44': 128,
    'v44TNG50': 128,
    'v45': 128,
    'v46': 128,
    'v47': 128,
    'v48': 128,
    'v49': 128,
    'v50': 128,
    'v51': 128,
    'v51D': 128,
    'v51T': 128,
    'v52': 128,
    'v53': 128,
    'v54': 128,
    'v55': 128,
    'v56': 128,
    'v57': 128,
}

for _v in range(58, 130):
    RESOLUTION[f'v{_v}'] = 128

# ---------------------------------------------------------------------------
# Normalization profiles
# ---------------------------------------------------------------------------
# The map (mean, std) constants in baryonbridge/data/{dreams,tng,fire}.py's NORM_DICT are
# keyed only by (dims, map_type, resolution) -- they carry no notion of which dataset
# variant produced the maps. But the csv_fov_* variants have very different statistics: at
# a fixed 12.8 arcsec angular FOV, a galaxy placed at z=0.05 is imaged over ~12.7 physical
# kpc while one at z=0.65 spans ~90 kpc, so the wide-z rhoDM maps are both lower and much
# broader than the fixed-z ones. Historically these constants were switched by hand
# (commenting blocks in and out of NORM_DICT), which meant only one dataset variant was
# loadable at a time and older runs silently stopped being reproducible.
#
# A profile is an overlay applied on top of NORM_DICT, selected by version tag, so fixed-z
# and wide-z models can coexist. "fixed32" is empty: it means "NORM_DICT as written".
NORM_PROFILES = {
    "fixed32": {},
    # csv_fov_z0p050_0p650 -- per-galaxy FOV drawn from redshifts spanning 0.05-0.65.
    "widez": {
        "2D": {
            "rhoDM": {
                128: {"mean": 6.1720, "std": 0.6126, "min": 4.3779},
                32: {"mean": 6.1720, "std": 0.6126, "min": 4.3779},
            },
            "rhoDMrand": {
                128: {"mean": 6.1720, "std": 0.6126, "min": 4.3779},
                32: {"mean": 6.1720, "std": 0.6126, "min": 4.3779},
            },
            # rhostar had no wide-z entry at all. Measured (job 43769311) from
            # DREAMS_data/2D_maps/CDM/MW_zooms/SB5/Maps_rhostar_TNG_SB5_128_z=0.00_
            # csv_fov_z0p050_0p650.npy over all 1024 galaxies, matching how the rhoDM and
            # EuVISfluxnoiselog10 entries were derived -- that same run reproduced both of
            # those to 4 decimals over the full 1024, which is what pins the convention.
            # Train-only (204-1023) would give 4.7780 / 1.4158; the difference is well below
            # the fixed-z-vs-wide-z gap (4.5496 / 1.1889) that motivates the override.
            # See ML_dm_distribution/training_prep/check_widez_inputs.py.
            "rhostar": {
                128: {"mean": 4.7992, "std": 1.4120, "min": 0.6139},
                32: {"mean": 4.7992, "std": 1.4120, "min": 0.6139},
            },
            # EuVISfluxnoiselog10 and EuVISnoiselog10 already carry their wide-z values in
            # NORM_DICT itself, so they need no override here.
        },
    },
    # csv_fov_z0p200_0p500 ("medium redshift variation range" in the NORM_DICT comments).
    "medz": {
        "2D": {
            "rhoDM": {
                128: {"mean": 6.0832, "std": 0.5055, "min": 4.4544},
                32: {"mean": 6.0832, "std": 0.5055, "min": 4.4544},
            },
        },
    },
    # csv_fov_z0p300_0p400 ("small redshift variation range").
    "narrowz": {
        "2D": {
            "rhoDM": {
                128: {"mean": 6.0528, "std": 0.4842, "min": 4.6043},
                32: {"mean": 6.0528, "std": 0.4842, "min": 4.6043},
            },
        },
    },
    # DREAMS FIRE dwarf suite, +/-25 kpc FOV, face-on. Measured over all 396
    # galaxies in log10(Sigma + 1) units by
    #   ML_dm_distribution/training_prep/dwarf_fire_prep.py measure-norms
    #
    # Unlike the profiles above, this one is not an overlay on NORM_DICT --
    # baryonbridge/data/dwarffire.py merges it onto an EMPTY base, so these are
    # the only normalization constants that suite can see. A field missing here
    # raises rather than silently inheriting an MW-zoom value.
    #
    # No floor, no mask, no clip. The zero pixels sit inside the bulk's own left tail,
    # not outside it, and there is no gap between zero and the first nonzero value --
    # the distribution runs continuously down to 6e-14, so a floor would have nothing
    # to separate. A physical floor at "one star particle per resolution element"
    # (1e4 Msun/kpc^2) was considered and rejected: scatter deposition spreads each
    # particle over pi*hsml^2 with hsml the 32nd-NN distance, kpc-scale in dwarf
    # outskirts, so that threshold lands between p95 and p99 and would censor ~95% of
    # the map.
    #
    # Measured 2026-09-01 over all 396 galaxies. sigmastar has 1.53% exactly-zero
    # pixels, which standardize to -1.60 sigma. sigmaDMpred (= sigmaDM - satdark) has
    # none at all: its floor is 4.14e4 Msun/kpc^2, so it needs no special treatment
    # either.
    "dwarffire_fov25": {
        "2D": {
            "sigmastar": {
                128: {"mean": 1.7280, "std": 1.0806, "min": 0.0000},
            },
            "sigmaDM": {
                128: {"mean": 6.3107, "std": 0.3884, "min": 4.6166},
            },
            "sigmaDMpred": {
                128: {"mean": 6.3018, "std": 0.3884, "min": 4.6166},
            },
        },
    },
    # DREAMS MW-zoom SB5 Sigma suite, legacy32 grid (+/-32.596465 kpc, NOT 32).
    # Measured on the 820 TRAIN rows only by
    # ML_dm_distribution/training_prep/mw_sigma_prep.py measure-norms.
    #
    # These sit ~1.4 dex above the dwarf sigmaDM because the MW maps integrate a
    # +/-205 kpc column against the dwarfs' shallower one -- a real difference in the
    # data, not a units slip. Nothing here is interchangeable with either the dwarf
    # constants or the MW rhoDM ones: this suite has its own channels and its own depth.
    "mwzoom_legacy32": {
        "2D": {
            "sigmastar": {
                128: {"mean": 5.7078, "std": 0.9473, "min": 0.6061},
            },
            "sigmagas": {
                128: {"mean": 6.3230, "std": 0.5972, "min": 3.6870},
            },
            "sigmaDM": {
                128: {"mean": 7.7076, "std": 0.2889, "min": 6.7067},
            },
            "sigmaDMpred": {
                128: {"mean": 7.7064, "std": 0.2894, "min": 6.7067},
            },
            "sigmaDMsatdark": {
                128: {"mean": 1.0042, "std": 1.9444, "min": 0.0000},
            },
        },
    },
    # Same suite, the OTHER field of view: +/-110 kpc ("halo"), 1719 pc/pix. Not a
    # rescaling of legacy32 -- a different pixel grid over a different aperture, so
    # every constant differs. sigmaDM is 0.78 dex lower here simply because the wider
    # aperture averages in the low-density outskirts.
    #
    # This is the FOV the dwarf substructure comparison transfers to: 0.55 x median
    # R200c against the dwarfs' 0.54, ~38 detectable peaks per galaxy against
    # legacy32's 4. Mixing the two profiles would be a plausible-looking 0.78 dex error.
    "mwzoom_fov110": {
        "2D": {
            "sigmastar": {
                128: {"mean": 3.6413, "std": 1.1313, "min": 0.0000},
            },
            "sigmagas": {
                128: {"mean": 5.6242, "std": 0.5077, "min": 1.5867},
            },
            "sigmaDM": {
                128: {"mean": 6.9317, "std": 0.3932, "min": 5.4997},
            },
            "sigmaDMpred": {
                128: {"mean": 6.9269, "std": 0.3940, "min": 5.4997},
            },
            "sigmaDMsatdark": {
                128: {"mean": 1.1394, "std": 2.0065, "min": 0.0000},
            },
        },
    },
}

# Derived from GRID_SUFFIX so the two can never drift: a version tag's dataset variant
# determines its normalization. Any new tag registered above with a csv_fov redshift-range
# suffix picks up the matching profile automatically.
_NORM_PROFILE_BY_SUFFIX = {
    '_csv_fov_z0p050_0p650': 'widez',
    '_csv_fov_z0p200_0p500': 'medz',
    '_csv_fov_z0p300_0p400': 'narrowz',
    # DREAMS FIRE dwarf suite. Fixed physical FOV and an explicit view, so the
    # suffix names both. Different suite, different channels (sigmastar/sigmaDM
    # rather than rhostar/rhoDM) and a ~100x smaller halo mass, so none of the
    # MW-zoom constants apply.
    '_fov25kpc_faceon': 'dwarffire_fov25',
    # MW-zoom Sigma suite. Same channel NAMES as the dwarf suite but a different
    # column depth and a ~70x more massive host, so the constants are unrelated --
    # the shared names are exactly why this needs its own profile.
    '_legacy32_faceon': 'mwzoom_legacy32',
    '_fov110kpc_faceon': 'mwzoom_fov110',
}
class _NormProfileMap(dict):
    """
    Version tag -> profile name, derived from GRID_SUFFIX on a miss.

    This is a dict subclass rather than a plain dict comprehension for one specific reason:
    new runs are registered by appending to GRID_SUFFIX *after* this module's body has
    executed (see docs/final_model_pipeline.md). A frozen comprehension would not see those
    tags, so a wide-z model registered later would silently resolve to 'fixed32' and be
    loaded with fixed-redshift normalization constants -- wrong, and with nothing raising.
    Deriving on lookup makes the mapping order-independent.

    Explicit entries still win, so a tag can be pinned to a profile that disagrees with its
    suffix if a run ever needs that.
    """

    def __missing__(self, version):
        return _NORM_PROFILE_BY_SUFFIX.get(GRID_SUFFIX.get(version), 'fixed32')

    def get(self, version, default='fixed32'):
        if dict.__contains__(self, version):
            return dict.__getitem__(self, version)
        suffix = GRID_SUFFIX.get(version)
        if suffix is None:
            return default
        return _NORM_PROFILE_BY_SUFFIX.get(suffix, default)


# Deliberately EMPTY: every lookup derives from GRID_SUFFIX. Pre-populating it from
# GRID_SUFFIX at import would defeat the whole point -- the range(58, 130) loop already
# stamps v58..v129 with the fixed-redshift suffix, so those tags would carry a stale
# 'fixed32' entry that shadows the derivation when they are later re-registered as wide-z.
# Assigning into this dict still works and takes precedence, for the rare tag that needs a
# profile disagreeing with its suffix.
NORM_PROFILE = _NormProfileMap()


def norm_profile_table():
    """Materialize the derived version -> profile mapping, for inspection/debugging."""
    return {_v: NORM_PROFILE.get(_v) for _v in GRID_SUFFIX}
# Caveat on the legacy tags: the NORM_DICT comment "used for large redshift variation range
# and for all redshift ranges after v39" suggests v40 (z0p300_0p400) and v41 (z0p200_0p500)
# were actually trained with the wide-z constants rather than their range-matched ones.
# Mapping strictly by suffix is the defensible default; if either is ever re-run, check the
# job log before trusting the profile assigned here. v33-v39, v42-v44, v48, v49 are
# unambiguous.


# ---------------------------------------------------------------------------
# Conditioning-clip profiles
# ---------------------------------------------------------------------------
# Applied to the standardized maps, as {dims_key: {map_type: [lo_sigma, hi_sigma]}} with
# None meaning "unbounded on that side".
#
# Motivation: the wide-z EuVISfluxnoiselog10 maps reach about -55 sigma, at pixels where the
# noise realization drove the measured flux near zero. Their log10 value is set by how close
# to zero the noise landed, not by the galaxy -- unbounded downward and carrying no signal.
# Measured on the DREAMS wide-z file (job 43770959): the 0.01 percentile is -5.15 sigma, so
# a -5 sigma floor bounds the pathology while touching 1.2e-4 of pixels (about 2 per image).
#
# The clip is deliberately ONE-SIDED. The +21 sigma tail is bright galaxy centres -- real
# signal, and the single most informative part of the image. A symmetric clip would flatten
# it and destroy what the model is meant to use.
#
# Unlike NORM_PROFILE this CANNOT be derived from GRID_SUFFIX: clipping is a per-model recipe
# choice, not a property of the dataset variant, so two models trained on identical maps may
# legitimately differ here. That means every clipped run must be registered explicitly below,
# and inference resolves the clip from the version tag exactly as it resolves the
# normalization -- a tag missing from this table gets no clipping, which for a model trained
# with a clip would be a silent train/inference mismatch.
CLIP_PROFILES = {
    "none": {},
    "euvis_lo5": {"2D": {"EuVISfluxnoiselog10": [-5.0, None]}},
}

CLIP_PROFILE = {}  # version tag -> profile name; anything absent means "none"


def get_clip_profile(version):
    """Conditioning-clip overlay for a version tag; empty dict for unclipped runs."""
    return CLIP_PROFILES[CLIP_PROFILE.get(version, 'none')]


def get_norm_profile(version):
    """Normalization overlay for a version tag; empty dict for plain fixed-FOV runs."""
    return NORM_PROFILES[NORM_PROFILE.get(version, 'fixed32')]


def merged_norm_dict(base_norm_dict, profile):
    """
    Overlay `profile` on `base_norm_dict` without mutating either.

    Both are nested {dims_key: {map_type: {resolution: {stat: value}}}}. Only the leaves
    present in the profile are replaced, so a profile that only names rhoDM leaves every
    other field at its NORM_DICT value.
    """
    if not profile:
        return base_norm_dict
    merged = {
        dims_key: {
            map_type: {res: dict(stats) for res, stats in by_res.items()}
            for map_type, by_res in by_type.items()
        }
        for dims_key, by_type in base_norm_dict.items()
    }
    for dims_key, by_type in profile.items():
        merged.setdefault(dims_key, {})
        for map_type, by_res in by_type.items():
            merged[dims_key].setdefault(map_type, {})
            for res, stats in by_res.items():
                if any(v is None for v in stats.values()):
                    raise ValueError(
                        f"Normalization profile entry {dims_key}/{map_type}/{res} has "
                        f"unfilled (None) stats: {stats}. Run "
                        f"ML_dm_distribution/training_prep/check_widez_inputs.py and "
                        f"record the measured values in NORM_PROFILES."
                    )
                merged[dims_key][map_type][res] = dict(stats)
    return merged


# Version aliases for easier access
VERSIONS = list(GRID_SUFFIX.keys())

def get_grid_suffix(version):
    """Get the grid suffix for a given version."""
    if version not in GRID_SUFFIX:
        raise ValueError(f"Unknown version: {version}. Available versions: {VERSIONS}")
    return GRID_SUFFIX[version]

def get_resolution(version):
    """Get the resolution for a given version."""
    if version not in RESOLUTION:
        raise ValueError(f"Unknown version: {version}. Available versions: {VERSIONS}")
    return RESOLUTION[version]

def get_version_info(version):
    """Get both grid suffix and resolution for a given version."""
    return {
        'grid_suffix': get_grid_suffix(version),
        'resolution': get_resolution(version)
    } 
# ---------------------------------------------------------------------------
# Wide-redshift Euclid mock-image study (2026-08-17). Registered AFTER the
# range(58, 130) loop above, which would otherwise stamp these tags with the
# fixed-redshift suffix. NORM_PROFILE derives from GRID_SUFFIX on lookup, so the
# widez normalization follows automatically; CLIP_PROFILE does not derive and is
# set explicitly. See ML_dm_distribution/docs/final_model_pipeline.md appendix.
GRID_SUFFIX['v102'] = '_csv_fov_z0p050_0p650'  # EuVISfluxnoiselog10, theta=on
RESOLUTION['v102'] = 128
GRID_SUFFIX['v103'] = '_csv_fov_z0p050_0p650'  # EuVISfluxnoiselog10, theta=off
RESOLUTION['v103'] = 128
GRID_SUFFIX['v104'] = '_csv_fov_z0p050_0p650'  # rhostar, theta=on
RESOLUTION['v104'] = 128
GRID_SUFFIX['v105'] = '_csv_fov_z0p050_0p650'  # rhostar, theta=off
RESOLUTION['v105'] = 128
GRID_SUFFIX['v106'] = '_csv_fov_z0p050_0p650'  # EuVISfluxnoiselog10, theta=on, clip=euvis_lo5
RESOLUTION['v106'] = 128
CLIP_PROFILE['v106'] = 'euvis_lo5'

# ---------------------------------------------------------------------------
# DREAMS FIRE dwarf suite (2026-09-01). A 'dw' prefix rather than a vNN tag, so
# these cannot be captured by the range(58, 130) loop above -- that loop stamps
# every vNN tag with the MW fixed-redshift suffix, which for this suite would
# resolve to the wrong maps AND the wrong normalization with nothing raising.
# NORM_PROFILE derives from GRID_SUFFIX on lookup, so registering the suffix here
# is enough to pull in 'dwarffire_fov25'.
# See ML_dm_distribution/training_prep/dwarf_fire_run_manifest.csv.
GRID_SUFFIX['dw01'] = '_fov25kpc_faceon'   # sigmastar -> sigmaDM,     theta=on
RESOLUTION['dw01'] = 128
GRID_SUFFIX['dw02'] = '_fov25kpc_faceon'   # sigmastar -> sigmaDMpred, theta=on
RESOLUTION['dw02'] = 128

# DREAMS MW-zoom SB5 Sigma suite. The 'mw' prefix keeps these clear of the
# range(58, 130) loop above, which stamps every numeric vNN tag with the MW fixed-z
# rhoDM suffix -- a numeric tag here would silently inherit rhoDM normalization.
# See ML_dm_distribution/training_prep/mw_sigma_run_manifest.csv.
GRID_SUFFIX['mw01'] = '_legacy32_faceon'   # sigmastar -> sigmaDM,     theta=on
RESOLUTION['mw01'] = 128
GRID_SUFFIX['mw02'] = '_legacy32_faceon'   # sigmastar -> sigmaDMpred, theta=on
RESOLUTION['mw02'] = 128
# The +/-110 kpc "halo" FOV of the same suite. Separate tags rather than a flag on
# mw01/mw02 because the FOV changes the normalization, the annulus geometry and what
# the maps physically contain -- a tag that meant two apertures would be the single
# easiest way to mix them.
GRID_SUFFIX['mw03'] = '_fov110kpc_faceon'  # sigmastar -> sigmaDM,     theta=on
RESOLUTION['mw03'] = 128
GRID_SUFFIX['mw04'] = '_fov110kpc_faceon'  # sigmastar -> sigmaDMpred, theta=on
RESOLUTION['mw04'] = 128
