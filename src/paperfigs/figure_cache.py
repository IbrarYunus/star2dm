#!/usr/bin/env python3
"""Small reduced arrays that stand in for 15 GB of prediction draws.

The figures in this paper are not drawn from the draws directly. A population figure --
the deviation distributions, the predicted-vs-true scatters, every panel that puts 1024
galaxies on one axis -- is drawn from arrays of shape (n_galaxies, 32): one value per
galaxy per radial annulus. Getting there costs 6.7 GB of reading and ~20 minutes; the
result is under a megabyte.

So the result is what gets cached. A reader who wants to restyle a panel, change a label,
or check a number needs the cache and nothing else. A reader who wants to change the
binning, the annulus geometry or the combiner needs the draws, because those choices are
made upstream of what is stored here.

What a cache entry holds
------------------------
    profiles.npz            the population arrays: truth profiles, predicted profiles,
                            and the spread measures derived from the draws
    draws_galaxy_<id>.npz   full draws for the handful of galaxies that get their own
                            pixel-level panel (the featured galaxy's stochastic-draw
                            figure genuinely shows individual draws; nothing else does)
    meta.json               what produced it, so a stale entry is recognisable

Entries are keyed by everything that changes the numbers -- version tag, draws pattern,
galaxy selection, draw slice, and the two combiner settings -- so a cache built for one
figure set can never be served to another. The key is a readable prefix plus a hash;
the readable part is for humans reading `ls`, the hash is what guarantees correctness.

Modes
-----
    auto   read the cache if it is there, otherwise compute and write it (default)
    read   require the cache; fail loudly if it is missing, rather than silently
           starting a 20-minute read of draws the reader may not have
    write  always recompute and overwrite -- use after changing a reduction
    off    ignore the cache entirely
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
MODES = ("auto", "read", "write", "off")


def _slug(text, limit=48):
    """A filesystem-safe, readable fragment of an arbitrary string."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")[:limit] or "none"


class FigureCache:
    """One cache entry, addressed by the settings that determine its contents."""

    def __init__(self, root, mode="auto", **key_parts):
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        # A key part whose value is None is dropped, not hashed as the string "None". This
        # is what lets an optional argument be ADDED to a call site later without
        # invalidating every entry built before it existed: passing draw_slice=None must
        # hash identically to not passing draw_slice at all. Anything with a value is
        # hashed, so a real setting still separates entries.
        self.key_parts = {k: str(v) for k, v in sorted(key_parts.items()) if v is not None}
        digest = hashlib.sha256(
            json.dumps(self.key_parts, sort_keys=True).encode()
        ).hexdigest()[:12]
        self.digest = digest
        # The directory name is a readable prefix plus the hash. Only the hash decides
        # identity -- the prefix exists so `ls` and pack_figure_cache.py --list are legible,
        # and is built from whichever identifying fields this kind of entry actually has
        # (the calibration entries carry no version tag, so they would all read "none").
        prefix = "-".join(
            _slug(self.key_parts[k], limit=24)
            for k in ("version", "kind", "sample")
            if self.key_parts.get(k)
        ) or "entry"
        name = f"{prefix}__{digest}"
        self.dir = Path(root) / name
        self.profiles_path = self.dir / "profiles.npz"
        self.meta_path = self.dir / "meta.json"

    # -- state ---------------------------------------------------------------------
    @property
    def readable(self):
        return self.mode in ("auto", "read") and self.profiles_path.exists()

    @property
    def writable(self):
        return self.mode in ("auto", "write")

    def require(self):
        """Fail with an actionable message rather than a KeyError three frames down."""
        if self.mode == "read" and not self.profiles_path.exists():
            raise FileNotFoundError(
                f"--cache-mode read was requested but no cache entry exists at "
                f"{self.profiles_path}.\nKey: {json.dumps(self.key_parts, indent=2)}\n"
                f"Either fetch the published cache (python scripts/fetch_data.py --tier cache) "
                f"or build it from draws with --cache-mode write."
            )

    # -- population arrays ----------------------------------------------------------
    def load_profiles(self):
        """(structured_data, draw_profiles) as the plotting functions expect them."""
        with np.load(self.profiles_path, allow_pickle=False) as data:
            structured, draws = {}, {}
            for key in data.files:
                if key.startswith("dp__"):
                    draws[key[4:]] = data[key]
                elif key.startswith("sd__"):
                    structured[key[4:]] = data[key]
        return structured, (draws or None)

    def save_profiles(self, structured, draw_profiles):
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {f"sd__{k}": np.asarray(v) for k, v in structured.items()}
        if draw_profiles:
            payload.update({f"dp__{k}": np.asarray(v) for k, v in draw_profiles.items()})
        np.savez_compressed(self.profiles_path, **payload)
        self._write_meta(payload)

    # -- per-galaxy draws -----------------------------------------------------------
    # Only for galaxies whose figure shows individual draws rather than their summary.
    #
    # float32, not float16. float16 was tried and is wrong here: near log10(rho) = 6 its
    # spacing is 0.0039 dex, and the residual maps divide by an ensemble std of ~0.09 dex,
    # so the quantization lands at 0.04 in units of Delta -- enough to move counts across
    # bin boundaries. The galaxy-19 histogram and binned residual map came out visibly
    # different from the same figure built from the draws. At float32 they are pixel-
    # identical. The saving was 3 MB.
    def draws_path(self, galaxy_id):
        return self.dir / f"draws_galaxy_{int(galaxy_id)}.npz"

    def load_galaxy_draws(self, galaxy_id):
        path = self.draws_path(galaxy_id)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return data["draws"].astype(np.float32)

    def save_galaxy_draws(self, galaxy_id, draws):
        self.dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.draws_path(galaxy_id), draws=np.asarray(draws, dtype=np.float32)
        )

    def cached_galaxy_ids(self):
        return sorted(
            int(p.stem.rsplit("_", 1)[1]) for p in self.dir.glob("draws_galaxy_*.npz")
        )

    # -- provenance -------------------------------------------------------------------
    def _write_meta(self, payload):
        self.meta_path.write_text(json.dumps({
            "cache_version": CACHE_VERSION,
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "key": self.key_parts,
            "arrays": {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)}
                       for k, v in sorted(payload.items())},
        }, indent=2) + "\n")

    def describe(self):
        if not self.meta_path.exists():
            return f"no cache entry at {self.dir}"
        meta = json.loads(self.meta_path.read_text())
        n_draws = len(self.cached_galaxy_ids())
        return (f"cache {self.dir.name}: {len(meta['arrays'])} arrays, "
                f"{n_draws} per-galaxy draw files, written {meta['written']}")


def add_cache_arguments(parser, default_root=None):
    """The two flags every cache-aware figure script takes."""
    parser.add_argument(
        "--cache-dir", default=default_root,
        help="directory holding reduced figure data (default: $STARS2DM_CACHE_ROOT)")
    parser.add_argument(
        "--cache-mode", default="auto", choices=list(MODES),
        help="auto: use the cache if present, else build it. read: require it. "
             "write: rebuild it. off: ignore it.")
