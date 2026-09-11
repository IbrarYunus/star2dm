#!/usr/bin/env python3
"""Assemble the upload directory for the data archive, in the layout MANIFEST.yaml expects.

    python scripts/stage_archive.py --out dist/ --tier maps --tier checkpoints
    python scripts/stage_archive.py --out dist/ --tier predictions       # ~23 GB, slow
    python scripts/stage_archive.py --out dist/ --all --dry-run

Each tier is staged under `--out` at exactly the `url` path the manifest gives it, so the
whole directory can be uploaded verbatim and `archive_base_url` set to wherever it lands.
Nothing else in the manifest then needs to change.

Files are hard-linked where the filesystem allows it and copied otherwise, so staging 23 GB
usually costs no additional space. `--dry-run` reports the plan without touching anything.

After staging, the tiers built here (cache, predictions) need their bytes and sha256 filled
into MANIFEST.yaml -- this script prints both. The maps and checkpoints are already hashed in
the manifest, and staging re-verifies them against it rather than trusting the copy.
"""

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import stars2dm_paths  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "MANIFEST.yaml"
CHUNK = 1 << 22

ROOTS = {
    "dreams": lambda: stars2dm_paths.DREAMS_ROOT,
    "tng": lambda: stars2dm_paths.TNG_ROOT,
    "fire": lambda: stars2dm_paths.FIRE_ROOT,
    "pred": lambda: stars2dm_paths.PRED_ROOT,
    "checkpoints": lambda: stars2dm_paths.CKPT_ROOT,
    "cache": lambda: stars2dm_paths.CACHE_ROOT,
}

# Which prediction files belong in the published tier. Everything else under the prediction
# root is from an earlier model version or an abandoned experiment and is not shipped.
PREDICTION_PATTERNS = [
    "*_v95_pred_draws_100_heun100_n100.npz",
    "*_v95_pred_draws_10_heun100.npz",
    "*_v95_pred_draws_50_heun100_*_sweep_5.npz",
    "*_v97_pred_draws_100_heun100_n100.npz",
    "*_v97_pred_draws_10_heun100.npz",
    "*_v95_heun100*_radial_profile.npz",
    "*_v97_heun100*_radial_profile.npz",
    "*TNG50_v95_heun100_n100_pred_draws_100.npz",
    "*TNG50_v97_heun100_pred_draws_10.npz",
    "*FIREm12_v95_pred_draws_100_heun100_FIREm12_flat_prior_baryon_100.npz",
    "*FIREm12_v97_heun100_n100_pred_draws_100.npz",
]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def place(src, dst, dry):
    """Hard-link if possible, else copy. Same bytes either way."""
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if src.stat().st_dev == os.stat(dst.parent).st_dev:
        os.link(src, dst)
    else:
        dst.write_bytes(src.read_bytes())


def stage_listed(manifest, tier_name, out, dry, verify):
    """maps and checkpoints: individual files, already hashed in the manifest."""
    total, bad = 0, 0
    for entry in manifest["tiers"][tier_name]["files"]:
        src = ROOTS[entry["root"]]() / entry["path"]
        if not src.exists():
            print(f"  MISSING  {src}")
            bad += 1
            continue
        size = src.stat().st_size
        if size != entry["bytes"]:
            print(f"  SIZE     {entry['path']}: {size} != {entry['bytes']} in manifest")
            bad += 1
            continue
        if verify and sha256_file(src) != entry["sha256"]:
            print(f"  CHECKSUM {entry['path']} does not match the manifest")
            bad += 1
            continue
        place(src, out / entry["url"], dry)
        total += size
        print(f"  {human(size):>9}  {entry['url']}")
    if bad:
        raise SystemExit(f"{bad} file(s) failed; nothing uploaded from this tier.")
    return total


def stage_cache(out, dry):
    """The figure cache, as one tarball."""
    root = stars2dm_paths.CACHE_ROOT
    entries = sorted(p.parent for p in root.glob("*/meta.json"))
    if not entries:
        raise SystemExit(
            f"No cache entries under {root}. Build the cache first by running the figure "
            f"stages with draws present:\n"
            f"  CACHE_MODE=write bash scripts/submit_all.sh figures")
    print(f"  {len(entries)} cache entries")
    dest = out / "cache" / "stars2dm-figure-cache.tar.gz"
    if dry:
        return sum(f.stat().st_size for e in entries for f in e.iterdir())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        for entry in entries:
            tar.add(entry, arcname=entry.name)
    size = dest.stat().st_size
    print(f"  {human(size):>9}  cache/stars2dm-figure-cache.tar.gz")
    print(f"  sha256 {sha256_file(dest)}")
    return size


def stage_predictions(out, dry):
    """The prediction draws the paper uses, as one uncompressed tar.

    Uncompressed on purpose: npz members are already deflated, so gzip buys almost nothing
    and costs hours on 23 GB.
    """
    root = stars2dm_paths.PRED_ROOT
    if not root.is_dir():
        raise SystemExit(f"No prediction tree at {root}")
    members, total = [], 0
    for pattern in PREDICTION_PATTERNS:
        hits = sorted(root.glob(f"**/{pattern}"))
        n = len(hits)
        s = sum(h.stat().st_size for h in hits)
        total += s
        members.extend(hits)
        print(f"  {n:>6} files  {human(s):>9}  {pattern}")
    print(f"  {len(members):>6} files  {human(total):>9}  TOTAL")
    if dry:
        return total
    dest = out / "predictions" / "stars2dm-predictions.tar"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w") as tar:
        for path in members:
            tar.add(path, arcname=str(path.relative_to(root)))
    size = dest.stat().st_size
    print(f"  {human(size):>9}  predictions/stars2dm-predictions.tar")
    print(f"  sha256 {sha256_file(dest)}")
    return size


def main():
    manifest = yaml.safe_load(MANIFEST.read_text())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist", help="staging directory (default: dist/)")
    ap.add_argument("--tier", action="append", choices=list(manifest["tiers"]))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, touch nothing")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip re-checksumming maps and checkpoints against the manifest")
    args = ap.parse_args()

    tiers = list(manifest["tiers"]) if args.all else (args.tier or [])
    if not tiers:
        raise SystemExit("Nothing to do: pass --tier NAME or --all.")
    out = Path(args.out)
    grand = 0
    for name in tiers:
        print(f"=== {name} ===")
        if name in ("maps", "checkpoints"):
            grand += stage_listed(manifest, name, out, args.dry_run, not args.no_verify)
        elif name == "cache":
            grand += stage_cache(out, args.dry_run)
        elif name == "predictions":
            grand += stage_predictions(out, args.dry_run)
        print()
    print(f"{'would stage' if args.dry_run else 'staged'} {human(grand)} into {out}/")
    if not args.dry_run:
        print("\nNext: upload the contents of that directory, then set archive_base_url in\n"
              "data/MANIFEST.yaml, and fill in bytes/sha256 for the cache and prediction\n"
              "tarballs from the lines above.")


if __name__ == "__main__":
    main()
