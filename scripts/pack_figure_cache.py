#!/usr/bin/env python3
"""Package the reduced figure-data cache for upload, and report what is in it.

The cache is built as a side effect of running the figure stages once with the draws
present (scripts/07_figures.sh, CACHE_MODE=auto or write). This packs the result into a
single archive, with a checksum, so it can be published alongside the paper and fetched by
scripts/fetch_data.py instead of the ~15 GB of raw prediction draws.

    python scripts/pack_figure_cache.py --list                 # what is cached, and how big
    python scripts/pack_figure_cache.py --out dist/            # write the archive

Run --list before packing. It prints one line per entry with the key that produced it, and
the figure set each entry belongs to should be recognisable from that line. An entry you
cannot account for is a sign that a figure stage ran with settings that differ from the
published ones -- the key includes the combiner settings precisely so that shows up here
rather than as a quietly different number in a regenerated panel.
"""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import stars2dm_paths  # noqa: E402

CHUNK = 1 << 20


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def entries(cache_root):
    for meta_path in sorted(cache_root.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        size = sum(p.stat().st_size for p in meta_path.parent.iterdir())
        yield meta_path.parent, meta, size


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default=str(stars2dm_paths.CACHE_ROOT))
    ap.add_argument("--out", default=None,
                    help="directory to write stars2dm-figure-cache.tar.gz into")
    ap.add_argument("--list", action="store_true", help="report contents and exit")
    args = ap.parse_args()

    cache_root = Path(args.cache_dir)
    if not cache_root.is_dir():
        raise SystemExit(
            f"No cache at {cache_root}. Build it by running the figure stages with draws "
            f"present:  CACHE_MODE=write sbatch --export=ALL,STAGE=dreams scripts/07_figures.sh")

    found = list(entries(cache_root))
    if not found:
        raise SystemExit(f"No cache entries under {cache_root} (no */meta.json).")

    total = 0
    for path, meta, size in found:
        total += size
        key = meta["key"]
        summary = " ".join(f"{k}={v}" for k, v in key.items() if v not in (None, "None"))
        n_draws = len(list(path.glob("draws_galaxy_*.npz")))
        # The digest is printed because two entries can have identical-looking summaries --
        # a key part added later, or one whose value does not fit the line. The digest is
        # what actually distinguishes them, and a duplicate summary with two digests is the
        # signature of a stale entry left behind by a changed key.
        print(f"  {path.name.rsplit('__', 1)[1]}  {human(size):>9}  {len(meta['arrays']):>3} arrays  "
              f"{n_draws:>2} galaxy-draw files  {summary}")
    print(f"\n{len(found)} entries, {human(total)} total")

    if args.list or args.out is None:
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "stars2dm-figure-cache.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path, _, _ in found:
            tar.add(path, arcname=path.name)
    digest = sha256_file(archive)
    (out_dir / "stars2dm-figure-cache.sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"\nwrote {archive} ({human(archive.stat().st_size)})")
    print(f"sha256 {digest}")
    print("Add both to data/MANIFEST.yaml before publishing.")


if __name__ == "__main__":
    main()
