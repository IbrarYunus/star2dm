#!/usr/bin/env python3
"""Download the data tiers this repository does not carry, and verify them.

The repository holds code and the 43 published figures. Everything else -- the maps, the
two checkpoints, the reduced figure cache, the raw prediction draws -- lives in an archive,
because the smallest of them is larger than the whole source tree and the largest is 23 GB.

    python scripts/fetch_data.py --list                 # tiers, sizes, and what each unlocks
    python scripts/fetch_data.py --tier maps --tier cache
    python scripts/fetch_data.py --verify               # checksum what is already on disk

Pick the smallest tier that answers your question; `--list` says which is which. Files land
under the STARS2DM_* roots (python -m stars2dm_paths shows where), so setting those before
running puts the data straight onto a scratch filesystem rather than beside the source.

Every file is checksummed after download. A truncated or corrupted map is otherwise a very
quiet failure: the arrays still load, the figures still draw, and the numbers are wrong.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import stars2dm_paths  # noqa: E402

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "MANIFEST.yaml"
PLACEHOLDER = "REPLACE-ME"
CHUNK = 1 << 20

# Which STARS2DM root each manifest path is relative to.
ROOTS = {
    "dreams": lambda: stars2dm_paths.DREAMS_ROOT,
    "tng": lambda: stars2dm_paths.TNG_ROOT,
    "fire": lambda: stars2dm_paths.FIRE_ROOT,
    "pred": lambda: stars2dm_paths.PRED_ROOT,
    "checkpoints": lambda: stars2dm_paths.CKPT_ROOT,
    "cache": lambda: stars2dm_paths.CACHE_ROOT,
}


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


def load_manifest():
    if not MANIFEST.exists():
        raise SystemExit(f"No manifest at {MANIFEST}")
    return yaml.safe_load(MANIFEST.read_text())


def target_path(entry):
    return ROOTS[entry["root"]]() / entry["path"]


def check_base_url(manifest):
    base = manifest.get("archive_base_url", "")
    if PLACEHOLDER in base:
        raise SystemExit(
            f"The archive is not published yet: archive_base_url in {MANIFEST} is still a\n"
            f"placeholder ({base}).\n\n"
            f"If you are the author: upload the tiers, then set that one line to the base URL.\n"
            f"If you are not: the paper's data link is the place to look -- this repository\n"
            f"was published before the archive URL was assigned.\n\n"
            f"--verify and --list work without it.")
    return base.rstrip("/")


def cmd_list(manifest):
    print(f"archive: {manifest.get('archive_base_url')}\n")
    for name, tier in manifest["tiers"].items():
        total = sum(f["bytes"] for f in tier["files"])
        have = sum(1 for f in tier["files"] if target_path(f).exists())
        print(f"  {name:<13} {human(total):>9}  {len(tier['files']):>5} files  "
              f"{have} present")
        for line in tier["description"].strip().splitlines():
            print(f"                {line}")
        print()


def cmd_verify(manifest, tiers):
    bad = 0
    for name in tiers:
        for entry in manifest["tiers"][name]["files"]:
            path = target_path(entry)
            if not path.exists():
                print(f"  absent    {name}/{entry['path']}")
                continue
            size = path.stat().st_size
            if size != entry["bytes"]:
                print(f"  SIZE      {name}/{entry['path']}: {size} on disk, "
                      f"{entry['bytes']} expected")
                bad += 1
                continue
            if entry["sha256"].startswith(PLACEHOLDER):
                print(f"  no-hash   {name}/{entry['path']} (manifest has no checksum yet)")
                continue
            actual = sha256_file(path)
            if actual != entry["sha256"]:
                print(f"  CHECKSUM  {name}/{entry['path']}: {actual[:16]}... on disk, "
                      f"{entry['sha256'][:16]}... expected")
                bad += 1
            else:
                print(f"  ok        {name}/{entry['path']}")
    if bad:
        raise SystemExit(f"\n{bad} file(s) failed verification. Re-download them; a corrupt "
                         f"map loads cleanly and silently changes every number built on it.")
    print("\nall present files verified")


def cmd_fetch(manifest, tiers, base):
    for name in tiers:
        tier = manifest["tiers"][name]
        print(f"=== {name} ===")
        for entry in tier["files"]:
            path = target_path(entry)
            if path.exists() and path.stat().st_size == entry["bytes"]:
                print(f"  have  {entry['path']}")
                continue
            url = f"{base}/{entry['url']}"
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  get   {entry['path']}  ({human(entry['bytes'])})")
            # Download beside the target, then move: an interrupted fetch leaves a .part
            # file rather than a short file that looks complete on the next run.
            part = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, part)
            if not entry["sha256"].startswith(PLACEHOLDER):
                actual = sha256_file(part)
                if actual != entry["sha256"]:
                    part.unlink()
                    raise SystemExit(
                        f"Checksum mismatch for {entry['path']}:\n"
                        f"  got      {actual}\n  expected {entry['sha256']}\n"
                        f"Download discarded.")
            part.rename(path)
    print("\ndone")


def main():
    manifest = load_manifest()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", action="append", choices=list(manifest["tiers"]),
                    help="tier to fetch; repeatable")
    ap.add_argument("--all", action="store_true", help="every tier, including the 23 GB one")
    ap.add_argument("--list", action="store_true", help="report tiers and exit")
    ap.add_argument("--verify", action="store_true",
                    help="checksum what is on disk; downloads nothing")
    args = ap.parse_args()

    if args.list:
        cmd_list(manifest)
        return
    tiers = list(manifest["tiers"]) if args.all else (args.tier or [])
    if not tiers:
        raise SystemExit("Nothing to do: pass --tier NAME, --all, --list or --verify.")
    if args.verify:
        cmd_verify(manifest, tiers)
        return
    cmd_fetch(manifest, tiers, check_base_url(manifest))


if __name__ == "__main__":
    main()
