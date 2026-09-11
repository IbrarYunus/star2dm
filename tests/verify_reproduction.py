#!/usr/bin/env python3
"""Compare regenerated figures against the ones published in the paper.

    python tests/verify_reproduction.py                       # against $STARS2DM_FIG_DIR
    python tests/verify_reproduction.py --figures data/figures --dpi 150

Every figure in `reference_figures/` is the file that appears in the paper. After running the
figure stages, this says -- per panel -- whether what you produced is the same image.

It rasterizes both PDFs and compares pixels, because a PDF byte comparison is useless here:
matplotlib stamps a creation date into every file, so two runs of the same code with the same
data never produce identical bytes. Rasterizing at 100 dpi catches anything a reader would
see and ignores nothing that matters.

Exit status is 0 only if every published figure was reproduced. Anything else is a real
difference worth understanding before the repository is published -- the reduced-figure-data
path is verified pixel-for-pixel, so "close enough" is not the expected outcome.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import stars2dm_paths  # noqa: E402

# matplotlib stamps these into every PDF; they differ between runs of identical code.
VOLATILE = re.compile(rb"/(CreationDate|ModDate)\s*\([^)]*\)")


def rasterize(pdf, out_prefix, dpi):
    """PDF -> PNG via poppler. Returns the page image, or None if the render produced none."""
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(out_prefix)],
        check=True, capture_output=True)
    png = Path(f"{out_prefix}.png")
    return mpimg.imread(png) if png.exists() else None


def compare_pixels(ref, new, tmp, dpi):
    a = rasterize(ref, Path(tmp) / "ref", dpi)
    b = rasterize(new, Path(tmp) / "new", dpi)
    if a is None or b is None:
        return "render-failed", None
    if a.shape != b.shape:
        return "different-size", f"{a.shape} vs {b.shape}"
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    worst = float(diff.max())
    if worst == 0.0:
        return "identical", None
    n_diff = int((diff.max(axis=-1) > 0).sum()) if diff.ndim == 3 else int((diff > 0).sum())
    frac = n_diff / (a.shape[0] * a.shape[1])
    return "differs", f"max {worst:.4f} over {frac:.3%} of pixels"


def compare_bytes(ref, new):
    """Weaker fallback: PDF bytes with the volatile metadata removed."""
    def norm(p):
        return hashlib.sha256(VOLATILE.sub(b"", p.read_bytes())).hexdigest()
    return ("identical", None) if norm(ref) == norm(new) else ("differs", "byte comparison")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", default=str(stars2dm_paths.DATA_ROOT / "figures"),
                    help="directory holding the regenerated figures")
    ap.add_argument("--reference", default=str(REPO / "reference_figures"))
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--quiet", action="store_true", help="only report problems")
    args = ap.parse_args()

    ref_dir, new_dir = Path(args.reference), Path(args.figures)
    if not new_dir.is_dir():
        raise SystemExit(f"No regenerated figures at {new_dir}. Run the figure stages first:\n"
                         f"  bash scripts/submit_all.sh figures")

    raster = shutil.which("pdftoppm") is not None
    if not raster:
        print("pdftoppm not found; falling back to a byte comparison with the creation date\n"
              "stripped. That catches a changed figure but cannot quantify a small difference.\n"
              "Install poppler-utils for the real check.\n")

    published = sorted(ref_dir.glob("*.pdf"))
    counts, problems = {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for ref in published:
            new = new_dir / ref.name
            if not new.exists():
                verdict, detail = "missing", None
            elif raster:
                verdict, detail = compare_pixels(ref, new, tmp, args.dpi)
            else:
                verdict, detail = compare_bytes(ref, new)
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict != "identical":
                problems.append((ref.name, verdict, detail))
            if not args.quiet or verdict != "identical":
                print(f"  {verdict:<14} {ref.name}" + (f"   [{detail}]" if detail else ""))

    extra = sorted({p.name for p in new_dir.glob("*.pdf")} - {p.name for p in published})
    print(f"\n{len(published)} published figures: " +
          ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    if extra:
        print(f"{len(extra)} file(s) produced that the paper does not use "
              f"(expected: see docs/FIGURE_MAP.md): {', '.join(extra[:4])}"
              + (" ..." if len(extra) > 4 else ""))

    if problems:
        print("\nNot reproduced:")
        for name, verdict, detail in problems:
            print(f"  {verdict:<14} {name}" + (f"   [{detail}]" if detail else ""))
        raise SystemExit(1)
    print("\nEvery published figure reproduced.")


if __name__ == "__main__":
    main()
