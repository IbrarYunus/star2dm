#!/usr/bin/env python3
"""Copy just the figures the draft actually includes into a clean directory.

``figures_pdf/`` holds every figure the plotting scripts emit -- 180-odd files, of which
the paper uses about 35. Handing that whole directory to Overleaf means uploading five
times more than is needed and leaves no way to tell a live figure from an abandoned one.

This walks ``main_v2.tex``, keeps the ``\\includegraphics`` paths that are not commented
out, and copies those files (only those) to the destination.

    python collect_paper_figures.py \\
        --tex ../docs/image_to_dark_matter_draft_source_updated/main_v2.tex \\
        --src ../docs/image_to_dark_matter_draft_source_updated/figures_pdf \\
        --dst ../docs/image_to_dark_matter_draft_source_updated/figures_paper

File names are preserved, so switching the draft over is a single substitution of the
directory component. Anything referenced but missing is reported and makes the run fail,
because a silently absent figure turns into a LaTeX error hundreds of lines later.
"""

import argparse
import re
import shutil
from pathlib import Path

# \includegraphics, optionally with an option group, then the braced path.
INCLUDE_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")

# An unescaped % starts a LaTeX comment. \% is a literal percent sign and does not.
COMMENT_RE = re.compile(r"(?<!\\)%")


def strip_comment(line):
    """Everything on the line before an unescaped %."""
    m = COMMENT_RE.search(line)
    return line if m is None else line[: m.start()]


def referenced_figures(tex_path):
    """Included graphics paths, in order of first appearance, comments excluded.

    Returns (live, commented). The commented list is reported rather than dropped: a
    reference that was commented out on purpose looks exactly like one commented out by
    accident, and only the author can tell them apart.
    """
    live, commented, seen = [], [], set()
    for raw in Path(tex_path).read_text().splitlines():
        code = strip_comment(raw)
        for path in INCLUDE_RE.findall(code):
            path = path.strip()
            if path not in seen:
                seen.add(path)
                live.append(path)
        # Whatever the comment hid, for the report only.
        hidden = raw[len(code):]
        for path in INCLUDE_RE.findall(hidden):
            path = path.strip()
            if path not in commented:
                commented.append(path)
    return live, commented


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", required=True, help="the .tex file to read \\includegraphics from")
    ap.add_argument("--src", required=True, help="directory holding the generated figures")
    ap.add_argument("--dst", required=True, help="directory to populate (created if absent)")
    ap.add_argument("--ref-dir", default=None,
                    help="directory component the .tex writes, when it differs from the "
                         "basename of --src (e.g. --ref-dir figures_pdf --src "
                         "figures_pdf_nogrid to collect the no-grid build of the same "
                         "figure list). Defaults to the basename of --src.")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra file names in --src to copy even though the .tex has no "
                         "reference yet (e.g. schematic.pdf before its figure* block "
                         "is pasted in)")
    ap.add_argument("--dry-run", action="store_true", help="report without copying")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")

    live, commented = referenced_figures(args.tex)

    # Only the basename matters: the .tex writes figures_pdf/foo.pdf, and the point of
    # this script is to change that directory component. References pointing at some other
    # directory are listed separately rather than hunted for.
    ref_dir = args.ref_dir or src.name
    wanted, elsewhere = [], []
    for ref in live:
        if Path(ref).parent.name == ref_dir:
            wanted.append(Path(ref).name)
        else:
            elsewhere.append(ref)

    for name in args.extra:
        if name not in wanted:
            wanted.append(name)

    present = [n for n in wanted if (src / n).is_file()]
    missing = [n for n in wanted if not (src / n).is_file()]

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        for name in present:
            shutil.copy2(src / name, dst / name)

    total_src = len(list(src.glob("*.pdf"))) + len(list(src.glob("*.png")))
    verb = "would copy" if args.dry_run else "copied"
    print(f"{args.tex}: {len(live)} live \\includegraphics, {len(commented)} commented out")
    if ref_dir != src.name:
        print(f"resolving {ref_dir}/ references against {src}")
    print(f"{src} holds {total_src} figures; {verb} {len(present)} to {dst}")

    if elsewhere:
        print(f"\n{len(elsewhere)} live reference(s) NOT under {ref_dir}/ -- left alone:")
        for ref in elsewhere:
            print(f"  {ref}")
    if commented:
        print(f"\n{len(commented)} commented-out reference(s), not copied:")
        for ref in commented:
            print(f"  {ref}")
    if missing:
        print(f"\n{len(missing)} referenced file(s) MISSING from {src}:")
        for name in missing:
            print(f"  {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
