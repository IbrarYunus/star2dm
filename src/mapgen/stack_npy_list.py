#!/usr/bin/env python3
"""
Stack multiple npy files along the first axis (n_galaxy dimension).

Reads a txt file where each line is either:
  - One column: path to an npy file (entire array stacked), or
  - Three or more columns: path, start_row, end_row — only rows [start_row:end_row)
    along axis 0 are taken from that file (NumPy slice semantics; end exclusive).

Blank lines and # comments are ignored.

Usage:
  python stack_npy_list.py --list npy_list.txt --output stacked.npy
"""

import argparse
import numpy as np
import os


def parse_list_line(line):
    """
    Split a non-empty, non-comment line into (path, row_start, row_end).
    row_start/row_end are None when the line has a single column (use full array).
    """
    parts = line.split()
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        raise ValueError(
            f"Expected 1 column (path only) or 3+ columns (path start_row end_row); got 2: {line!r}"
        )
    path = parts[0]
    try:
        row_start = int(parts[1])
        row_end = int(parts[2])
    except ValueError as e:
        raise ValueError(f"start_row and end_row must be integers: {line!r}") from e
    return path, row_start, row_end


def main():
    parser = argparse.ArgumentParser(
        description="Stack npy files along axis 0 from a list file."
    )
    parser.add_argument(
        "--list", "-l",
        required=True,
        help=(
            "Path to txt file: one path per line, or path + start_row + end_row (whitespace-separated) "
            "to stack only rows [start:end) along axis 0; blank lines and # comments ignored."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the output stacked npy file.",
    )
    args = parser.parse_args()

    # Read list of npy paths (optional row slice per line)
    with open(args.list, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not raw_lines:
        raise SystemExit("No file paths in list file.")

    entries = []
    for raw in raw_lines:
        try:
            entries.append(parse_list_line(raw))
        except ValueError as e:
            raise SystemExit(str(e)) from e

    ref_height, ref_width = None, None
    arrays = []

    for path, row_start, row_end in entries:
        if not os.path.isfile(path):
            print(f"Warning: skipping missing file: {path}")
            continue
        arr = np.load(path)
        if arr.ndim != 3:
            print(f"Warning: skipping {path} (expected 3D, got shape {arr.shape})")
            continue
        n, h, w = arr.shape
        if row_start is not None:
            arr = arr[row_start:row_end]
            if arr.size == 0:
                print(f"Warning: skipping {path} (empty slice rows [{row_start}:{row_end}] from n={n})")
                continue
        if ref_height is None:
            ref_height, ref_width = h, w
        if h != ref_height or w != ref_width:
            print(f"Warning: skipping {path} (shape {(n, h, w)} vs reference {ref_height}x{ref_width})")
            continue
        arrays.append(arr)
        if row_start is None:
            print(f"  {path}: shape {arr.shape}")
        else:
            print(f"  {path}: rows [{row_start}:{row_end}] -> shape {arr.shape}")

    if not arrays:
        raise SystemExit("No valid arrays to stack.")

    stacked = np.concatenate(arrays, axis=0)
    np.save(args.output, stacked)
    print(f"\nStacked {len(arrays)} files -> {args.output}")
    print(f"  Shape: {stacked.shape} (n_galaxies={stacked.shape[0]}, height={stacked.shape[1]}, width={stacked.shape[2]})")
    print(f"  Dtype: {stacked.dtype}")


if __name__ == "__main__":
    main()
