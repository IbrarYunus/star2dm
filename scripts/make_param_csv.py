"""Generate parameter-sweep config CSVs for batch_plot_pred_param_sweep.py."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from baryonbridge.utils.galaxy_info import load_dreams_parameters

PARAM_NAMES = ["Om", "s8", "SN1", "SN2", "BHFF"]
FLAT_PRIOR_LINEAR_PARAMS = ["Om", "s8"]
FLAT_PRIOR_LOG_BOUNDS = {
    "SN1": (0.25, 4.0),
    "SN2": (0.5, 2.0),
    "BHFF": (0.25, 4.0),
}


def parse_flat_prior_params(param_list):
    """Validate --flat-prior-params; default is all five parameters."""
    if param_list is None:
        return list(PARAM_NAMES)
    if not param_list:
        raise ValueError("Provide at least one parameter name for --flat-prior-params.")

    parsed = []
    for param in param_list:
        if param not in PARAM_NAMES:
            raise ValueError(
                f"Unknown flat-prior parameter {param!r}. Expected one of {PARAM_NAMES}."
            )
        if param not in parsed:
            parsed.append(param)
    return parsed


def parse_vary_specs(vary_args):
    """Parse repeated --vary PARAM V1 V2 ... arguments."""
    if not vary_args:
        raise ValueError("Provide at least one --vary PARAM V1 [V2 ...] group.")

    specs = []
    for group in vary_args:
        if len(group) < 2:
            raise ValueError(
                f"Each --vary group needs a parameter name and values, got: {group}"
            )
        param = group[0]
        if param not in PARAM_NAMES:
            raise ValueError(
                f"Unknown parameter {param!r}. Expected one of {PARAM_NAMES}."
            )
        values = [float(v) for v in group[1:]]
        specs.append((param, values))
    return specs


def get_dreams_param_bounds(data_root, suite, dataset):
    """Min/max of each parameter across the DREAMS parameter table."""
    params = load_dreams_parameters(root=data_root, suite=suite, dataset=dataset)
    bounds = {}
    for name in PARAM_NAMES:
        bounds[name] = (float(params[name].min()), float(params[name].max()))
    return bounds, len(params)


def get_flat_prior_bounds(data_root, suite, dataset):
    """Bounds and sampling space for the joint flat-prior mode."""
    dreams_bounds, n_galaxies = get_dreams_param_bounds(data_root, suite, dataset)
    bounds = {}
    sampling_space = {}

    for name in FLAT_PRIOR_LINEAR_PARAMS:
        bounds[name] = dreams_bounds[name]
        sampling_space[name] = "linear"

    for name, (lo, hi) in FLAT_PRIOR_LOG_BOUNDS.items():
        dreams_lo, dreams_hi = dreams_bounds[name]
        clipped_lo = max(lo, dreams_lo)
        clipped_hi = min(hi, dreams_hi)
        if clipped_lo >= clipped_hi:
            raise ValueError(
                f"Flat-prior range for {name} [{lo}, {hi}] has no overlap with "
                f"DREAMS range [{dreams_lo:g}, {dreams_hi:g}]."
            )
        bounds[name] = (clipped_lo, clipped_hi)
        sampling_space[name] = "log"

    return bounds, sampling_space, n_galaxies


def sample_flat_prior_value(rng, lo, hi, sampling_space):
    if sampling_space == "linear":
        return float(rng.uniform(lo, hi))
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def build_config_rows(vary_specs, include_baseline, label_prefix):
    """Build config rows as a Cartesian product over varied parameters."""
    rows = []
    param_names = [name for name, _ in vary_specs]
    value_lists = [values for _, values in vary_specs]

    for combo_idx, combo in enumerate(itertools.product(*value_lists)):
        row = {name: np.nan for name in PARAM_NAMES}
        label_parts = []
        for param, value in zip(param_names, combo):
            row[param] = value
            label_parts.append(f"{param}={value:g}")
        rows.append(
            {
                "config_id": combo_idx,
                "label": f"{label_prefix}{'_'.join(label_parts)}",
                **row,
            }
        )

    if include_baseline:
        baseline_row = {
            "config_id": 0,
            "label": f"{label_prefix}baseline_true",
            **{name: np.nan for name in PARAM_NAMES},
        }
        for row in rows:
            row["config_id"] += 1
        rows.insert(0, baseline_row)

    return rows


def build_flat_prior_config_rows(
    n_samples,
    bounds,
    sampling_space,
    flat_prior_params,
    seed,
    include_baseline,
    label_prefix,
):
    """Draw joint flat-prior samples for selected params; others stay NaN."""
    if n_samples < 1:
        raise ValueError("--flat-prior-samples must be at least 1.")

    rng = np.random.default_rng(seed)

    rows = []
    for sample_idx in range(n_samples):
        row = {name: np.nan for name in PARAM_NAMES}
        for name in flat_prior_params:
            lo, hi = bounds[name]
            row[name] = sample_flat_prior_value(
                rng, lo, hi, sampling_space[name]
            )
        label_suffix = "_".join(flat_prior_params)
        rows.append(
            {
                "config_id": sample_idx,
                "label": f"{label_prefix}flat_prior_{label_suffix}_{sample_idx:04d}",
                **row,
            }
        )

    if include_baseline:
        baseline_row = {
            "config_id": 0,
            "label": f"{label_prefix}baseline_true",
            **{name: np.nan for name in PARAM_NAMES},
        }
        for row in rows:
            row["config_id"] += 1
        rows.insert(0, baseline_row)

    return rows


def write_bounds_table(bounds, sampling_space, output_path, flat_prior_params):
    rows = []
    flat_prior_set = set(flat_prior_params)
    for name in PARAM_NAMES:
        if name in flat_prior_set:
            lo, hi = bounds[name]
            rows.append(
                {
                    "parameter": name,
                    "min": lo,
                    "max": hi,
                    "sampling_space": sampling_space[name],
                }
            )
        else:
            rows.append(
                {
                    "parameter": name,
                    "min": np.nan,
                    "max": np.nan,
                    "sampling_space": "galaxy_true",
                }
            )
    bounds_df = pd.DataFrame(rows)
    bounds_path = output_path.with_name(f"{output_path.stem}_bounds.csv")
    bounds_df.to_csv(bounds_path, index=False)
    return bounds_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a parameter-sweep config CSV. Grid mode (--vary) overrides "
            "only listed parameters; flat-prior mode draws randomized parameter "
            "sets for a chosen subset (default: all five). Om/s8 use linear-uniform "
            "in DREAMS bounds; SN1/SN2/BHFF use log-uniform in [0.25,4], [0.5,2], "
            "and [0.25,4]. Unsampled parameters are NaN (use per-galaxy true values)."
        )
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--vary",
        nargs="+",
        action="append",
        metavar=("PARAM", "VALUE"),
        help=(
            "Parameter sweep group. Repeat for grids, e.g. "
            "'--vary Om 0.25 0.30 0.35 --vary s8 0.70 0.80'. "
            "Mutually exclusive with --flat-prior-samples."
        ),
    )
    parser.add_argument(
        "--flat-prior-samples",
        type=int,
        default=None,
        help=(
            "Number of jointly randomized parameter configs. By default all five "
            "parameters are sampled; use --flat-prior-params to restrict (e.g. "
            "SN1 SN2 BHFF for FIRE with fixed Om/s8)."
        ),
    )
    parser.add_argument(
        "--flat-prior-params",
        nargs="+",
        default=None,
        metavar="PARAM",
        help=(
            "Parameters to sample in flat-prior mode. Others are left NaN so "
            "batch_plot_pred_param_sweep uses each galaxy's true value. "
            "Example for FIRE: --flat-prior-params SN1 SN2 BHFF"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/DREAMS_data",
        help="DREAMS data root used to infer parameter bounds.",
    )
    parser.add_argument("--suite", type=str, default="TNG")
    parser.add_argument("--dataset", type=str, default="SB5")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --flat-prior-samples.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Add a baseline row with all NaN (use true per-galaxy parameters).",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="",
        help="Optional prefix for auto-generated labels.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.flat_prior_samples is not None:
        if args.vary:
            raise ValueError("Use either --flat-prior-samples or --vary, not both.")
        flat_prior_params = parse_flat_prior_params(args.flat_prior_params)
        bounds, sampling_space, n_galaxies = get_flat_prior_bounds(
            args.data_root, args.suite, args.dataset
        )
        bounds_path = write_bounds_table(
            bounds, sampling_space, output_path, flat_prior_params
        )
        rows = build_flat_prior_config_rows(
            n_samples=args.flat_prior_samples,
            bounds=bounds,
            sampling_space=sampling_space,
            flat_prior_params=flat_prior_params,
            seed=args.seed,
            include_baseline=args.include_baseline,
            label_prefix=args.label_prefix,
        )
        print(
            f"Flat-prior sampling: {args.flat_prior_samples} joint draws over "
            f"{flat_prior_params} ({args.suite}/{args.dataset}, n={n_galaxies})"
        )
        print(f"Saved bounds table to {bounds_path}")
        for name in PARAM_NAMES:
            if name in flat_prior_params:
                lo, hi = bounds[name]
                space = sampling_space[name]
                print(f"  {name}: [{lo:g}, {hi:g}] ({space})")
            else:
                print(f"  {name}: galaxy_true (NaN in config CSV)")
    else:
        vary_specs = parse_vary_specs(args.vary)
        rows = build_config_rows(
            vary_specs=vary_specs,
            include_baseline=args.include_baseline,
            label_prefix=args.label_prefix,
        )

    columns = ["config_id", "label"] + PARAM_NAMES
    df = pd.DataFrame(rows)[columns]
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} config rows to {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
