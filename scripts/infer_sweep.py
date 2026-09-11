"""Batch predictions over cosmological parameter configs with stacked MC draws."""

import argparse
import importlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baryonbridge.config import GRID_SUFFIX, RESOLUTION
from baryonbridge.interpolants.stochastic import LightningStochasticInterpolant

from batch_plot_pred_uncertainty import custom_plot_evals, plot_uncertainty_maps

PARAM_NAMES = ["Om", "s8", "SN1", "SN2", "BHFF"]


def _dreams_kwargs(args):
    return {"suite": args.suite, "dataset": args.dataset}


def _tng_kwargs(args):
    return {"tng_run": args.data_source}


def _fire_kwargs(args):
    return {"fire_run": args.data_source}


DATA_SOURCE_REGISTRY = {
    "DREAMS": (
        ("baryonbridge.data.dreams", "DREAMS"),
        "./data/DREAMS_data",
        _dreams_kwargs,
    ),
    "TNG50": (
        ("baryonbridge.data.tng", "TNG"),
        "./data/TNG_data",
        _tng_kwargs,
    ),
    "TNG100": (
        ("baryonbridge.data.tng", "TNG"),
        "./data/TNG_data",
        _tng_kwargs,
    ),
    "FIREm12": (
        ("baryonbridge.data.fire", "FIRE"),
        "./data/FIRE_data",
        _fire_kwargs,
    ),
    "FIREm11": (
        ("baryonbridge.data.fire", "FIRE"),
        "./data/FIRE_data",
        _fire_kwargs,
    ),
}


def get_dataset_and_norm(args):
    """Build dataset and norm dict from args.data_source."""
    data_source = args.data_source
    entry = DATA_SOURCE_REGISTRY.get(data_source)
    if entry is None:
        raise ValueError(
            f"Unknown --data-source '{data_source}'. "
            f"Choices: {list(DATA_SOURCE_REGISTRY.keys())}"
        )
    (module_path, class_name), default_root, kwargs_fn = entry
    mod = importlib.import_module(module_path)
    dataset_class = getattr(mod, class_name)
    norm_dict = getattr(mod, "NORM_DICT")

    root = getattr(args, "data_root", None) or default_root
    resolution = RESOLUTION[args.version]
    file_suffix = GRID_SUFFIX[args.version][1:]
    idx_list = list(range(args.start_idx, args.end_idx))

    common = dict(
        root=root,
        dims=args.dims,
        resolution=resolution,
        redshift=args.redshift,
        x1_fields=args.x1_fields,
        x_conditioning_fields=args.x_conditioning_fields,
        file_suffix=file_suffix,
        idx_list=idx_list,
        transform=None,
        randomize_theta=False,
        randomize_x_conditioning=False,
        randomize_x1=False,
    )
    extra = kwargs_fn(args) if kwargs_fn else {}
    dataset = dataset_class(**common, **extra)
    return dataset, norm_dict, data_source, root


def resolve_output_layout(output_dir, sim_label, version):
    """Match batch_plot_pred_uncertainty_da layout for OOD; flat for DREAMS."""
    if sim_label == "DREAMS":
        base_output_dir = output_dir

        def draw_stem(dataset_index):
            return f"galaxy_{dataset_index}_{version}"
    else:
        base_output_dir = os.path.join(output_dir, sim_label)

        def draw_stem(dataset_index):
            return f"galaxy_{dataset_index}_{sim_label}_{version}"

    return base_output_dir, draw_stem


def normalize_output_suffix(suffix):
    if suffix is None:
        return ""
    suffix = suffix.strip()
    if not suffix:
        return ""
    return suffix.strip("_")


def tagged_filename(basename, suffix):
    """Insert _suffix before the file extension, e.g. manifest.csv -> manifest_run1.csv."""
    tag = normalize_output_suffix(suffix)
    if not tag:
        return basename
    stem, ext = os.path.splitext(basename)
    return f"{stem}_{tag}{ext}"


def load_param_configs(config_csv):
    df = pd.read_csv(config_csv)
    if "config_id" not in df.columns:
        df["config_id"] = np.arange(len(df), dtype=int)
    for name in PARAM_NAMES:
        if name not in df.columns:
            df[name] = np.nan
    return df


def build_draw_index_manifest(config_df, num_samples):
    """One row per global draw index: config_idx * num_samples + draw_within_config."""
    rows = []
    global_draw_idx = 0
    for config_idx, config_row in config_df.reset_index(drop=True).iterrows():
        for draw_within_config in range(num_samples):
            row = {
                "global_draw_idx": global_draw_idx,
                "config_idx": int(config_idx),
                "config_id": int(config_row["config_id"]),
                "draw_within_config": draw_within_config,
            }
            if "label" in config_row and pd.notna(config_row["label"]):
                row["label"] = config_row["label"]
            for name in PARAM_NAMES:
                value = config_row[name]
                row[name] = np.nan if pd.isna(value) else float(value)
            rows.append(row)
            global_draw_idx += 1
    return pd.DataFrame(rows)


def apply_config_to_theta(true_theta, config_row):
    """Override only non-NaN config values; keep per-galaxy true values otherwise."""
    theta = true_theta.clone()
    for param_idx, name in enumerate(PARAM_NAMES):
        value = config_row[name]
        if pd.notna(value):
            theta[:, param_idx] = float(value)
    return theta


def draws_filepath(galaxy_dir, draw_stem, total_draws, output_suffix=""):
    filename = tagged_filename(
        f"{draw_stem}_pred_draws_{total_draws}.npz",
        output_suffix,
    )
    return os.path.join(galaxy_dir, filename)


def append_draws(filepath, new_draws, overwrite, is_first_config):
    new_draws_np = new_draws.numpy()
    if os.path.exists(filepath) and not (overwrite and is_first_config):
        existing = np.load(filepath)["draws"]
        combined = np.concatenate([existing, new_draws_np], axis=0)
    else:
        combined = new_draws_np
    np.savez_compressed(filepath, draws=combined)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run stochastic predictions for galaxies over parameter configs. "
            "Workflow: for each config, loop all galaxies and append num_samples "
            "draws to each galaxy file (total draws = n_configs * num_samples)."
        )
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--param-config-csv",
        type=str,
        required=True,
        help="CSV of parameter configs (only overridden columns need values).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/DREAMS_data/Pred/CDM/MW_zooms/SB5/",
    )
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--end-idx", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--version", type=str, default="v12")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--x1-fields", nargs="+", default=["Pfluc"])
    parser.add_argument(
        "--x-conditioning-fields",
        nargs="+",
        default=["Mstar", "vRavg", "vTavg", "Wavg", "Pagama"],
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="DREAMS",
        choices=list(DATA_SOURCE_REGISTRY.keys()),
        help="Which simulation data to load (DREAMS, TNG50, FIREm12, ...). "
        "Recorded in output paths and filenames for non-DREAMS sources.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory for the chosen data source.",
    )
    parser.add_argument("--suite", type=str, default="TNG")
    parser.add_argument("--dataset", type=str, default="SB5")
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Stochastic draws per galaxy per parameter config.",
    )
    parser.add_argument(
        "--config-start",
        type=int,
        default=0,
        help="First config row index (inclusive) to run.",
    )
    parser.add_argument(
        "--config-end",
        type=int,
        default=None,
        help="Last config row index (exclusive). Default: all configs.",
    )
    parser.add_argument(
        "--overwrite-draws",
        action="store_true",
        help="Replace existing draw files instead of appending.",
    )
    parser.add_argument(
        "--plot-one-per-batch",
        action="store_true",
        help="Render plots for the first galaxy in each batch (first config only).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all figure generation.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help=(
            "Optional tag appended to draw .npz files and manifest/metadata "
            "filenames, e.g. 'om_sweep' -> draw_index_manifest_om_sweep.csv."
        ),
    )
    parser.add_argument(
        "--n-sampling-steps",
        type=int,
        default=None,
        help="Override checkpoint n_sampling_steps for SDE integration.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["euler", "heun"],
        default="euler",
        help="SDE integration scheme: euler (O(dt) bias) or heun (O(dt^2), 2 model evals/step).",
    )
    args = parser.parse_args()

    output_suffix = normalize_output_suffix(args.output_suffix)

    dataset, norm_dict, sim_label, data_root = get_dataset_and_norm(args)
    norm_dict = norm_dict[f"{args.dims}D"]
    base_output_dir, draw_stem_fn = resolve_output_layout(
        args.output_dir, sim_label, args.version
    )
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Data source: {sim_label} (outputs under {base_output_dir})")

    full_config_df = load_param_configs(args.param_config_csv)
    config_end = len(full_config_df) if args.config_end is None else args.config_end
    config_df = full_config_df.iloc[args.config_start : config_end].reset_index(
        drop=True
    )
    n_configs = len(config_df)
    n_configs_full = len(full_config_df)
    total_draws = n_configs_full * args.num_samples

    manifest_path = os.path.join(
        base_output_dir, tagged_filename("draw_index_manifest.csv", output_suffix)
    )
    manifest_df = build_draw_index_manifest(full_config_df, args.num_samples)
    manifest_df.to_csv(manifest_path, index=False)

    config_copy_path = os.path.join(
        base_output_dir, tagged_filename("param_config_used.csv", output_suffix)
    )
    full_config_df.to_csv(config_copy_path, index=False)

    metadata_path = os.path.join(
        base_output_dir, tagged_filename("param_sweep_metadata.txt", output_suffix)
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"param_config_csv={args.param_config_csv}\n")
        f.write(f"data_source={sim_label}\n")
        f.write(f"data_root={data_root}\n")
        f.write(f"output_suffix={output_suffix or '(none)'}\n")
        f.write(f"config_start={args.config_start}\n")
        f.write(f"config_end={config_end}\n")
        f.write(f"n_configs_run={n_configs}\n")
        f.write(f"n_configs_full={n_configs_full}\n")
        f.write(f"num_samples={args.num_samples}\n")
        f.write(f"total_draws_per_galaxy={total_draws}\n")
        f.write(f"sampler={args.sampler}\n")
        f.write(
            f"n_sampling_steps={args.n_sampling_steps if args.n_sampling_steps is not None else '(checkpoint hparam)'}\n"
        )
        f.write(
            "draw_order=config_major: "
            "global_draw_idx = config_idx * num_samples + draw_within_config\n"
        )
        f.write(
            "stacking=each config loops all galaxies; "
            f"{args.num_samples} draws appended per galaxy per config\n"
        )

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = LightningStochasticInterpolant.load_from_checkpoint(args.checkpoint_path)
    model = model.to(args.device)
    model.eval()
    n_sampling_steps = (
        args.n_sampling_steps
        if args.n_sampling_steps is not None
        else model.hparams.n_sampling_steps
    )
    print(f"Sampler: {args.sampler}, n_sampling_steps: {n_sampling_steps}")
    condition_on_parameters = getattr(model.hparams, "condition_on_parameters", True)
    if not condition_on_parameters:
        print(
            "Note: model trained without parameter conditioning; "
            "using h=None at inference (theta overrides are unused)."
        )

    print(f"Loading data for indices: {args.start_idx}..{args.end_idx}")
    print(f"Dataset loaded with {len(dataset)} samples")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=9
    )

    render_plots = not args.no_plots

    with torch.no_grad():
        for config_idx, config_row in config_df.iterrows():
            config_label = config_row.get("label", f"config_{config_idx}")
            print(
                f"Config {config_idx}/{n_configs - 1} "
                f"(config_id={int(config_row['config_id'])}, label={config_label})"
            )

            for batch_idx, batch in enumerate(dataloader):
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(args.device)

                batch = {
                    k: v.to(args.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                true_theta = batch["theta"]
                theta = apply_config_to_theta(true_theta, config_row)
                h_cond = theta if condition_on_parameters else None
                x_conditioning = batch["x_conditioning"]
                x1 = batch["x1"]
                dataset_indices = batch["dataset_index"].cpu().numpy()
                x0 = x_conditioning[:, 0:1, ...]

                draws = []
                for _ in range(args.num_samples):
                    pred = model.stochastic_interpolant.predict(
                        x0=x0,
                        x_conditioning=x_conditioning,
                        h=h_cond,
                        n_sampling_steps=n_sampling_steps,
                        sampler=args.sampler,
                    )
                    if model.hparams.fit_residual:
                        pred = pred + x_conditioning
                    draws.append(pred.cpu())
                draws = torch.stack(draws)

                plots_rendered = 0
                for i in range(theta.shape[0]):
                    dataset_index = int(dataset_indices[i])
                    galaxy_dir = os.path.join(
                        base_output_dir, f"galaxy_{dataset_index:04d}"
                    )
                    os.makedirs(galaxy_dir, exist_ok=True)

                    file_tag = draw_stem_fn(dataset_index)
                    single_draws = draws[:, i]
                    draw_path = draws_filepath(
                        galaxy_dir,
                        file_tag,
                        total_draws,
                        output_suffix=output_suffix,
                    )
                    append_draws(
                        draw_path,
                        single_draws,
                        overwrite=args.overwrite_draws,
                        is_first_config=(config_idx == 0),
                    )

                    should_plot = (
                        render_plots
                        and config_idx == 0
                        and (
                            (not args.plot_one_per_batch)
                            or (plots_rendered == 0)
                        )
                    )
                    if not should_plot:
                        continue

                    median_pred = torch.median(single_draws, dim=0).values
                    x_conditioning_i = x_conditioning[i : i + 1]
                    x1_i = x1[i : i + 1]
                    x0_i = x_conditioning_i[:, 0:1, ...]

                    additional_conditioning = None
                    if x_conditioning_i.shape[1] > 1:
                        additional_conditioning = x_conditioning_i[0, 1:, ...].cpu()

                    x0mean = norm_dict[args.x_conditioning_fields[0]][
                        RESOLUTION[args.version]
                    ]["mean"]
                    x0std = norm_dict[args.x_conditioning_fields[0]][
                        RESOLUTION[args.version]
                    ]["std"]
                    x1mean = norm_dict[args.x1_fields[0]][RESOLUTION[args.version]][
                        "mean"
                    ]
                    x1std = norm_dict[args.x1_fields[0]][RESOLUTION[args.version]][
                        "std"
                    ]

                    if additional_conditioning is not None:
                        x_conditioning_mean = [
                            norm_dict[field][RESOLUTION[args.version]]["mean"]
                            for field in args.x_conditioning_fields[1:]
                        ]
                        x_conditioning_std = [
                            norm_dict[field][RESOLUTION[args.version]]["std"]
                            for field in args.x_conditioning_fields[1:]
                        ]
                    else:
                        x_conditioning_mean = None
                        x_conditioning_std = None

                    custom_plot_evals(
                        x0=x0_i[0].cpu().squeeze(),
                        x1=x1_i[0].cpu().squeeze(),
                        x_pred=median_pred.cpu().squeeze(),
                        save_dir=galaxy_dir,
                        galaxy_id=dataset_index,
                        x_conditioning=additional_conditioning,
                        fig_name=file_tag,
                        x0mean=x0mean,
                        x0std=x0std,
                        x1mean=x1mean,
                        x1std=x1std,
                        x_conditioning_mean=x_conditioning_mean,
                        x_conditioning_std=x_conditioning_std,
                    )
                    plot_uncertainty_maps(
                        draws=single_draws.squeeze(1),
                        save_dir=galaxy_dir,
                        galaxy_id=dataset_index,
                        fig_name=file_tag,
                        x1mean=x1mean,
                        x1std=x1std,
                    )
                    plots_rendered += 1

                if args.device.startswith("cuda") and torch.cuda.is_available():
                    current_alloc = torch.cuda.memory_allocated(args.device) / 1e9
                    peak_alloc = torch.cuda.max_memory_allocated(args.device) / 1e9
                    print(
                        f"[GPU] config {config_idx} batch {batch_idx}: "
                        f"alloc={current_alloc:.2f}GB peak={peak_alloc:.2f}GB"
                    )

    print(f"Saved draw index manifest to {manifest_path}")
    print(f"Saved config copy to {config_copy_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
