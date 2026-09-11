import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from baryonbridge.config import (
    CLIP_PROFILE,
    GRID_SUFFIX,
    NORM_PROFILE,
    RESOLUTION,
    get_norm_profile,
    merged_norm_dict,
)
from baryonbridge.data.mwsigma import (
    DEFAULT_ROOT as MWSIGMA_DEFAULT_ROOT,
    DEFAULT_SPLIT_PATH as MWSIGMA_DEFAULT_SPLIT,
    DEFAULT_THETA_TABLE as MWSIGMA_THETA_TABLE,
    compute_theta_norm_stats as mwsigma_theta_norm_stats,
    load_split as mwsigma_load_split,
)
from baryonbridge.data.dwarffire import (
    DEFAULT_ROOT as DWARF_DEFAULT_ROOT,
    DEFAULT_SPLIT_PATH as DWARF_DEFAULT_SPLIT,
    DEFAULT_THETA_TABLE as DWARF_THETA_TABLE,
    compute_theta_norm_stats as dwarf_theta_norm_stats,
    load_split as dwarf_load_split,
)
from baryonbridge.interpolants.stochastic import LightningStochasticInterpolant


# Registry for test data loaders: add new sims here to make them selectable via --data-source.
# Each entry: ((module_path, class_name), default_root, kwargs_fn(args) -> dict).
def _dreams_kwargs(args):
    return {"suite": args.suite, "dataset": args.dataset}


def _tng_kwargs(args):
    return {"tng_run": args.data_source}  # TNG50 or TNG100


def _fire_kwargs(args):
    return {"fire_run": args.data_source}  # FIREm12 or FIREm11

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


def _dwarffire_kwargs(args):
    """
    Extra constructor kwargs for the dwarf FIRE dataset.

    theta has to be standardized at inference with the SAME statistics as at training,
    i.e. the train split's -- not the statistics of whichever galaxies are being
    predicted. DwarfFIREDataModule.setup() does this once and passes them down; this
    reproduces it, and DwarfFIRE raises rather than silently deriving its own.
    """
    root = getattr(args, "data_root", None) or DWARF_DEFAULT_ROOT
    split_path = args.split_file or DWARF_DEFAULT_SPLIT
    splits = dwarf_load_split(root, split_path)
    theta_mean, theta_std = dwarf_theta_norm_stats(root, DWARF_THETA_TABLE, splits["train"])
    return {
        "dataset": args.dataset,
        "normalize_theta": True,
        "theta_norm_mean": theta_mean,
        "theta_norm_std": theta_std,
    }


def _mwsigma_kwargs(args):
    """
    Extra constructor kwargs for the MW-zoom Sigma dataset.

    Same contract as _dwarffire_kwargs: theta must be standardized at inference with
    the TRAIN split's statistics, the ones training used, not the statistics of
    whichever galaxies are being predicted.

    Note what is deliberately NOT set here. This suite has two fields of view with
    different normalization constants (legacy32 at +/-32.596465 kpc, fov110 at +/-110),
    but file_suffix and norm_profile are already derived from the VERSION TAG by
    get_dataset_and_norm via GRID_SUFFIX/NORM_PROFILE. Passing them here as well would
    create a second place for the FOV to be decided, which is exactly how mw01's
    aperture would end up paired with mw03's constants.
    """
    root = getattr(args, "data_root", None) or MWSIGMA_DEFAULT_ROOT
    split_path = args.split_file or MWSIGMA_DEFAULT_SPLIT
    splits = mwsigma_load_split(root, split_path)
    theta_mean, theta_std = mwsigma_theta_norm_stats(
        root, MWSIGMA_THETA_TABLE, splits["train"]
    )
    return {
        "dataset": args.dataset,
        "normalize_theta": True,
        "theta_norm_mean": theta_mean,
        "theta_norm_std": theta_std,
    }


DATA_SOURCE_REGISTRY["MWSIGMA"] = (
    ("baryonbridge.data.mwsigma", "MWSigma"),
    "./data/DREAMS_data",
    _mwsigma_kwargs,
)
DATA_SOURCE_REGISTRY["DWARFFIRE"] = (
    ("baryonbridge.data.dwarffire", "DwarfFIRE"),
    "./data/DREAMS_data",
    _dwarffire_kwargs,
)
# Suites whose predictions are IN-DOMAIN for the model being run, so they get no
# sim-label subdirectory and no sim label in filenames. This keeps the dwarf runs on
# the plain `galaxy_N_dw01_...` convention, which is what pred_data_loaders.file_stem
# builds when parse_version_tag finds no OOD prefix -- so every downstream path
# resolver takes the in-domain branch with no changes and none of the
# DREAMS-vs-OOD filename-ordering ambiguity.
#
# "DREAMS" is deliberately NOT listed, even though it is also in-domain and
# pred_data_loaders would resolve it the same way: this script has always written
# DREAMS output to output_dir/DREAMS/, and changing that is not part of the dwarf
# work. No DREAMS/ directory exists under the pred root today, so nothing depends on
# either behaviour yet.
IN_DOMAIN_SOURCES = {"DWARFFIRE", "MWSIGMA"}


def _resolve_idx_list(args):
    """
    Galaxy row indices to run inference on.

    --split-file/--split-name select a named split from a permutation .npz, which is
    the only correct way to reach the dwarf test galaxies: that split is permuted, so
    a contiguous --start-idx/--end-idx range is a different set of galaxies entirely.
    --start-idx/--end-idx still slice the chosen list, so a shard of a split can be run
    per job.
    """
    if not args.split_file or not args.split_name:
        return list(range(args.start_idx, args.end_idx))

    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = Path(getattr(args, "data_root", None) or "") / split_path
    with np.load(split_path) as data:
        idx = [int(i) for i in data[f"{args.split_name}_idx"]]
    sliced = idx[args.start_idx:args.end_idx]
    print(
        f"Split '{args.split_name}' from {split_path.name}: {len(idx)} galaxies, "
        f"running [{args.start_idx}:{args.end_idx}] -> {len(sliced)} galaxies"
    )
    print(f"  global row indices: {sliced}")
    return sliced


def get_dataset_and_norm(args):
    """Build dataset and norm dict from args.data_source. Returns (dataset, norm_dict, sim_label)."""
    import importlib

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
    # Default {} for suites that deliberately carry no NORM_DICT of their own (the dwarf
    # FIRE loader keeps its constants only in config.NORM_PROFILES, so there is no fourth
    # copy to drift). merged_norm_dict then returns the profile alone.
    norm_dict = getattr(mod, "NORM_DICT", {})

    root = getattr(args, "data_root", None) or default_root
    resolution = RESOLUTION[args.version]
    file_suffix = GRID_SUFFIX[args.version][1:]
    idx_list = _resolve_idx_list(args)

    # The version tag already selects the dataset variant via GRID_SUFFIX; it selects the
    # matching normalization constants the same way, so the maps are normalized here exactly
    # as they were during training. The merged dict is returned (not the module global) so
    # the denormalization feeding plot_radial_profile stays consistent with it.
    norm_profile_name = NORM_PROFILE.get(args.version, "fixed32")
    norm_dict = merged_norm_dict(norm_dict, get_norm_profile(args.version))
    # The clip is a per-model recipe choice, so unlike the normalization it cannot be
    # derived from the dataset variant -- a tag missing from CLIP_PROFILE gets no clipping,
    # which for a model trained with one would be a silent train/inference mismatch. Printed
    # loudly for that reason.
    clip_profile_name = CLIP_PROFILE.get(args.version, "none")
    print(f"Normalization profile for {args.version}: {norm_profile_name}")
    print(f"Conditioning clip profile for {args.version}: {clip_profile_name}")

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
        norm_profile=norm_profile_name,
        clip_profile=clip_profile_name,
    )
    extra = kwargs_fn(args) if kwargs_fn else {}
    dataset = dataset_class(**common, **extra)
    return dataset, norm_dict, data_source


def custom_plot_evals(
    x0,
    x1,
    x_pred,
    save_dir,
    galaxy_id=None,
    x_conditioning=None,
    fig_name="deltas",
    x0mean=None,
    x0std=None,
    x1mean=None,
    x1std=None,
    x_conditioning_mean=None,
    x_conditioning_std=None,
):
    if galaxy_id is not None:
        fig_name = f"{fig_name}_galaxy_{galaxy_id}"

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    x0_np = to_numpy(x0)
    x1_np = to_numpy(x1)
    x_pred_np = to_numpy(x_pred)

    x_conditioning_np = None
    if x_conditioning is not None:
        x_conditioning_np = to_numpy(x_conditioning)
        n_conditioning = (
            x_conditioning_np.shape[0] if len(x_conditioning_np.shape) > 2 else 1
        )
    else:
        n_conditioning = 0

    if x0mean is not None and x0std is not None:
        x0_np = (x0_np * x0std) + x0mean
    if x1mean is not None and x1std is not None:
        x1_np = (x1_np * x1std) + x1mean
        x_pred_np = (x_pred_np * x1std) + x1mean
    if (
        x_conditioning_mean is not None
        and x_conditioning_std is not None
        and x_conditioning is not None
    ):
        if len(x_conditioning_np.shape) > 2:
            assert (
                n_conditioning == len(x_conditioning_mean)
            ), "Conditioning mean/std size mismatch."
            for i in range(n_conditioning):
                x_conditioning_np[i] = (
                    x_conditioning_np[i] * x_conditioning_std[i]
                ) + x_conditioning_mean[i]
        else:
            assert len(x_conditioning_std) == len(x_conditioning_mean) == 1
            x_conditioning_np = (
                x_conditioning_np * x_conditioning_std[0]
            ) + x_conditioning_mean[0]

    plt.figure(figsize=(10, 6))
    plt.grid(alpha=0.5)
    _, edges, _ = plt.hist(
        x0_np.reshape(-1),
        label="x0",
        density=True,
        alpha=0.5,
        bins=300,
    )
    _, edges, _ = plt.hist(
        x1_np.reshape(-1),
        label="x1",
        density=True,
        alpha=0.5,
        bins=100,
    )
    plt.hist(
        x_pred_np.reshape(-1),
        label="Median pred x1",
        bins=edges,
        alpha=0.5,
        density=True,
    )
    plt.legend()
    if galaxy_id is not None:
        plt.title(f"Galaxy {galaxy_id}")
    plt.savefig(
        os.path.join(save_dir, f"{fig_name}_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    if len(x1_np.shape) == 3:
        x1_plot = x1_np[:10].sum(axis=0)
        x0_plot = x0_np[:10].sum(axis=0)
        x_pred_plot = x_pred_np[:10].sum(axis=0)
    else:
        x1_plot = x1_np
        x0_plot = x0_np
        x_pred_plot = x_pred_np

    diff_plot = x_pred_plot - x1_plot
    ref_diff_plot = (x_pred_plot - x1_plot) / np.clip(np.abs(x1_plot), 1e-6, None)
    diff_cmap = plt.cm.RdBu_r
    diff_vmin, diff_vmax = -0.5, 0.5
    ref_diff_vmin, ref_diff_vmax = -0.2, 0.2
    x1_vmin, x1_vmax = x1_plot.min(), x1_plot.max()

    if x_conditioning is not None:
        n_panels = 5 + n_conditioning
        fig, ax = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
        im = ax[0].imshow(x0_plot)
        ax[0].set_title("x0")
        plt.colorbar(im, ax=ax[0])

        if len(x_conditioning_np.shape) > 2:
            for i in range(n_conditioning):
                cond_plot = x_conditioning_np[i]
                if len(cond_plot.shape) == 3:
                    cond_plot = cond_plot[:10].sum(axis=0)
                im = ax[i + 1].imshow(cond_plot)
                plt.colorbar(im, ax=ax[i + 1])
                ax[i + 1].set_title(f"Conditioning {i+1}")
        else:
            im = ax[1].imshow(x_conditioning_np)
            plt.colorbar(im, ax=ax[1])
            ax[1].set_title("Conditioning")

        im = ax[-4].imshow(x1_plot, vmin=x1_vmin, vmax=x1_vmax)
        plt.colorbar(im, ax=ax[-4])
        ax[-4].set_title("x1")
        im = ax[-3].imshow(x_pred_plot, vmin=x1_vmin, vmax=x1_vmax)
        plt.colorbar(im, ax=ax[-3])
        ax[-3].set_title("Median pred x1")
        im = ax[-2].imshow(diff_plot, vmin=diff_vmin, vmax=diff_vmax, cmap=diff_cmap)
        plt.colorbar(im, ax=ax[-2])
        ax[-2].set_title("pred - x1")
        im = ax[-1].imshow(
            ref_diff_plot, vmin=ref_diff_vmin, vmax=ref_diff_vmax, cmap=diff_cmap
        )
        plt.colorbar(im, ax=ax[-1])
        ax[-1].set_title("(pred - x1)/|x1|")
    else:
        fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        im = ax[0].imshow(x0_plot)
        plt.colorbar(im, ax=ax[0])
        ax[0].set_title("x0")
        im = ax[1].imshow(x1_plot, vmin=x1_vmin, vmax=x1_vmax)
        plt.colorbar(im, ax=ax[1])
        ax[1].set_title("x1")
        im = ax[2].imshow(x_pred_plot, vmin=x1_vmin, vmax=x1_vmax)
        plt.colorbar(im, ax=ax[2])
        ax[2].set_title("Median pred x1")
        im = ax[3].imshow(diff_plot, vmin=diff_vmin, vmax=diff_vmax, cmap=diff_cmap)
        plt.colorbar(im, ax=ax[3])
        ax[3].set_title("pred - x1")
        im = ax[4].imshow(
            ref_diff_plot, vmin=ref_diff_vmin, vmax=ref_diff_vmax, cmap=diff_cmap
        )
        plt.colorbar(im, ax=ax[4])
        ax[4].set_title("(pred - x1)/|x1|")

    if galaxy_id is not None:
        fig.suptitle(f"Galaxy {galaxy_id}")
    plt.savefig(
        os.path.join(save_dir, f"{fig_name}_deltas.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    bins = np.arange(-0.5, 0.5, 0.01)
    plt.grid(alpha=0.5)
    plt.hist(
        diff_plot.reshape(-1),
        label="pred - x1",
        density=True,
        alpha=0.5,
        bins=bins,
    )
    plt.hist(
        ref_diff_plot.reshape(-1),
        label="(pred - x1)/|x1|",
        density=True,
        alpha=0.5,
        bins=bins,
    )
    plt.xlim(-1.0, 1.0)
    plt.legend()
    if galaxy_id is not None:
        plt.title(f"Galaxy {galaxy_id}")
    plt.savefig(
        os.path.join(save_dir, f"{fig_name}_deltas_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_uncertainty_maps(
    draws,
    save_dir,
    galaxy_id=None,
    fig_name="uncertainty",
    x1mean=None,
    x1std=None,
):
    """Show median and ±34 percentile range derived from stochastic draws."""
    if galaxy_id is not None:
        fig_name = f"{fig_name}_galaxy_{galaxy_id}"

    if isinstance(draws, torch.Tensor):
        draws_np = draws.cpu().numpy()
    else:
        draws_np = np.asarray(draws)

    median_map = np.median(draws_np, axis=0)
    lower_map = np.percentile(draws_np, 16, axis=0)
    upper_map = np.percentile(draws_np, 84, axis=0)
    minus_sigma = median_map - lower_map
    plus_sigma = upper_map - median_map
    half_width = 0.5 * (upper_map - lower_map)

    if x1mean is not None and x1std is not None:
        median_map = (median_map * x1std) + x1mean
        minus_sigma = minus_sigma * x1std
        plus_sigma = plus_sigma * x1std
        half_width = half_width * x1std

    _, axs = plt.subplots(1, 4, figsize=(20, 5))

    im = axs[0].imshow(median_map.squeeze())
    axs[0].set_title("Median pred")
    plt.colorbar(im, ax=axs[0])

    im = axs[1].imshow(minus_sigma.squeeze())
    axs[1].set_title("Median - 16th")
    plt.colorbar(im, ax=axs[1])

    im = axs[2].imshow(plus_sigma.squeeze())
    axs[2].set_title("84th - Median")
    plt.colorbar(im, ax=axs[2])

    im = axs[3].imshow(half_width.squeeze())
    axs[3].set_title("Half width (±34%)")
    plt.colorbar(im, ax=axs[3])

    if galaxy_id is not None:
        plt.suptitle(f"Galaxy {galaxy_id}")

    plt.savefig(
        os.path.join(save_dir, f"{fig_name}_percentile_band.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_radial_profile(
    x0,
    x1,
    x_pred,
    save_dir,
    galaxy_id=None,
    x_conditioning=None,
    fig_name="radial_profile",
    x0mean=None,
    x0std=None,
    x1mean=None,
    x1std=None,
    x_conditioning_mean=None,
    x_conditioning_std=None,
    xmin=-8,
    xmax=8,
    ymin=-8,
    ymax=8,
    save_profile=False,
    render_plot=True,
):
    if galaxy_id is not None:
        fig_name = f"{fig_name}_galaxy_{galaxy_id}"

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    x0_np = to_numpy(x0)
    x1_np = to_numpy(x1)
    x_pred_np = to_numpy(x_pred)

    if x_conditioning is not None:
        x_conditioning_np = to_numpy(x_conditioning)
        n_conditioning = (
            x_conditioning_np.shape[0] if len(x_conditioning_np.shape) > 2 else 1
        )

    if x0mean is not None and x0std is not None:
        x0_np = (x0_np * x0std) + x0mean
    if x1mean is not None and x1std is not None:
        x1_np = (x1_np * x1std) + x1mean
        x_pred_np = (x_pred_np * x1std) + x1mean
    if (
        x_conditioning_mean is not None
        and x_conditioning_std is not None
        and x_conditioning is not None
    ):
        if len(x_conditioning_np.shape) > 2:
            assert (
                n_conditioning == len(x_conditioning_mean)
            ), "Conditioning mean/std size mismatch."
            for i in range(n_conditioning):
                x_conditioning_np[i] = (
                    x_conditioning_np[i] * x_conditioning_std[i]
                ) + x_conditioning_mean[i]
        else:
            assert len(x_conditioning_std) == len(x_conditioning_mean) == 1
            x_conditioning_np = (
                x_conditioning_np * x_conditioning_std[0]
            ) + x_conditioning_mean[0]

    if x0_np.ndim != 2 or x1_np.ndim != 2 or x_pred_np.ndim != 2:
        raise ValueError("plot_radial_profile expects 2D arrays.")
    if x0_np.shape != x1_np.shape or x0_np.shape != x_pred_np.shape:
        raise ValueError("x0, x1, and x_pred must share shape.")
    if x0_np.shape[0] != x0_np.shape[1]:
        raise ValueError("Input arrays must be square.")
    resolution = x0_np.shape[0]

    r_min = 0.0
    r_max = float(np.sqrt(xmax**2 + ymax**2))
    num_bins = max(1, int(resolution // 4))
    radial_edges = np.linspace(r_min, r_max, num_bins + 1)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])

    x0_radial_profile = np.full(num_bins, np.nan)
    x1_radial_profile = np.full(num_bins, np.nan)
    x_pred_radial_profile = np.full(num_bins, np.nan)
    x0_radial_profile_std = np.full(num_bins, np.nan)
    x1_radial_profile_std = np.full(num_bins, np.nan)
    x_pred_radial_profile_std = np.full(num_bins, np.nan)

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")

    for i in range(num_bins):
        r1 = radial_edges[i]
        r2 = radial_edges[i + 1]
        if i < num_bins - 1:
            in_bin = ((x_coords**2 + y_coords**2) >= r1**2) & (
                (x_coords**2 + y_coords**2) < r2**2
            )
        else:
            in_bin = ((x_coords**2 + y_coords**2) >= r1**2) & (
                (x_coords**2 + y_coords**2) <= r2**2
            )
        if in_bin.any():
            x0_vals = x0_np[in_bin]
            x1_vals = x1_np[in_bin]
            x_pred_vals = x_pred_np[in_bin]
            x0_radial_profile[i] = x0_vals.mean()
            x1_radial_profile[i] = x1_vals.mean()
            x_pred_radial_profile[i] = x_pred_vals.mean()
            x0_radial_profile_std[i] = x0_vals.std()
            x1_radial_profile_std[i] = x1_vals.std()
            x_pred_radial_profile_std[i] = x_pred_vals.std()

    if save_profile:
        np.savez(
            os.path.join(save_dir, f"{fig_name}_radial_profile.npz"),
            radial_centers=radial_centers,
            x0_radial_profile=x0_radial_profile,
            x1_radial_profile=x1_radial_profile,
            x_pred_radial_profile=x_pred_radial_profile,
            x0_radial_profile_std=x0_radial_profile_std,
            x1_radial_profile_std=x1_radial_profile_std,
            x_pred_radial_profile_std=x_pred_radial_profile_std,
        )
        print(
            f"Saved radial profile data for galaxy {galaxy_id} "
            f"to {os.path.join(save_dir, f'{fig_name}_radial_profile.npz')}"
        )

    if render_plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex=True)
        axs[0].plot(radial_centers, x0_radial_profile, label="x0", color="black")
        axs[0].fill_between(
            radial_centers,
            x0_radial_profile - x0_radial_profile_std,
            x0_radial_profile + x0_radial_profile_std,
            color="black",
            alpha=0.5,
        )
        axs[0].legend()
        axs[1].plot(radial_centers, x1_radial_profile, label="x1", color="black")
        axs[1].fill_between(
            radial_centers,
            x1_radial_profile - x1_radial_profile_std,
            x1_radial_profile + x1_radial_profile_std,
            color="black",
            alpha=0.5,
        )
        axs[1].plot(
            radial_centers, x_pred_radial_profile, label="Median pred", color="red"
        )
        axs[1].fill_between(
            radial_centers,
            x_pred_radial_profile - x_pred_radial_profile_std,
            x_pred_radial_profile + x_pred_radial_profile_std,
            color="red",
            alpha=0.5,
        )
        axs[1].legend()
        axs[0].set_xlabel("Radius [kpc]")
        axs[0].set_ylabel("log10(Density)")
        axs[1].set_xlabel("Radius [kpc]")
        axs[1].set_ylabel("log10(Density)")
        plt.savefig(
            os.path.join(save_dir, f"{fig_name}_radial_profile.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    ## Example for TNG:
    # python batch_plot_pred_uncertainty_da.py --checkpoint-path $STARS2DM_CKPT_ROOT/last-v95.ckpt --start-idx 0 --end-idx 70 --version v44 --x-conditioning-fields EuVISfluxnoiselog10 --x1-fields rhoDM --xmin -32 --xmax 32 --ymin -32 --ymax 32 --save-full-trajectory --data-source TNG50 --data-root $STARS2DM_TNG_ROOT
    parser = argparse.ArgumentParser(
        description="Batch plot galaxy predictions with MC uncertainty."
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
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
    parser.add_argument("--xmin", type=float, default=-32)
    parser.add_argument("--xmax", type=float, default=32)
    parser.add_argument("--ymin", type=float, default=-32)
    parser.add_argument("--ymax", type=float, default=32)
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
        help="Which simulation data to load for testing (DREAMS, TNG50, TNG100, ...). "
        "Recorded in all output paths and filenames.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory for the chosen data source.",
    )
    parser.add_argument("--suite", type=str, default="TNG")
    parser.add_argument("--dataset", type=str, default="SB5")
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="Permutation .npz holding train_idx/val_idx/test_idx (path, or relative to "
             "--data-root). Required for the dwarf FIRE suite, whose split is permuted: "
             "a contiguous --start-idx/--end-idx range there is a different galaxy set.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Which split from --split-file to run on. --start-idx/--end-idx then shard it.",
    )
    parser.add_argument("--redshift", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of stochastic draws per galaxy.",
    )
    parser.add_argument(
        "--save-full-trajectory",
        action="store_true",
        help="Save the stacked MC draws for each galaxy.",
    )
    parser.add_argument(
        "--plot-one-per-batch",
        action="store_true",
        help="Only render plots for the first galaxy in each batch.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix appended to version in output filenames (e.g. heun100).",
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
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip histogram/map/radial-profile PNG generation; save npz only.",
    )
    args = parser.parse_args()

    dataset, norm_dict, sim_label = get_dataset_and_norm(args)
    norm_dict = norm_dict[f"{args.dims}D"]

    # OOD outputs go under output_dir / sim_label so the sim is recorded in paths and
    # filenames. In-domain suites (DREAMS, DWARFFIRE) write directly to output_dir on the
    # plain galaxy_N_<version>_... convention that pred_data_loaders resolves by default.
    path_label = "" if sim_label in IN_DOMAIN_SOURCES else sim_label
    base_output_dir = (
        os.path.join(args.output_dir, path_label) if path_label else args.output_dir
    )
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Data source: {sim_label} (outputs under {base_output_dir})")

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = LightningStochasticInterpolant.load_from_checkpoint(args.checkpoint_path)
    model = model.to(args.device)
    model.eval()
    if not getattr(model.hparams, "condition_on_parameters", True):
        print(
            "Note: model trained without parameter conditioning; "
            "using h=None at inference (theta from the loader is unused)."
        )

    n_sampling_steps = (
        args.n_sampling_steps
        if args.n_sampling_steps is not None
        else model.hparams.n_sampling_steps
    )

    print(f"Loading data for indices: {args.start_idx}..{args.end_idx}")
    print(f"Dataset loaded with {len(dataset)} samples")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=9
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            torch.cuda.reset_peak_memory_stats(args.device)

            batch = {
                k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            theta = batch["theta"]
            h_cond = (
                theta
                if getattr(model.hparams, "condition_on_parameters", True)
                else None
            )
            x_conditioning = batch["x_conditioning"]
            x1 = batch["x1"]
            dataset_indices = batch["dataset_index"].cpu().numpy()

            x0 = x_conditioning[:, 0:1, ...]

            draws = []
            for sample_idx in range(args.num_samples):
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
            draws = torch.stack(draws)  # [num_samples, batch, channels, ...]
            median_pred = torch.median(draws, dim=0).values

            plots_rendered = 0
            for i in range(theta.shape[0]):
                should_plot = (
                    (not args.no_plots)
                    and ((not args.plot_one_per_batch) or (plots_rendered == 0))
                )
                dataset_index = int(dataset_indices[i])
                galaxy_dir = os.path.join(
                    base_output_dir, f"galaxy_{dataset_index:04d}"
                )
                os.makedirs(galaxy_dir, exist_ok=True)

                # All filenames include sim_label and version for traceability.
                file_tag = (
                    f"galaxy_{dataset_index}_{path_label}_{args.version}"
                    if path_label
                    else f"galaxy_{dataset_index}_{args.version}"
                )
                if args.output_suffix:
                    file_tag = f"{file_tag}_{args.output_suffix}"

                single_draws = draws[:, i]
                if args.save_full_trajectory:
                    np.savez_compressed(
                        os.path.join(galaxy_dir, f"{file_tag}_pred_draws_{args.num_samples}.npz"),
                        draws=single_draws.numpy(),
                    )

                x_conditioning_i = x_conditioning[i : i + 1]
                x1_i = x1[i : i + 1]
                median_i = median_pred[i : i + 1]

                x0_i = x_conditioning_i[:, 0:1, ...]

                additional_conditioning = None
                if x_conditioning_i.shape[1] > 1:
                    additional_conditioning = x_conditioning_i[0, 1:, ...].cpu()

                x0mean = norm_dict[args.x_conditioning_fields[0]][RESOLUTION[args.version]][
                    "mean"
                ]
                x0std = norm_dict[args.x_conditioning_fields[0]][RESOLUTION[args.version]][
                    "std"
                ]
                x1mean = norm_dict[args.x1_fields[0]][RESOLUTION[args.version]]["mean"]
                x1std = norm_dict[args.x1_fields[0]][RESOLUTION[args.version]]["std"]

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

                if should_plot:
                    custom_plot_evals(
                        x0=x0_i[0].cpu().squeeze(),
                        x1=x1_i[0].cpu().squeeze(),
                        x_pred=median_i[0].cpu().squeeze(),
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

                plot_radial_profile(
                    x0=x0_i[0].cpu().squeeze(),
                    x1=x1_i[0].cpu().squeeze(),
                    x_pred=median_i[0].cpu().squeeze(),
                    save_dir=galaxy_dir,
                    galaxy_id=dataset_index,
                    fig_name=file_tag,
                    x0mean=x0mean,
                    x0std=x0std,
                    x1mean=x1mean,
                    x1std=x1std,
                    x_conditioning_mean=x_conditioning_mean,
                    x_conditioning_std=x_conditioning_std,
                    xmin=args.xmin,
                    xmax=args.xmax,
                    ymin=args.ymin,
                    ymax=args.ymax,
                    save_profile=True,
                    render_plot=should_plot,
                )

                if should_plot:
                    plot_uncertainty_maps(
                        draws=single_draws.squeeze(1),
                        save_dir=galaxy_dir,
                        galaxy_id=dataset_index,
                        fig_name=file_tag,
                        x1mean=x1mean,
                        x1std=x1std,
                    )
                    plots_rendered += 1
                status = "plots rendered" if should_plot else "plots skipped"
                print(
                    f"Saved prediction data for galaxy {dataset_index} "
                    f"to {galaxy_dir} ({status})"
                )

            current_alloc = torch.cuda.memory_allocated(args.device) / 1e9
            peak_alloc = torch.cuda.max_memory_allocated(args.device) / 1e9
            print(
                f"[GPU] batch {batch_idx}: alloc={current_alloc:.2f}GB "
                f"peak={peak_alloc:.2f}GB"
            )


if __name__ == "__main__":
    main()

