import argparse
import os

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
from baryonbridge.data.dreams import DREAMS, NORM_DICT
from baryonbridge.interpolants.stochastic import LightningStochasticInterpolant


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
    parser.add_argument("--suite", type=str, default="TNG")
    parser.add_argument("--dataset", type=str, default="SB5")
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
        "--force-h-none",
        action="store_true",
        help="Pass h=None at inference even for theta-conditioned checkpoints.",
    )
    parser.add_argument(
        "--force-mean-theta",
        action="store_true",
        help="Replace per-galaxy theta with dataset mean vector at inference.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix appended to version in output filenames (e.g. hnone, hmean).",
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

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = LightningStochasticInterpolant.load_from_checkpoint(args.checkpoint_path)
    model = model.to(args.device)
    model.eval()
    if not getattr(model.hparams, "condition_on_parameters", True):
        print(
            "Note: model trained without parameter conditioning; "
            "using h=None at inference (theta from the loader is unused)."
        )

    idx_list = list(range(args.start_idx, args.end_idx))
    print(f"Loading data for indices: {idx_list}")

    dataset = DREAMS(
        root="./data/DREAMS_data",
        dims=args.dims,
        resolution=RESOLUTION[args.version],
        redshift=args.redshift,
        suite=args.suite,
        dataset=args.dataset,
        x1_fields=args.x1_fields,
        x_conditioning_fields=args.x_conditioning_fields,
        file_suffix=GRID_SUFFIX[args.version][1:],
        idx_list=idx_list,
        transform=None,
        randomize_theta=False,  # Explicitly disable randomization for evaluation
        randomize_x_conditioning=False,  # Explicitly disable randomization for evaluation
        randomize_x1=False,  # Explicitly disable randomization for evaluation
        norm_profile=NORM_PROFILE.get(args.version, "fixed32"),
        clip_profile=CLIP_PROFILE.get(args.version, "none"),
    )
    # The version tag already selects the dataset variant via GRID_SUFFIX; it selects the
    # matching normalization constants the same way, so the maps are normalized here exactly
    # as they were during training, and the denormalization below (which feeds
    # plot_radial_profile) stays consistent with them.
    print(
        f"Normalization profile for {args.version}: "
        f"{NORM_PROFILE.get(args.version, 'fixed32')}"
    )
    # The clip is a per-model recipe choice, so unlike the normalization it cannot be
    # derived from the dataset variant -- a tag missing from CLIP_PROFILE gets no clipping,
    # which for a model trained with one would be a silent train/inference mismatch.
    print(
        f"Conditioning clip profile for {args.version}: "
        f"{CLIP_PROFILE.get(args.version, 'none')}"
    )
    norm_dict = merged_norm_dict(NORM_DICT, get_norm_profile(args.version))[
        f"{args.dims}D"
    ]
    print(f"Dataset loaded with {len(dataset)} samples")

    if args.force_mean_theta:
        full_dataset = DREAMS(
            root="./data/DREAMS_data",
            dims=args.dims,
            resolution=RESOLUTION[args.version],
            redshift=args.redshift,
            suite=args.suite,
            dataset=args.dataset,
            x1_fields=args.x1_fields,
            x_conditioning_fields=args.x_conditioning_fields,
            file_suffix=GRID_SUFFIX[args.version][1:],
            idx_list=None,
            transform=None,
            randomize_theta=False,
            randomize_x_conditioning=False,
            randomize_x1=False,
            norm_profile=NORM_PROFILE.get(args.version, "fixed32"),
        )
        mean_theta = full_dataset.theta.mean(dim=0, keepdim=True)
        print(f"Using dataset-mean theta: {mean_theta.squeeze().tolist()}")

    n_sampling_steps = (
        args.n_sampling_steps
        if args.n_sampling_steps is not None
        else model.hparams.n_sampling_steps
    )
    version_label = args.version
    if args.output_suffix:
        version_label = f"{args.version}_{args.output_suffix}"

    def pred_draws_filename(dataset_index):
        stem = f"galaxy_{dataset_index}_{args.version}_pred_draws_{args.num_samples}"
        if args.output_suffix:
            stem = f"{stem}_{args.output_suffix}"
        return f"{stem}.npz"

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
            if args.force_h_none:
                h_cond = None
            elif args.force_mean_theta:
                h_cond = mean_theta.expand(theta.shape[0], -1).to(args.device)
            elif getattr(model.hparams, "condition_on_parameters", True):
                h_cond = theta
            else:
                h_cond = None
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
                    args.output_dir, f"galaxy_{dataset_index:04d}"
                )
                os.makedirs(galaxy_dir, exist_ok=True)

                single_draws = draws[:, i]
                if args.save_full_trajectory:
                    np.savez_compressed(
                        os.path.join(galaxy_dir, pred_draws_filename(dataset_index)),
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
                        fig_name=f"galaxy_{dataset_index}_{version_label}",
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
                    fig_name=f"galaxy_{dataset_index}_{version_label}",
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
                        fig_name=f"galaxy_{dataset_index}_{version_label}",
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

