import torch
import numpy as np
from typing import Tuple, Optional, List
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from scipy import ndimage


from baryonbridge.models.pixels.cunet import CUNet
from baryonbridge.eval import plot_evals, plot_loss_correlation


def build_radial_annulus_mask(
    resolution,
    xmin=-32.0,
    xmax=32.0,
    ymin=-32.0,
    ymax=32.0,
):
    """Same 32-bin annulus layout as analysis (resolution // 4 bins at 128²)."""
    r_min = 0.0
    r_max = float(np.sqrt(xmax ** 2 + ymax ** 2))
    num_bins = max(1, int(resolution // 4))
    radial_edges = np.linspace(r_min, r_max, num_bins + 1)

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")

    mask = np.full((num_bins, resolution, resolution), False)
    for i in range(num_bins):
        r1 = radial_edges[i]
        r2 = radial_edges[i + 1]
        if i < num_bins - 1:
            in_bin = ((x_coords ** 2 + y_coords ** 2) >= r1 ** 2) & (
                (x_coords ** 2 + y_coords ** 2) < r2 ** 2
            )
        else:
            in_bin = ((x_coords ** 2 + y_coords ** 2) >= r1 ** 2) & (
                (x_coords ** 2 + y_coords ** 2) <= r2 ** 2
            )
        if in_bin.any():
            mask[i] = in_bin
    return mask


def load_annulus_abs_bias_from_csv(
    csv_path,
    config_label="v47",
    aggregation="median_median",
):
    """Load |median_bias| per annulus bin from population_bias_summary-style CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "abs_bias" in df.columns and "config_label" not in df.columns:
        sub = df.sort_values("radial_bin")
        return sub["radial_bin"].values, sub["abs_bias"].values.astype(float)
    sub = df[
        (df["config_label"] == config_label)
        & (df["aggregation"] == aggregation)
    ].sort_values("radial_bin")
    if sub.empty:
        raise ValueError(
            f"No rows for config_label={config_label!r} aggregation={aggregation!r} in {csv_path}"
        )
    return sub["radial_bin"].values, np.abs(sub["median_bias"].values.astype(float))


def build_radial_pixel_weight_map(
    resolution,
    inner_weight=4.0,
    r_threshold_kpc=10.0,
    xmin=-32.0,
    xmax=32.0,
    ymin=-32.0,
    ymax=32.0,
    mode="step",
    normalize=True,
    bias_csv_path=None,
    bias_config_label="v47",
    bias_aggregation="median_median",
    bias_power=1.0,
    power=1.0,
    r_min_kpc=5.0,
):
    """
    Pixel loss weights for radially weighted MSE.

    mode:
      step — inner_weight inside r_threshold_kpc (v55 legacy).
      annulus_equal — w_k = 1/N_k per annulus, mean-normalized over the grid.
      bias_calibrated — w_k = |b_k|^p from v47 (or other) population bias CSV (Option C).
      radial_power — w(r) = 1/max(r, r_min_kpc)^power from each pixel's true radius,
                     mean-normalized. power=1 approximates annulus_equal (1/r) but is
                     monotonic in r (no square-FOV corner blow-up); r_min_kpc floors the
                     central divergence for high power.
    """
    if mode == "radial_power":
        x_edges = np.linspace(xmin, xmax, resolution + 1)
        y_edges = np.linspace(ymin, ymax, resolution + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")
        r = np.sqrt(x_coords ** 2 + y_coords ** 2)
        r_eff = np.maximum(r, float(r_min_kpc))
        weights = (1.0 / r_eff ** float(power)).astype(np.float32)
        if normalize:
            weights = weights / np.mean(weights)
        return torch.tensor(weights)
    if mode == "annulus_equal":
        mask = build_radial_annulus_mask(
            resolution, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )
        weights = np.zeros((resolution, resolution), dtype=np.float32)
        for k in range(mask.shape[0]):
            n_k = int(mask[k].sum())
            if n_k > 0:
                weights[mask[k]] = 1.0 / n_k
        if normalize:
            weights = weights / np.mean(weights)
        return torch.tensor(weights)

    if mode == "bias_calibrated":
        if bias_csv_path is None:
            raise ValueError("bias_calibrated mode requires bias_csv_path")
        _, bias_abs = load_annulus_abs_bias_from_csv(
            bias_csv_path, config_label=bias_config_label, aggregation=bias_aggregation
        )
        mask = build_radial_annulus_mask(
            resolution, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )
        if len(bias_abs) != mask.shape[0]:
            raise ValueError(
                f"Bias profile has {len(bias_abs)} bins but annulus mask has {mask.shape[0]}"
            )
        w_k = np.maximum(bias_abs, 1e-6) ** float(bias_power)
        weights = np.zeros((resolution, resolution), dtype=np.float32)
        for k in range(mask.shape[0]):
            if mask[k].any():
                weights[mask[k]] = w_k[k]
        if normalize:
            weights = weights / np.mean(weights)
        return torch.tensor(weights)

    if mode != "step":
        raise ValueError(f"Unknown radial loss weight mode: {mode!r}")

    x_edges = np.linspace(xmin, xmax, resolution + 1)
    y_edges = np.linspace(ymin, ymax, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing="ij")
    r = np.sqrt(x_coords ** 2 + y_coords ** 2)
    weights = np.ones((resolution, resolution), dtype=np.float32)
    weights[r < r_threshold_kpc] = inner_weight
    if normalize:
        weights = weights / np.mean(weights)
    return torch.tensor(weights)

# Add import for galaxy info utility
try:
    from baryonbridge.utils.galaxy_info import print_galaxy_summary
except ImportError:
    # Fallback if the utility is not available
    def print_galaxy_summary(galaxy_id, **kwargs):
        print(f"Galaxy {galaxy_id} (detailed info not available)")


class EulerSDE(torch.nn.Module):
    def __init__(self, drift_func, sigma, delta_t, method="euler"):
        super().__init__()
        if method not in ("euler", "heun"):
            raise ValueError(f"Unknown SDE integration method: {method!r}")
        self.drift_func = drift_func
        self.sigma = sigma
        self.delta_t = delta_t
        self.method = method

    def step_forward(self, x, x0, t: torch.tensor, is_last=False):
        """Euler-Maruyama, or stochastic Heun (predictor-corrector)."""
        if is_last:
            dW = 0.0
        else:
            dW = torch.sqrt(self.delta_t) * torch.randn(size=x.shape).to(x.device)
        drift_start = self.drift_func(t, x, x0)
        if self.method == "euler":
            x += drift_start * self.delta_t + self.sigma(t) * dW
            return x
        # Heun: Euler predictor, then trapezoidal corrector reusing the same dW.
        t_next = t + self.delta_t
        x_pred = x + drift_start * self.delta_t + self.sigma(t) * dW
        drift_end = self.drift_func(t_next, x_pred, x0)
        x += (
            0.5 * (drift_start + drift_end) * self.delta_t
            + 0.5 * (self.sigma(t) + self.sigma(t_next)) * dW
        )
        return x

    def integrate(
        self,
        x0,
        return_trajectory: bool = False,
    ):
        n_steps = int(1 / self.delta_t)
        x = x0.clone()
        if return_trajectory:
            trajectory = [x0.clone().detach().cpu()]
        for i in range(n_steps):
            x = self.step_forward(
                x=x, x0=x0, t=i * self.delta_t, is_last=(i == n_steps - 1)
            )
            if return_trajectory:
                trajectory.append(x.clone().detach().cpu())
        if return_trajectory:
            return trajectory
        return x


class StochasticInterpolant(torch.nn.Module):
    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model

    def get_alpha_t(self, t):
        return 1 - t

    def get_beta_t(self, t):
        return t**2

    def get_sigma_t(self, t):
        return 1 - t

    def get_alpha_t_dot(self, t):
        return -torch.ones_like(t)

    def get_beta_t_dot(self, t):
        return 2.0 * t

    def get_sigma_t_dot(self, t):
        return -torch.ones_like(t)

    def get_xt(self, x0, x1, t, epsilon):
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        
        # Check inputs
        if torch.isnan(x0).any() or torch.isnan(x1).any():
            print("NaN in input data")
        
        alpha = self.get_alpha_t(t)
        beta = self.get_beta_t(t)
        sigma = self.get_sigma_t(t)
        
        xt = (alpha * x0 + beta * x1 + torch.sqrt(t) * sigma * epsilon)
        
        if torch.isnan(xt).any():
            print(f"NaN in xt: alpha={alpha.min():.3f}-{alpha.max():.3f}, "
                  f"beta={beta.min():.3f}-{beta.max():.3f}")
        
        return xt

    def get_rt(self, x0, x1, t, epsilon):
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        return (
            self.get_alpha_t_dot(t) * x0
            + self.get_beta_t_dot(t) * x1
            + self.get_sigma_t_dot(t) * torch.sqrt(t) * epsilon
        )

    def compute_loss(self, x0, x1=None, x_conditioning=None, h=None, t=None, weight_map=None):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        eps = torch.randn_like(x0)
        xt = self.get_xt(x0=x0, x1=x1, t=t, epsilon=eps)
        rt = self.get_rt(x0=x0, x1=x1, t=t, epsilon=eps)
        
        if x_conditioning is None:
            x_conditioning = x0
        
        b_pred = self.velocity_model(t, xt, conditioning=x_conditioning, h=h)
        
        if torch.isnan(b_pred).any():
            print(f"NaN in b_pred: t={t.min():.3f}-{t.max():.3f}")
            print(f"xt range: {xt.min():.3f}-{xt.max():.3f}")
            return torch.tensor(float('inf')).type_as(x0)

        squared_error = (b_pred - rt) ** 2
        if weight_map is not None:
            w = weight_map.to(squared_error.device)
            while w.dim() < squared_error.dim():
                w = w.unsqueeze(0)
            denom = w.sum() * squared_error.shape[0] * squared_error.shape[1]
            return (squared_error * w).sum() / denom
        loss = torch.mean(squared_error)
        return loss

    def predict(
        self,
        x0,
        x_conditioning=None,
        h=None,
        n_sampling_steps=10,
        return_trajectory=False,
        sampler="euler",
    ):
        if x_conditioning is None:
            x_conditioning = x0

        delta_t = 1.0 / n_sampling_steps
        x0_size = x0.size()

        def drift_func(t, xt, x0):
            return self.velocity_model(
                t, xt.view(x0_size), conditioning=x_conditioning, h=h
            ).flatten(start_dim=1)

        def diffusion_func(t):
            return self.get_sigma_t(t)

        sde = EulerSDE(
            drift_func=drift_func,
            sigma=diffusion_func,
            delta_t=torch.tensor(delta_t).to(x0.device),
            method=sampler,
        )
        with torch.no_grad():
            ys = sde.integrate(
                x0.flatten(start_dim=1), return_trajectory=return_trajectory
            )
        if not return_trajectory:
            ys = ys.view(x0_size)
        else:
            ys = torch.stack(ys)
            ys = ys.view((n_sampling_steps + 1, *x0_size))
        return ys


class LightningStochasticInterpolant(LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 1,
        out_channels: int = 1,
        num_channels: Tuple = (32,64,64),
        layers_per_block: int = 2,
        dropout_prob: float = 0.1,
        num_groups_norm: int =32,
        num_h_embeds: Optional[int] = None,
        act_fn: str = 'SiLU',
        spatial_theta_conditioning: bool = False,
        spatial_theta_hidden: int = 64,
        condition_on_parameters=False,
        learning_rate: float =1.0e-4,
        n_sampling_steps: int =5,
        log_figures: bool =True,
        fit_residual: bool = False,
        dim: int = 3,
        x0_indices_in_conditioning: List[int] = [0],
        test_model: bool = False,
        validation_batch_index: int = 0,
        use_radial_loss_weight: bool = False,
        radial_loss_weight_mode: str = "step",
        radial_loss_weight_inner: float = 4.0,
        radial_loss_r_threshold_kpc: float = 10.0,
        radial_loss_resolution: int = 128,
        radial_loss_xmin: float = -32.0,
        radial_loss_xmax: float = 32.0,
        radial_loss_ymin: float = -32.0,
        radial_loss_ymax: float = 32.0,
        radial_loss_bias_csv_path: Optional[str] = None,
        radial_loss_bias_config_label: str = "v47",
        radial_loss_bias_aggregation: str = "median_median",
        radial_loss_bias_power: float = 1.0,
        radial_loss_step_normalize: bool = False,
        radial_loss_power: float = 1.0,
        radial_loss_r_min_kpc: float = 5.0,
        # Which parameter table print_galaxy_summary reads at validation time. Galaxy ids
        # overlap between suites (dwarf 0-395 sits inside DREAMS 0-1023), so the default
        # would report Milky Way TNG parameters for a FIRE dwarf without raising.
        galaxy_info_suite: str = "DREAMS",
    ):
        super().__init__()
        self.x0_indices_in_conditioning = x0_indices_in_conditioning
        self.test_model = test_model
        self.save_hyperparameters()
        velocity_model = CUNet(
            in_channels=in_channels,
            cond_channels=cond_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            layers_per_block=layers_per_block,
            dropout_prob=dropout_prob,
            num_groups_norm=num_groups_norm,
            num_h_embeds=num_h_embeds,
            act_fn=act_fn,
            spatial_theta_conditioning=spatial_theta_conditioning,
            spatial_theta_hidden=spatial_theta_hidden,
            dim=dim,
        )
        self.stochastic_interpolant = StochasticInterpolant(velocity_model)
        self.validation_step_outputs = []
        self.condition_on_parameters = condition_on_parameters
        self.learning_rate = learning_rate
        self.n_sampling_steps = n_sampling_steps
        self.log_figures = log_figures
        self.fit_residual = fit_residual
        self.epoch_losses = []
        if use_radial_loss_weight:
            step_normalize = (
                radial_loss_step_normalize if radial_loss_weight_mode == "step" else True
            )
            weight_map = build_radial_pixel_weight_map(
                resolution=radial_loss_resolution,
                inner_weight=radial_loss_weight_inner,
                r_threshold_kpc=radial_loss_r_threshold_kpc,
                xmin=radial_loss_xmin,
                xmax=radial_loss_xmax,
                ymin=radial_loss_ymin,
                ymax=radial_loss_ymax,
                mode=radial_loss_weight_mode,
                normalize=step_normalize,
                bias_csv_path=radial_loss_bias_csv_path,
                bias_config_label=radial_loss_bias_config_label,
                bias_aggregation=radial_loss_bias_aggregation,
                bias_power=radial_loss_bias_power,
                power=radial_loss_power,
                r_min_kpc=radial_loss_r_min_kpc,
            )
            self.register_buffer("radial_loss_weight_map", weight_map)
        else:
            self.radial_loss_weight_map = None

        annulus_mask = build_radial_annulus_mask(
            resolution=radial_loss_resolution,
            xmin=radial_loss_xmin,
            xmax=radial_loss_xmax,
            ymin=radial_loss_ymin,
            ymax=radial_loss_ymax,
        )  # (32, H, W) bool, same binning as population_bias_summary.csv's radial_bin
        inner_mask = annulus_mask[list(range(7))].any(axis=0)
        outer_mask = annulus_mask[list(range(14, 32))].any(axis=0)
        # persistent=False: purely derived from config, not learned -- must not be
        # required in state_dict, or loading any checkpoint saved before this code
        # existed (v76, v56, v55, v52, v47, ...) fails with "Missing key(s)".
        self.register_buffer("val_inner_mask", torch.from_numpy(inner_mask), persistent=False)
        self.register_buffer("val_outer_mask", torch.from_numpy(outer_mask), persistent=False)

    def get_batch(self, batch,):
        h, x_conditioning = batch['theta'], batch['x_conditioning']
        if self.fit_residual:
            x1 = batch['x_residual']
        else:
            x1 = batch['x1']
        x0 = torch.cat([x_conditioning[:, idx:idx+1, ...] for idx in self.x0_indices_in_conditioning], dim=1)
        if not self.condition_on_parameters:
            h = None
        if self.test_model:
            x0_np = x0.cpu().numpy()
            x1_np = np.zeros_like(x0_np)
            sigma_values = [1.0, 20.]  
            for i in range(x0_np.shape[1]):
                x1_np[:, i] = ndimage.gaussian_filter(x0_np[:, i], sigma=[0, sigma_values[i], sigma_values[i]])
            x1 = torch.tensor(x1_np, dtype=x_conditioning.dtype).to(x_conditioning.device)
        return h, x_conditioning, x0, x1

    def get_loss(self, batch, batch_idx):
        h, x_conditioning, x0, x1 = self.get_batch(
            batch
        )
        return self.stochastic_interpolant.compute_loss(
            x0=x0,
            x1=x1,
            h=h,
            x_conditioning=x_conditioning,
            weight_map=self.radial_loss_weight_map,
        )

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train_loss", loss, 
                 prog_bar=True, 
                 sync_dist=True, 
                 on_step=True, 
                 on_epoch=False, 
                 batch_size=batch['theta'].shape[0],
        ) 
        return loss

    def validation_step(self, batch, batch_idx,):
        loss = self.get_loss(batch, batch_idx)
        self.log("val_loss", loss,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch['theta'].shape[0])
        
        # Get predictions for MSE loss
        h, x_conditioning, x0, x1 = self.get_batch(batch)
        pred = self.stochastic_interpolant.predict(
            x0=x0,
            h=h,
            n_sampling_steps=self.hparams.n_sampling_steps,
            x_conditioning=x_conditioning,
        )
        if self.fit_residual:
            pred = pred + x_conditioning
            x1 = x1 + x_conditioning
        
        mse_loss = torch.mean((pred - x1) ** 2).item()

        diff = pred - x1
        inner_bias = diff[:, :, self.val_inner_mask].mean().item()
        outer_bias = diff[:, :, self.val_outer_mask].mean().item()

        self.validation_step_outputs.append({
            "loss": loss,
            "mse_loss": mse_loss,
            "inner_bias": inner_bias,
            "outer_bias": outer_bias,
            "batch": batch})
        return loss

    def on_validation_epoch_end(self):
        # Debug: Print galaxy IDs in each batch
        print("\n=== Validation Batch Galaxy IDs ===")
        for i, output in enumerate(self.validation_step_outputs):
            batch = output["batch"]
            galaxy_ids = batch.get('dataset_index', [])
            print(f"Batch {i}: Galaxy IDs {galaxy_ids.tolist()}")

        # Live proxy for the offline inner/outer radial bias metric (tilt_metrics.csv):
        # single draw over this epoch's validation batches rather than 10 draws over the
        # full held-out test set, so expect more noise epoch-to-epoch.
        mean_inner = sum(o["inner_bias"] for o in self.validation_step_outputs) / len(self.validation_step_outputs)
        mean_outer = sum(o["outer_bias"] for o in self.validation_step_outputs) / len(self.validation_step_outputs)
        self.log("val_inner_bias", mean_inner, sync_dist=True)
        self.log("val_outer_bias", mean_outer, sync_dist=True)
        self.log("val_tilt", mean_outer - mean_inner, sync_dist=True)

        # Use the configurable validation batch index
        batch_index = self.hparams.validation_batch_index
        print(f"\nSelected batch {batch_index} for plotting")
        sample_batch = self.validation_step_outputs[batch_index]["batch"]
        self._log_figures(sample_batch)
        self.validation_step_outputs.clear()

    @rank_zero_only
    def _log_figures(self, batch):
        h, x_conditioning, x0, x1 = self.get_batch(batch,)
        pred = self.stochastic_interpolant.predict(
            x0=x0,
            h=h,
            n_sampling_steps=self.hparams.n_sampling_steps,
            x_conditioning=x_conditioning,
        )
        if self.fit_residual:
            pred = pred + x_conditioning
            x1 = x1 + x_conditioning

        # Get the galaxy ID from the batch (first galaxy in the batch)
        galaxy_id = batch.get('dataset_index', [None])[0] if 'dataset_index' in batch else None
        
        if galaxy_id is not None:
            print(f"Plotting figures for Galaxy {galaxy_id}")
            # Print detailed galaxy information
            print_galaxy_summary(
                galaxy_id.item(),
                suite_key=getattr(self.hparams, "galaxy_info_suite", "DREAMS"),
            )
            # Also log the cosmological parameters for this galaxy
            if h is not None:
                cosmo_params = h[0].cpu().numpy()  # Get parameters for first galaxy in batch
                print(f"Cosmological parameters from batch: {cosmo_params}")

        for i in range(x1.shape[1]):
            x0_idx = self.x0_indices_in_conditioning[i]
            # Get all conditioning variables for this sample, excluding the x0 indices
            all_conditioning = x_conditioning[0]  # Shape: (n_conditioning, height, width)
            
            # Create a mask to exclude x0 indices from additional conditioning
            mask = torch.ones(all_conditioning.shape[0], dtype=torch.bool)
            for idx in self.x0_indices_in_conditioning:
                mask[idx] = False
            
            # Get additional conditioning variables (excluding x0)
            additional_conditioning = all_conditioning[mask]
            
            plot_evals(
                all_conditioning[x0_idx].squeeze(),  # x0 (the main conditioning variable)
                x1[0,i].squeeze(),
                pred[0,i].squeeze(),
                log=self.log_figures,
                fig_name=f"x1_{i}",
                x_conditioning=additional_conditioning if additional_conditioning.shape[0] > 0 else None,  # Pass only additional conditioning variables
                galaxy_id=galaxy_id,  # Pass the galaxy ID
            )

        # Create correlation plots
        velocity_losses = [x["loss"].cpu().numpy() if isinstance(x["loss"], torch.Tensor) else x["loss"] for x in self.validation_step_outputs]
        mse_losses = [x["mse_loss"] for x in self.validation_step_outputs]
        plot_loss_correlation(
            velocity_losses,
            mse_losses,
            log=self.log_figures,
            fig_name="sample_loss_correlation"
        )

        # Calculate and store epoch averages
        avg_velocity_loss = np.mean(velocity_losses)
        avg_mse_loss = np.mean(mse_losses)
        self.epoch_losses.append({
            "epoch": self.current_epoch,
            "velocity_loss": avg_velocity_loss,
            "mse_loss": avg_mse_loss
        })

        # Epoch-wise correlation
        if len(self.epoch_losses) > 1:  # Need at least 2 points
            epoch_velocity_losses = [x["velocity_loss"] for x in self.epoch_losses]
            epoch_mse_losses = [x["mse_loss"] for x in self.epoch_losses]
            plot_loss_correlation(
                epoch_velocity_losses,
                epoch_mse_losses,
                log=self.log_figures,
                fig_name="epoch_loss_correlation"
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            patience=15, 
            factor=0.5,
            mode='min',
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": { 
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                "monitor": "val_loss",
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0,):
        h, x_conditioning, x0, _ = self.get_batch(batch,)
        return self.stochastic_interpolant.predict(
            x0=x0,
            h=h,
            n_sampling_steps=self.n_sampling_steps,
            x_conditioning=x_conditioning,
        ).detach().cpu()