import torch
import torch.nn as nn
from typing import Tuple, Optional, Literal, Dict

import numpy as np

from torchdyn.core import NeuralODE
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
import matplotlib.pyplot as plt

from baryonbridge.models.pixels.cunet import CUNet
from baryonbridge.eval import plot_evals, plot_evals_pointcloud


class ConditionedVelocityModel(nn.Module):
    def __init__(self, velocity_model, h, pixel_conditioning, mask=None,):
        super(ConditionedVelocityModel, self).__init__()
        self.velocity_model = velocity_model
        self.register_buffer('h', h)
        self.register_buffer('pixel_conditioning', pixel_conditioning)
        self.register_buffer('mask', mask)

    def forward(self, t, x, *args, **kwargs):
        return self.velocity_model(t, x, h=self.h, conditioning=self.pixel_conditioning, mask=self.mask,)

class Interpolant(torch.nn.Module):
    def __init__(self, velocity_model,): 
        super().__init__()
        self.velocity_model = velocity_model

    def get_mu_t(self, x0, x1, t):
        return t * x1 + (1 - t) * x0

    def get_gamma_t(self, t):
        return torch.sqrt(2 * t * (1 - t))

    def sample_xt(self, x0, x1, t,):
        t = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        mu_t = self.get_mu_t(x0, x1, t)
        return mu_t

    def compute_loss(
        self,
        x0,
        x1,
        pixel_conditioning=None,
        h=None,
        t=None,
        mask=None,
    ):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        xt = self.sample_xt(x0, x1, t,)
        ut = x1 - x0
        vt = self.velocity_model(t, xt, conditioning=pixel_conditioning,h=h,mask=mask)
        squared_error = (vt - ut) ** 2
        if mask is not None:
            features_dim = squared_error.size(-1) 
            return (squared_error * mask[...,None]).sum() / mask.sum() / features_dim
        return squared_error.mean()

    def predict(
        self,
        x0,
        h=None,
        pixel_conditioning=None,
        mask=None,
        n_sampling_steps=50,
    ):
        conditional_velocity_model = ConditionedVelocityModel(
            velocity_model=self.velocity_model,
            h=h,
            pixel_conditioning=pixel_conditioning,
            mask=mask,
        )
        node = NeuralODE(
            conditional_velocity_model,
            solver="dopri5",
            sensitivity="adjoint",
        )
        with torch.no_grad():
            traj = node.trajectory(
                x0,
                t_span=torch.linspace(0, 1, n_sampling_steps),
            )
        return traj[-1]


class LightningInterpolant(LightningModule):
    def __init__(
        self,
        model_type: Literal["unet", "transformer"] = "unet",
        model_config: Optional[Dict] = None,
        learning_rate=1.0e-4,
        n_sampling_steps=50,
        fit_residual: bool = False,
        condition_on_parameters: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        if model_config is None:
            model_config = {}

        if model_type == 'unet':
            velocity_model = CUNet(
                **model_config,
            )
        elif model_type == 'transformer':
            # Imported here, not at module scope: this repository ships no pointcloud
            # config, and baryonbridge/models/pointclouds/ is not part of the release.
            from baryonbridge.models.pointclouds.transformer import PointTransformer
            velocity_model = PointTransformer(
                **model_config,
            )
        self.save_hyperparameters()
        #self.save_hyperparameters({'velocity_model_hparams': velocity_model.hparams})
        self.validation_step_outputs = []
        self.condition_on_parameters = condition_on_parameters
        self.interpolant = Interpolant(
            velocity_model=velocity_model, 
        )
        self.fit_residual = fit_residual
        

    def setup(self, stage):
        if stage == "fit":
            self.interpolant = self.interpolant.to(self.device)
    
    def get_batch(self, batch,):
        h = batch['theta']
        if not self.condition_on_parameters:
            h = None
        if 'x_conditioning' in batch:
            x_conditioning = batch['x_conditioning']
        else:
            x_conditioning = None
        if self.fit_residual:
            x1 = batch['x1_residual']
        else:
            x1 = batch['x1']
        if 'mask' in batch:
            mask = batch['mask']
        else:
            mask = None
        x0 = batch['x0']
        return h, x_conditioning, x0, x1, mask

    def training_step(self, batch, batch_idx):
        h, x_conditioning, x0, x1, mask = self.get_batch(batch,)

        loss = self.interpolant.compute_loss(x0=x0, x1=x1, h=h, pixel_conditioning=x_conditioning, mask=mask,)
        self.log(
            "train_loss", 
            loss,
            prog_bar=True, 
            sync_dist=True, 
            on_step=True, 
            on_epoch=False, 
            batch_size=batch['theta'].shape[0],
        ) 
        return loss

    def validation_step(self, batch, batch_idx):
        h, x_conditioning, x0, x1, mask = self.get_batch(batch,)
        loss = self.interpolant.compute_loss(x0=x0, x1=x1, h=h, pixel_conditioning=x_conditioning, mask=mask,)
        self.log("val_loss", loss,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch['theta'].shape[0],
        )
        self.validation_step_outputs.append({"loss": loss, "batch": batch})
        return loss

    @rank_zero_only
    def _log_figures(self, batch):
        h, x_conditioning, x0, x1, mask = self.get_batch(batch,)
        pred = self.interpolant.predict(
            x0=x0, 
            h=h, 
            pixel_conditioning=x_conditioning,
            n_sampling_steps=self.hparams.n_sampling_steps, 
            mask=mask,
        )
        if self.fit_residual:
            pred = pred + x_conditioning 
            x1 = x1 + x_conditioning 
        if self.hparams.model_type == 'transformer':
            plot_evals_pointcloud(
                x0[0][mask[0]].squeeze(),
                x1[0][mask[0]].squeeze(),
                pred[0][mask[0]].squeeze(),
                log=True,
            )
        else:
            plot_evals(
                x_conditioning[0].squeeze(),
                x1[0].squeeze(),
                pred[0].squeeze(),
                log=True,
            )
        plt.close()

    def on_validation_epoch_end(self,):
        sample_batch = self.validation_step_outputs[0]["batch"]
        self._log_figures(sample_batch)
        self.validation_step_outputs.clear()

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.AdamW(
            self.interpolant.parameters(), lr=self.hparams.learning_rate
        )
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
        h, x_conditioning, x0, x1, mask = self.get_batch(batch,)
        return self.interpolant.predict(
            x0=x0, h=h, pixel_conditioning=x_conditioning, mask=mask,
        ).detach().cpu()

