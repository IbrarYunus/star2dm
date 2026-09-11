from pathlib import Path
from lightning.pytorch.cli import LightningCLI
from baryonbridge.interpolants.interpolant import LightningInterpolant
from baryonbridge.interpolants.stochastic import LightningStochasticInterpolant
from baryonbridge.data.dreams import DREAMSDataModule
import torch
torch.set_float32_matmul_precision('medium')

# Write one class for interpolants? may be stochastic or not

def cli_main():
    config_path = Path(__file__).parent.parent / "configs/dm_2d.yaml"
    print(f"Using config file: {config_path}")

    cli = LightningCLI(
        # model_class=LightningStochasticInterpolant, 
        #LightningInterpolant, 
        # datamodule_class=DREAMSDataModule,
        save_config_callback=None,
        # parser_kwargs={
        #    "default_config_files": [str(config_path)],
        # },
        subclass_mode_model=True, 
        subclass_mode_data=True,
    )

if __name__ == "__main__":
    # command to run: python train.py fit --config ../configs/dm_2d.yaml
    # resuming from a checkpoint: python train.py fit --config ../configs/dm_2d.yaml --ckpt_path path/to/checkpoint.ckpt
    cli_main()
