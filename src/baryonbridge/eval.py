import matplotlib.pyplot as plt
import torch
from typing import Optional, Tuple
import numpy as np
import Pk_library as PKL
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from baryonbridge.utils.summaries import get_CDFkNN
import io


# plt.style.use("science") # this doesn't work with my matplotlib style file


def log_matplotlib_figure(figure_label: str):
    """log a matplotlib figure to wandb, avoiding plotly

    Args:
        figure_label (str): label for figure
    """
    # Save plot to a buffer, otherwise wandb does ugly plotly
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=300,
    )
    buf.seek(0)
    image = Image.open(buf)
    # Log the plot to wandb
    wandb.log({f"{figure_label}": wandb.Image(image)})

def compute_pk(
    mesh: np.array,
    mesh_2: Optional[np.array] = None,
    boxsize: Optional[float] = 400.0,
    MAS: Optional[str] = None,
    MAS_2: Optional[str] = None,
) -> Tuple[np.array, np.array]:
    """Compute the power spectrum of a mesh, or cross correlations of two meshes if mesh_2 given.
    Works with both 2D (planes/images) and 3D fields.
    
    Args:
        mesh (np.array): overdensity mesh (2D or 3D numpy array)
        mesh_2 (Optional[np.array], optional): overdensity mesh to cross correlate.
            Defaults to None.
        boxsize (Optional[float], optional): size of the periodic box.
            Defaults to 400.0.
        MAS (Optional[str], optional): Mass Assignment Scheme for first mesh.
            'NGP', 'CIC', 'TSC', 'PCS' or 'None'.
        MAS_2 (Optional[str], optional): Mass Assignment Scheme for second mesh.
            'NGP', 'CIC', 'TSC', 'PCS' or 'None'.
            
    Returns:
        Tuple[np.array, np.array]: k values and power spectrum / cross correlation coefficient
    """
    mesh = mesh.astype(np.float32)
    
    dimensions = len(mesh.shape)
    
    if dimensions not in [2, 3]:
        raise ValueError("Input mesh must be either 2D or 3D")
    
    # 2D case (planes/images)
    if dimensions == 2:
        if mesh_2 is not None:
            mesh_2 = mesh_2.astype(np.float32)
            cross_pk = PKL.XPk_plane(mesh, mesh_2, boxsize, MAS, MAS_2,)
            return cross_pk.k, cross_pk.r
        else:
            pk = PKL.Pk_plane(mesh, boxsize, MAS=MAS,)
            return pk.k, pk.Pk
    
    # 3D case
    else:
        if mesh_2 is not None:
            mesh_2 = mesh_2.astype(np.float32)
            cross_pk = PKL.XPk([mesh, mesh_2], boxsize, MAS=[MAS, MAS_2],)
            
            return cross_pk.k3D, cross_pk.XPk[:, 0, 0] / np.sqrt(
                cross_pk.Pk[:, 0, 0] * cross_pk.Pk[:, 0, 1]
            )
        else:
            # Calculate auto-power spectrum for 3D fields
            pk = PKL.Pk(mesh, boxsize, MAS=MAS,)
            return pk.k3D, pk.Pk[:, 0]

def compute_random_baseline_loss(x0: torch.Tensor, x1: torch.Tensor, n_samples: int = 100) -> float:
    """
    Compute the loss if predictions were just random normal values matching target statistics
    
    Args:
        x0 (torch.Tensor): conditioning tensor
        x1 (torch.Tensor): target tensor
        n_samples (int): number of random samples to average over
    
    Returns:
        float: mean squared error loss for random predictions
    """
    # Get statistics of the target distribution
    target_mean = x1.mean()
    target_std = x1.std()
    
    # Generate multiple random predictions
    losses = []
    for _ in range(n_samples):
        # Generate random predictions with same shape as target
        random_pred = torch.randn_like(x1) * target_std + target_mean
        
        # Compute MSE loss
        loss = torch.mean((random_pred - x1) ** 2).item()
        losses.append(loss)
    
    return np.mean(losses)

def plot_evals(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x_pred: torch.Tensor,
    log: bool = False,
    boxsize: float = 25.,
    fig_name: str = 'deltas',
    x_conditioning: Optional[torch.Tensor] = None,
    galaxy_id: Optional[int] = None,
):
    """Plot the evaluation metrics

    Args:
        x0 (torch.Tensor): tensor of final density fields
        x1 (torch.Tensor): tensor of initial density fields
        x_pred (torch.Tensor): tensor of predicted initial density field
        log (bool, optional): whether to log in weights and biases. Defaults to False.
        x_conditioning (Optional[torch.Tensor], optional): additional conditioning variables to plot.
            If provided, should be a tensor with shape (n_conditioning, height, width) or (n_conditioning, depth, height, width)
        galaxy_id (Optional[int], optional): identifier for the galaxy being plotted. Defaults to None.
    """
    # Add galaxy ID to figure name if provided
    if galaxy_id is not None:
        fig_name = f"{fig_name}_galaxy_{galaxy_id}"
    
    plt.grid(alpha=0.5)
    _, edges, _ = plt.hist(
        x0.cpu().numpy().reshape(-1),
        label="x0",
        density=True,
        alpha=0.5,
        bins=300,
    )
    _, edges, _ = plt.hist(
        x1.cpu().numpy().reshape(-1),
        label="x1",
        density=True,
        alpha=0.5,
        bins=100,
    )
    plt.hist(
        x_pred.cpu().numpy().reshape(-1),
        label="Pred x1",
        bins=edges,
        alpha=0.5,
        density=True,
    )
    plt.xlim(-3., 3.)
    plt.legend()
    if galaxy_id is not None:
        plt.title(f"Galaxy {galaxy_id}")
    if log:
        log_matplotlib_figure(f"{fig_name}_histogram")

    plt.close()
    plt.grid(alpha=0.5)
    k, dm_power = compute_pk(x0.cpu().numpy(), MAS=None, boxsize=boxsize,)
    k, true_power = compute_pk(x1.cpu().numpy(), MAS=None,boxsize=boxsize,)
    k, pred_power = compute_pk(x_pred.cpu().numpy(), MAS=None, boxsize=boxsize,)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}, sharex=True)

    axs[0].loglog(k, dm_power, label="x0")
    axs[0].loglog(k, true_power, label="x1")
    axs[0].loglog(k, pred_power, label="Pred x1")

    axs[1].set_ylim(0.8,1.2)
    axs[0].legend()
    axs[1].axhline(y=1.0, color="gray", alpha=0.5)
    axs[1].semilogx(k, np.sqrt(pred_power / true_power))
    axs[1].set_xlabel('k')
    if galaxy_id is not None:
        fig.suptitle(f"Galaxy {galaxy_id}")
    if log:
        log_matplotlib_figure(f"{fig_name}_pks")
    plt.close()
    plt.grid(alpha=0.5)
    k, cross_pred = compute_pk(
        x1.cpu().numpy(), mesh_2=x_pred.cpu().numpy(),
        MAS=None, MAS_2=None, boxsize=boxsize,
    )
    k, cross_dm= compute_pk(
        x1.cpu().numpy(), mesh_2=x0.cpu().numpy(),
        MAS=None, MAS_2=None, boxsize=boxsize,
    )

    plt.axhline(y=0.0, color="gray", alpha=0.5)
    plt.axhline(y=1.0, color="gray", alpha=0.5)
    plt.semilogx(k, cross_pred, label="pred x1")
    plt.semilogx(k, cross_dm, label="x0")
    plt.legend()
    if galaxy_id is not None:
        plt.title(f"Galaxy {galaxy_id}")
    if log:
        log_matplotlib_figure(f"{fig_name}_cross_corrs")
    plt.close()

    if len(x1.shape) == 3:
        x1_plot = x1[:10].sum(axis=0)
        x0_plot = x0[:10].sum(axis=0)
        x_pred_plot = x_pred[:10].sum(axis=0)
    else:
        x1_plot = x1
        x0_plot = x0
        x_pred_plot = x_pred

    vmin, vmax = x1_plot.min(), x1_plot.max()
    
    # Determine the number of panels based on whether x_conditioning is provided
    if x_conditioning is not None:
        # Count the number of conditioning variables (excluding x0 which is already shown)
        n_conditioning = x_conditioning.shape[0] if len(x_conditioning.shape) > 2 else 1
        n_panels = 3 + n_conditioning  # x0 + conditioning variables + x1 + pred
        fig, ax = plt.subplots(1, n_panels, figsize=(5*n_panels, 5))
        
        # Plot x0
        ax[0].imshow(x0_plot.cpu().numpy())
        ax[0].set_title("x0")
        
        # Plot conditioning variables
        if len(x_conditioning.shape) > 2:
            for i in range(n_conditioning):
                cond_plot = x_conditioning[i]
                if len(cond_plot.shape) == 3:
                    cond_plot = cond_plot[:10].sum(axis=0)
                ax[i+1].imshow(cond_plot.cpu().numpy())
                ax[i+1].set_title(f"Conditioning {i+1}")
        else:
            ax[1].imshow(x_conditioning.cpu().numpy())
            ax[1].set_title("Conditioning")
        
        # Plot x1 and prediction
        ax[-2].imshow(x1_plot.cpu().numpy(), vmin=vmin, vmax=vmax)
        ax[-2].set_title("x1")
        ax[-1].imshow(x_pred_plot.cpu().numpy(), vmin=vmin, vmax=vmax)
        ax[-1].set_title("pred x1")
    else:
        # Original 3-panel layout
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(x0_plot.cpu().numpy())
        ax[1].imshow(x1_plot.cpu().numpy(), vmin=vmin, vmax=vmax)
        ax[2].imshow(x_pred_plot.cpu().numpy(), vmin=vmin, vmax=vmax)
        ax[0].set_title("x0")
        ax[1].set_title("x1")
        ax[2].set_title("pred x1")
    
    if galaxy_id is not None:
        fig.suptitle(f"Galaxy {galaxy_id}")
    if log:
        log_matplotlib_figure(f"{fig_name}_deltas")
    plt.close()

    # Add the baseline comparison at the end
    if log:
        baseline_loss = compute_random_baseline_loss(x0, x1)
        model_loss = torch.mean((x_pred - x1) ** 2).item()
        
        # Log the raw values to wandb
        log_dict = {
            f"{fig_name}_model_loss": model_loss,
            f"{fig_name}_random_baseline_loss": baseline_loss,
            f"{fig_name}_loss_ratio": model_loss / baseline_loss,
        }
        
        # Add galaxy ID to logging if provided
        if galaxy_id is not None:
            log_dict[f"{fig_name}_galaxy_id"] = galaxy_id
            
        wandb.log(log_dict)



def plot_evals_pointcloud(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x1_pred: torch.Tensor,
    log: bool = False,
):
    # undo standarization
    x0 = (x0.cpu().numpy()*0.25 + 0.5) 
    x1 = (x1.cpu().numpy()*0.25 + 0.5) 
    x1_pred = (x1_pred.cpu().numpy()*0.25 + 0.5) #% 1.

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].plot(x0[:,0], x0[:,1], linestyle='', marker='o', markersize=1, label='x0')
    ax[1].plot(x1_pred[:,0], x1_pred[:,1], linestyle='', marker='o', markersize=1, label='x1 pred')
    ax[2].plot(x1[:,0], x1[:,1], linestyle='', marker='o', markersize=1, label='x1')
    ax[0].set_title('x0')
    ax[1].set_title('x1 pred')
    ax[2].set_title('x1')
    if log:
        log_matplotlib_figure(f"pointclouds")
    plt.close()
    # plot knns include x0, x1, x_pred
    randoms = np.random.rand(len(x0), 3)
    r_bins = np.linspace(0, 0.4, 10)
    ks = [1,3,5]
    x0 = x0 % 1.
    x1 = x1 % 1.
    x1_pred = x1_pred % 1.
    x0_knn = get_CDFkNN(r_bins, x0[...,:3], randoms, 1.,k=ks)
    x1_knn = get_CDFkNN(r_bins, x1[...,:3], randoms, 1., k=ks)
    x1_pred_knn = get_CDFkNN(r_bins, x1_pred[...,:3], randoms, 1., k=ks)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    ax[0].plot(r_bins, x0_knn[0], label='x0', linestyle='dashed', color='gray')
    ax[0].plot(r_bins, x1_knn[0], label='x1')
    ax[0].plot(r_bins, x1_pred_knn[0], label='x1 pred')

    ax[1].plot(r_bins, x0_knn[1], label='x0', linestyle='dashed', color='gray')
    ax[1].plot(r_bins, x1_knn[1], label='x1')
    ax[1].plot(r_bins, x1_pred_knn[1], label='x1')

    ax[2].plot(r_bins, x0_knn[1], label='x0', linestyle='dashed', color='gray')
    ax[2].plot(r_bins, x1_knn[1], label='x1')
    ax[2].plot(r_bins, x1_pred_knn[1], label='x1')


    ax[0].legend()
    for i, k in enumerate(ks):
        ax[i].set_title(f'k = {k}')
    if log:
        log_matplotlib_figure(f"knns")
    plt.close()

def plot_loss_correlation(train_loss, eval_loss, log=True, fig_name='loss_correlation'):
    """Plot correlation between velocity matching loss and final MSE loss
    
    Args:
        train_loss (list): velocity matching losses
        eval_loss (list): final MSE losses
        log (bool): whether to log to wandb
        fig_name (str): name for the figure
    """
    # Convert losses to numpy arrays if they are tensors
    train_loss = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in train_loss]
    eval_loss = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in eval_loss]
    
    # Convert lists to numpy arrays
    train_loss = np.array(train_loss)
    eval_loss = np.array(eval_loss)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(train_loss, eval_loss, alpha=0.5)
    plt.xlabel('Velocity Matching Loss')
    plt.ylabel('Final MSE Loss')
    plt.yscale('log')
    plt.xscale('log')
    
    # Add correlation coefficient and number of points
    corr = np.corrcoef(train_loss, eval_loss)[0,1]
    n_points = len(train_loss)
    plt.title(f'Loss Correlation (r = {corr:.3f}, n = {n_points})')
    
    if log:
        log_matplotlib_figure(f"{fig_name}")
        wandb.log({
            f"{fig_name}_correlation": corr,
            f"{fig_name}_n_points": n_points,
        })
    plt.close()


