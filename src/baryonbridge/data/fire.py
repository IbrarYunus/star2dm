from typing import Optional, Callable, Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from torchvision.datasets import VisionDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch

from baryonbridge.config import CLIP_PROFILES, NORM_PROFILES, merged_norm_dict
from baryonbridge.data.transforms import Flip, Permutate, Compose


logging.basicConfig(level=logging.WARNING)

# map_type: {resolution: , std: mean:, min: }

# For +/- 8; 3D is not updated yet
NORM_DICT = {
    '3D': {
        'T': {
            256: {'mean': 3.84, 'std': 1.01, 'min': 0.},
            128: {'mean': 3.85, 'std': 1.01, 'min': 205.},
        },
        'Mcdm': {
            256: {'mean': 10.014, 'std': 0.55, 'min': 0.},
            128: {'mean': 10.04, 'std': 0.55, 'min': 5855244.}
        },
        'Mgas': {
            128: {'mean': 9.5, 'std': 0.56, 'min': 1218244.4}
        },
        'Vgas': {
            128: {'mean': 139.9, 'std': 78.8, 'min': 1.45}
        },
        'Vcdm': {
            128: {'mean': 145.7, 'std': 88.45, 'min': 1.68}
        },
    },
    '2D': {
        # 'MDM': {128: {'mean': 2.00, 'std': 2.96, 'min': 0.}}, # old values from pulling values from the snpashots
        # 'rhoDM': {128: {'mean': 4.62, 'std': 3.93, 'min': 0.}},
        # 'Mstar': {128: {'mean': 3.39, 'std': 2.86, 'min': 0.}},
        'MDM': {128: {'mean': 2.00, 'std': 2.96, 'min': 0.}}, # using the kdtree reconstructed density map
        'Mstar': {128: {'mean': 3.39, 'std': 2.86, 'min': 0.}}, # double check the normalization for Mstar
        # 'rhoDM': {128: {'mean': 7.2178, 'std': 0.4273, 'min': 6.044},
        #     32: {'mean': 7.2178, 'std': 0.4273, 'min': 6.044}
        # }, # used for fixed +/-8 kpc range
        # 'rhostar': {128: {'mean': 6.8233, 'std': 0.8626, 'min': 3.6926},
        #     32: {'mean': 6.8233, 'std': 0.8626, 'min': 3.6926}
        # }, # used for fixed +/-8 kpc range
        'rhoDM': {128: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122},
            32: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122}
        }, # used for fixed +/-32 kpc range
        # 'rhoDM': {128: {'mean': 6.0528, 'std': 0.4842, 'min': 4.6043},
        #     32: {'mean': 6.0528, 'std': 0.4842, 'min': 4.6043} 
        # }, # used for small redshift variation range
        # 'rhoDM': {128: {'mean': 6.0832, 'std': 0.5055, 'min': 4.4544},
        #     32: {'mean': 6.0832, 'std': 0.5055, 'min': 4.4544} 
        # }, # used for medium redshift variation range
        # 'rhoDM': {128: {'mean': 6.1720, 'std': 0.6126, 'min': 4.3779},
        #     32: {'mean': 6.1720, 'std': 0.6126, 'min': 4.3779} 
        # }, # used for large redshift variation range and for all redshift ranges after v39
        # 'rhoDMrand': {128: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122},
        #     32: {'mean': 6.0495, 'std': 0.4818, 'min': 4.5122}
        # }, # used for fixed +/-32 kpc range
        # 'rhoDMrand': {128: {'mean': 6.0528, 'std': 0.4842, 'min': 4.6043},
        #     32: {'mean': 6.0528, 'std': 0.4842, 'min': 4.6043} 
        # }, # used for small redshift variation range
        # 'rhoDMrand': {128: {'mean': 6.0832, 'std': 0.5055, 'min': 4.4544},
        #     32: {'mean': 6.0832, 'std': 0.5055, 'min': 4.4544} 
        # }, # used for medium redshift variation range
        'rhoDMrand': {128: {'mean': 6.1720, 'std': 0.6126, 'min': 4.3779},
            32: {'mean': 6.1720, 'std': 0.6126, 'min': 4.3779} 
        }, # used for large redshift variation range and for all redshift ranges after v39
        'rhostar': {128: {'mean': 4.5496, 'std': 1.1889, 'min': 0.7828},
            32: {'mean': 4.5496, 'std': 1.1889, 'min': 0.7828}
        }, # used for fixed +/-32 kpc range
        # 'EuVISnoise': {128: {'mean': 1.09e25, 'std': 1.29e26, 'min': -2.34e25}}, # linear noise map from Euclid VIS from synthesizer; used for fixed +/-32 kpc range
        # 'EuVISnoiselog10': {128: {'mean': 25.44, 'std': 0.1924, 'min': 19.36}}, # log10 noise map from Euclid VIS from synthesizer; used for fixed +/-32 kpc range
        # 'EuVISnoise': {128: {'mean': 1.10e25, 'std': 1.28e26, 'min': -3.21e25}}, # linear noise map from Euclid VIS from synthesizer; used for small redshift variation range
        # 'EuVISnoiselog10': {128: {'mean': 25.56, 'std': 0.1621, 'min': 19.48}}, # log10 noise map from Euclid VIS from synthesizer; used for small redshift variation range
        # 'EuVISnoise': {128: {'mean': 1.09e25, 'std': 1.25e26, 'min': -7.17e25}}, # linear noise map from Euclid VIS from synthesizer; used for medium redshift variation range
        # 'EuVISnoiselog10': {128: {'mean': 25.89, 'std': 0.1064, 'min': 19.84}}, # log10 noise map from Euclid VIS from synthesizer; used for medium redshift variation range
        # 'EuVISnoise': {128: {'mean': 1.05e25, 'std': 1.24e26, 'min': -1.50e25}}, # linear noise map from Euclid VIS from synthesizer; used for large redshift variation range
        # 'EuVISnoiselog10': {128: {'mean': 26.20, 'std': 0.0738, 'min': 20.17}}, # log10 noise map from Euclid VIS from synthesizer; used for large redshift variation range
        'EuVISnoiselog10': {128: {'mean': 26.225, 'std': 0.0667, 'min': 24.97}}, # log10 noise map from Euclid VIS from synthesizer; used for all redshift variation range after v39
        'EuVISfluxnoiselog10': {128: {'mean': 2.1312, 'std': 0.1084, 'min': -3.8622}}, # log10 flux noise map from Euclid VIS from synthesizer; used for large redshift variation range
        'Uavg': {128: {'mean': 0.23, 'std': 109.78, 'min': -537.0}},
        'Vavg': {128: {'mean': 0.99, 'std': 106.03, 'min': -516.0}},
        'Wavg': {128: {'mean': -0.15, 'std': 22.20, 'min': -341.0}},
        'vRavg': {128: {'mean': 0.61, 'std': 26.61, 'min': -351.51}},
        'vTavg': {128: {'mean': 125.79, 'std': 82.45, 'min': -310.0}},
    }
}


class FIRE(VisionDataset):
    @staticmethod
    def create_file_suffix_2d(xmin: float = -32, xmax: float = 32, 
                             ymin: float = -32, ymax: float = 32, 
                             zmin_2d: float = -32, zmax_2d: float = 32) -> str:
        """
        Create a file suffix for 2D maps.
        
        Args:
            xmin, xmax, ymin, ymax: Grid boundaries for x and y
            zmin_2d, zmax_2d: Z-range for 2D map generation (controls which particles are included)
            
        Returns:
            String suffix in format: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z2d range)
        """
        return f"x{xmin:.0f}_{xmax:.0f}_y{ymin:.0f}_{ymax:.0f}_z{zmin_2d:.0f}_{zmax_2d:.0f}"
    
    @staticmethod
    def create_file_suffix_3d(xmin: float = -32, xmax: float = 32, 
                             ymin: float = -32, ymax: float = 32, 
                             zmin: float = -32, zmax: float = 32) -> str:
        """
        Create a file suffix for 3D grids.
        
        Args:
            xmin, xmax, ymin, ymax: Grid boundaries for x and y
            zmin, zmax: Z-range for 3D grids
            
        Returns:
            String suffix in format: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z range)
        """
        return f"x{xmin:.0f}_{xmax:.0f}_y{ymin:.0f}_{ymax:.0f}_z{zmin:.0f}_{zmax:.0f}"
    
    @staticmethod
    def create_file_suffix(xmin: float = -32, xmax: float = 32, 
                          ymin: float = -32, ymax: float = 32, 
                          zmin: float = -32, zmax: float = 32,
                          zmin_2d: float = -32, zmax_2d: float = 32,
                          dims: int = 2) -> str:
        """
        Create a file suffix from grid boundary parameters.
        
        Args:
            xmin, xmax, ymin, ymax: Grid boundaries for x and y
            zmin, zmax: Z-range for 3D grids
            zmin_2d, zmax_2d: Z-range for 2D map generation (controls which particles are included)
            dims: Dimension (2 or 3) to determine which z-range to include
            
        Returns:
            String suffix in format: 
            - 2D: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z2d range)
            - 3D: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z range)
        """
        if dims == 2:
            return FIRE.create_file_suffix_2d(xmin, xmax, ymin, ymax, zmin_2d, zmax_2d)
        else:  # dims == 3
            return FIRE.create_file_suffix_3d(xmin, xmax, ymin, ymax, zmin, zmax)
    
    def __init__(
        self,
        root: str = "./data/FIRE_data",
        fire_run: str = "FIREm12",  # 'FIREm12' or 'FIREm11' — subdirectory under 2D_maps/ (and 3D_grids/)
        redshift: float = 0.00,
        nsnap: int = 600,
        transform: Optional[Callable] = None,
        parameters: List[str] = ["Om", "s8", "SN1", "SN2", "BHFF"],
        resolution: int = 128,
        idx_list: Optional[List[int]] = None,
        dataset: str = "SB5",  # kept for API compatibility, not used in paths
        x1_fields: List[str] = ["rhoDM"],
        x_conditioning_fields: List[str] = ["EuVISfluxnoiselog10"],
        dims: int = 2,
        file_suffix: str = "",  # e.g. 'csv_fov_z0p050_0p650'
        randomize_theta: bool = False,
        randomize_x_conditioning: bool = False,
        randomize_x1: bool = False,
        random_seed: Optional[int] = None,
        randomization_mode: str = "dynamic",
        norm_profile: str = "fixed32",  # NORM_DICT overlay; see baryonbridge/config.py
        clip_profile: str = "none",  # post-normalization clip; see baryonbridge/config.py
    ):
        super().__init__(
            root,
            transform=transform,
        )
        self.root = Path(self.root)
        # OOD data must be normalized with the constants the model was trained on, so the
        # profile here is selected by the model's version tag, exactly as for DREAMS.
        if norm_profile not in NORM_PROFILES:
            raise ValueError(
                f"Unknown norm_profile {norm_profile!r}. "
                f"Available: {sorted(NORM_PROFILES)}"
            )
        self.norm_profile = norm_profile
        self.norm_dict = merged_norm_dict(NORM_DICT, NORM_PROFILES[norm_profile])
        if clip_profile not in CLIP_PROFILES:
            raise ValueError(
                f"Unknown clip_profile {clip_profile!r}. "
                f"Available: {sorted(CLIP_PROFILES)}"
            )
        self.clip_profile = clip_profile
        self.clip_spec = CLIP_PROFILES[clip_profile]
        self.fire_run = fire_run  # FIREm12 or FIREm11
        self.redshift = redshift
        self.idx_list = idx_list
        self.resolution = resolution
        self.dataset = dataset
        self.x1_fields = x1_fields
        self.x_conditioning_fields = x_conditioning_fields
        self.dims = dims
        self.file_suffix = file_suffix  # Store file suffix
        self.parameters = parameters
        self.randomize_theta = randomize_theta
        self.randomize_x_conditioning = randomize_x_conditioning
        self.randomize_x1 = randomize_x1
        self.random_seed = random_seed
        self.randomization_mode = randomization_mode
        # Initialize random number generator if randomization is enabled
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
        self.y = {}
        for mt in x1_fields + x_conditioning_fields:
            self._load_images(fire_run=fire_run, resolution=resolution, map_type=mt)
        self._load_parameters(fire_run=fire_run, parameters=parameters)
        
        # Create fixed permutation arrays if using fixed randomization mode
        if randomization_mode == 'fixed':
            dataset_size = len(self.theta)
            if randomize_theta:
                self.theta_permutation = self.rng.permutation(dataset_size)
            else:
                self.theta_permutation = None
            if randomize_x_conditioning:
                self.x_conditioning_permutation = self.rng.permutation(dataset_size)
            else:
                self.x_conditioning_permutation = None
            if randomize_x1:
                self.x1_permutation = self.rng.permutation(dataset_size)
            else:
                self.x1_permutation = None
            
            # Save randomization order to file
            save_dict = {
                'theta_permutation': self.theta_permutation,
                'x_conditioning_permutation': self.x_conditioning_permutation,
                'x1_permutation': self.x1_permutation,
                'dataset_size': dataset_size,
                'random_seed': random_seed,
                'randomize_theta': randomize_theta,
                'randomize_x_conditioning': randomize_x_conditioning,
                'randomize_x1': randomize_x1,
                'fire_run': self.fire_run,
                'dataset': self.dataset,
                'file_suffix': file_suffix,
            }
            # Create filename with relevant info
            seed_str = f"_seed{random_seed}" if random_seed is not None else "_noseed"
            suffix_str = f"_{file_suffix}" if file_suffix else ""
            perm_filename = self.root / f"permutations_{self.fire_run}_{self.dataset}{suffix_str}{seed_str}.npz"
            perm_filename.parent.mkdir(parents=True, exist_ok=True)
            np.savez(perm_filename, **save_dict)
            logging.info(f"Saved randomization order to: {perm_filename}")
        else:
            self.theta_permutation = None
            self.x_conditioning_permutation = None
            self.x1_permutation = None
 
    def __len__(
        self,
    ):
        return len(self.theta)

    def _load_images(self, fire_run: str, resolution: int, map_type: str = "Mtot"):
        norm_dict = self.norm_dict[f"{self.dims}D"][map_type][self.resolution]
        mean, std, min_val = norm_dict["mean"], norm_dict["std"], norm_dict["min"]

        # Construct file path with suffix if provided
        if self.file_suffix:
            suffix_part = f"_{self.file_suffix}"
        else:
            suffix_part = ""

        # FIRE layout: root/2D_maps/m12/ or root/2D_maps/m11/
        if self.dims == 3:
            file_path = self.root / f"3D_grids/{fire_run}/Grids_{map_type}_{fire_run}_{resolution}_z={self.redshift:.2f}{suffix_part}.npy"
        elif self.dims == 2:
            file_path = self.root / f"2D_maps/{fire_run}/Maps_{map_type}_{fire_run}_{resolution}_z={self.redshift:.2f}{suffix_part}.npy"

        print(f"Loading {map_type} from: {file_path}")

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Make sure you have generated the data with the correct file suffix: '{self.file_suffix}'\n"
                f"You can use TNG.create_file_suffix() to generate the correct suffix from grid parameters."
            )
            
        data = np.load(file_path)
        print(f'data shape for {map_type}: {data.shape}')
        if self.idx_list is not None:
            if max(self.idx_list) >= len(data):
                idx_list = [idx for idx in self.idx_list if idx < len(data)]
                logging.warning(f"Index list contains indices out of bounds for dataset {fire_run}. Truncating idx_list to fit the data size ({len(idx_list)}).")
            else:
                idx_list = self.idx_list
            data = data[idx_list]
        # make sure data is positive
        # if map_type not in ['Vgas', 'Vcdm']:
        #     if min_val == 0.:
        #         data += 1.e-6 
        #     elif min_val < 0.:
        #         data -= 1.05*min_val
        #     data = np.log10(data)[:,None]
        # else:
        #     data = data[:,None]

        # check the range of the data
        print('before normalization')
        print(f'min value of {map_type}: {data.min()}')
        print(f'max value of {map_type}: {data.max()}')
        print(f'mean value of {map_type}: {data.mean()}')
        print(f'std value of {map_type}: {data.std()}')
        self.y[map_type] =  torch.tensor((data - mean) / std, dtype=torch.float32)
        # self.y[map_type] =  torch.tensor(data, dtype=torch.float32)

        

        # if there is a inf in the data, print the index
        inf_idx = torch.where(torch.isinf(self.y[map_type]))[0]
        if len(inf_idx) > 0:
            print(f'inf indices of {map_type}: {inf_idx}')

        # set the inf to the max value
        self.y[map_type][inf_idx] = self.y[map_type].max()

        # Optional post-normalization clip, selected by clip_profile (see config.py).
        # Applied after the inf repair so any repaired value is bounded too. Bounds are in
        # sigma of the normalized map and may be one-sided (None = leave that side alone).
        # OOD data must be clipped exactly as the model's training data was, so the profile
        # is resolved from the model's version tag by the inference scripts.
        clip_bounds = self.clip_spec.get(f"{self.dims}D", {}).get(map_type)
        if clip_bounds is not None:
            lo, hi = clip_bounds
            n_lo = int((self.y[map_type] < lo).sum()) if lo is not None else 0
            n_hi = int((self.y[map_type] > hi).sum()) if hi is not None else 0
            self.y[map_type] = torch.clamp(self.y[map_type], min=lo, max=hi)
            print(
                f"clip_profile={self.clip_profile}: clamped {map_type} to "
                f"[{lo}, {hi}] sigma ({n_lo} pixels below, {n_hi} above)"
            )

        # check the range of the data again
        print('after normalization')
        print(f'min value of {map_type}: {self.y[map_type].min()}')
        print(f'max value of {map_type}: {self.y[map_type].max()}')
        print(f'mean value of {map_type}: {self.y[map_type].mean()}')
        print(f'std value of {map_type}: {self.y[map_type].std()}')



    def _load_parameters(
        self,
        fire_run: str,
        parameters= ['Om', 's8','SN1','SN2','BHFF',],
    ):
        # Dataset size from first loaded map
        n_samples = next(iter(self.y.values())).shape[0]
        params_path = self.root / f"Parameters/{fire_run}/CDM_TNG_MW_SB5.txt"
        if parameters is not None and params_path.exists():
            column_names = ["Om", "s8", "SN1", "SN2", "BHFF"]
            params = pd.read_csv(
                params_path,
                sep=" ",
                names=column_names,
                comment="#",
            )[parameters]
            if self.idx_list is not None:
                if max(self.idx_list) >= len(params):
                    logging.warning(
                        f"Index list contains indices out of bounds for {fire_run}. Truncating idx_list to fit the data size."
                    )
                    idx_list = [idx for idx in self.idx_list if idx < len(params)]
                else:
                    idx_list = self.idx_list
                params = params.iloc[idx_list].values
            else:
                params = params.values
            self.theta = torch.tensor(params.astype(np.float32))
        else:
            # FIRE has no cosmology parameters; use dummy theta (zeros) so __len__ and __getitem__ work
            # if parameters is not None:
            #     n_params = len(parameters)
            # else:
            #     n_params = 1
            # self.theta = torch.zeros((n_samples, n_params), dtype=torch.float32)
            # just error out
            raise ValueError(f"Parameters are not available for FIRE dataset: {params_path}")

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[np.array, np.array, np.array]:
        # Determine indices for each component (randomize if requested)
        if self.randomization_mode == 'fixed':
            # Use fixed permutation arrays
            theta_idx = self.theta_permutation[index] if self.randomize_theta else index
            x_conditioning_idx = self.x_conditioning_permutation[index] if self.randomize_x_conditioning else index
            x1_idx = self.x1_permutation[index] if self.randomize_x1 else index
        else:
            # Dynamic randomization: new random index on each access
            dataset_size = len(self.theta)
            theta_idx = self.rng.randint(0, dataset_size) if self.randomize_theta else index
            x_conditioning_idx = self.rng.randint(0, dataset_size) if self.randomize_x_conditioning else index
            x1_idx = self.rng.randint(0, dataset_size) if self.randomize_x1 else index
        
        theta = self.theta[theta_idx]
        x_conditioning = torch.concatenate([self.y[conditioning_map_type][x_conditioning_idx].unsqueeze(0) 
                                          for conditioning_map_type in self.x_conditioning_fields])
        x1 = torch.concatenate([self.y[map_type][x1_idx].unsqueeze(0) 
                              for map_type in self.x1_fields])
        if self.transform is not None:
            x1, x_conditioning = self.transform((x1, x_conditioning))
        
        # Get the original dataset index (before idx_list filtering)
        if self.idx_list is not None:
            original_index = self.idx_list[index]
        else:
            original_index = index
            
        return {
            'theta': theta,
            'x_conditioning': x_conditioning,
            'x1': x1,
            'x_residual': (x1 - x_conditioning),
            'dataset_index': original_index,  # Add the original dataset index
        }
        
class FIREDataModule(LightningDataModule):
    @staticmethod
    def create_file_suffix_2d(xmin: float = -32, xmax: float = 32, 
                             ymin: float = -32, ymax: float = 32, 
                             zmin_2d: float = -32, zmax_2d: float = 32) -> str:
        """
        Create a file suffix for 2D maps.
        
        Args:
            xmin, xmax, ymin, ymax: Grid boundaries for x and y
            zmin_2d, zmax_2d: Z-range for 2D map generation (controls which particles are included)
            
        Returns:
            String suffix in format: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z2d range)
        """
        return FIRE.create_file_suffix_2d(xmin, xmax, ymin, ymax, zmin_2d, zmax_2d)
    
    @staticmethod
    def create_file_suffix_3d(xmin: float = -32, xmax: float = 32, 
                             ymin: float = -32, ymax: float = 32, 
                             zmin: float = -32, zmax: float = 32) -> str:
        """
        Create a file suffix for 3D grids.
        
        Args:
            xmin, xmax, ymin, ymax: Grid boundaries for x and y
            zmin, zmax: Z-range for 3D grids
            
        Returns:
            String suffix in format: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z range)
        """
        return FIRE.create_file_suffix_3d(xmin, xmax, ymin, ymax, zmin, zmax)
    
    @staticmethod
    def create_file_suffix(xmin: float = -32, xmax: float = 32, 
                          ymin: float = -32, ymax: float = 32, 
                          zmin: float = -32, zmax: float = 32,
                          zmin_2d: float = -32, zmax_2d: float = 32,
                          dims: int = 2) -> str:
        """
        Create a file suffix from grid boundary parameters.
        
        Args:
            xmin, xmax, ymin, ymax: Grid boundaries for x and y
            zmin, zmax: Z-range for 3D grids
            zmin_2d, zmax_2d: Z-range for 2D map generation (controls which particles are included)
            dims: Dimension (2 or 3) to determine which z-range to include
            
        Returns:
            String suffix in format: 
            - 2D: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z2d range)
            - 3D: x{min}_{max}_y{min}_{max}_z{min}_{max} (using z range)
        """
        if dims == 2:
            return FIRE.create_file_suffix_2d(xmin, xmax, ymin, ymax, zmin_2d, zmax_2d)
        else:  # dims == 3
            return FIRE.create_file_suffix_3d(xmin, xmax, ymin, ymax, zmin, zmax)
    
    def __init__(
        self,
        root: str = "./data/FIRE_data",
        fire_run: str = "FIREm12",  # 'FIREm12' or 'FIREm11' — subdirectory under 2D_maps/ (and 3D_grids/)
        dims: int = 2,
        resolution: int = 128,
        n_train: int = 980,
        n_val: int = 10,
        n_test: int = 10,
        batch_size: int = 16,
        num_workers: int = 2,
        redshift: float = 0.0,
        x1_fields: List[str] = ["rhoDM"],
        x_conditioning_fields: List[str] = ["EuVISfluxnoiselog10"],
        file_suffix: str = "",  # e.g. 'csv_fov_z0p050_0p650'
        grid: Optional[dict] = None,
        randomize_theta: bool = False,
        randomize_x_conditioning: bool = False,
        randomize_x1: bool = False,
        random_seed: Optional[int] = None,
        randomization_mode: str = "dynamic",
        norm_profile: str = "fixed32",  # NORM_DICT overlay matching file_suffix; see config.py
        clip_profile: str = "none",  # post-normalization conditioning clip; see config.py
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_transform = Compose([
            Flip(dims), Permutate(dims)],
        )

    def setup(self, stage=None):
        # Define test indices
        test_idx = range(self.hparams.n_test)
        val_idx = range(self.hparams.n_test, self.hparams.n_test + self.hparams.n_val)
        train_idx = range(self.hparams.n_test + self.hparams.n_val, self.hparams.n_test + self.hparams.n_val + self.hparams.n_train)
        print(f'train_idx: {train_idx}')
        print(f'val_idx: {val_idx}')
        print(f'test_idx: {test_idx}')
        
        # Compute file_suffix from grid parameters if not provided
        file_suffix = self.hparams.file_suffix
        if not file_suffix and hasattr(self.hparams, 'grid') and self.hparams.grid:
            file_suffix = self.create_file_suffix(**self.hparams.grid, dims=self.hparams.dims)
            print(f"Computed file_suffix from grid parameters: {file_suffix}")
        else:
            print(f"Using provided file_suffix: {file_suffix}")
        
        print(f"Data files will be loaded with suffix: {file_suffix if file_suffix else '(default)'}")
        
        # Create datasets
        self.train_data = FIRE(
            root=self.hparams.root,
            fire_run=self.hparams.fire_run,
            dims=self.hparams.dims,
            transform=self.train_transform,
            idx_list=train_idx,
            resolution=self.hparams.resolution,
            redshift=self.hparams.redshift,
            x1_fields=self.hparams.x1_fields,
            x_conditioning_fields=self.hparams.x_conditioning_fields,
            file_suffix=file_suffix,
            randomize_theta=getattr(self.hparams, "randomize_theta", False),
            randomize_x_conditioning=getattr(self.hparams, "randomize_x_conditioning", False),
            randomize_x1=getattr(self.hparams, "randomize_x1", False),
            random_seed=getattr(self.hparams, "random_seed", None),
            randomization_mode=getattr(self.hparams, "randomization_mode", "dynamic"),
            norm_profile=getattr(self.hparams, "norm_profile", "fixed32"),
            clip_profile=getattr(self.hparams, "clip_profile", "none"),
        )

        self.val_data = FIRE(
            root=self.hparams.root,
            fire_run=self.hparams.fire_run,
            dims=self.hparams.dims,
            transform=None,
            idx_list=val_idx,
            resolution=self.hparams.resolution,
            redshift=self.hparams.redshift,
            x1_fields=self.hparams.x1_fields,
            x_conditioning_fields=self.hparams.x_conditioning_fields,
            file_suffix=file_suffix,
            randomize_theta=getattr(self.hparams, "randomize_theta", False),
            randomize_x_conditioning=getattr(self.hparams, "randomize_x_conditioning", False),
            randomize_x1=getattr(self.hparams, "randomize_x1", False),
            random_seed=getattr(self.hparams, "random_seed", None),
            randomization_mode=getattr(self.hparams, "randomization_mode", "dynamic"),
            norm_profile=getattr(self.hparams, "norm_profile", "fixed32"),
            clip_profile=getattr(self.hparams, "clip_profile", "none"),
        )

        self.test_data = FIRE(
            root=self.hparams.root,
            fire_run=self.hparams.fire_run,
            dims=self.hparams.dims,
            transform=None,
            idx_list=test_idx,
            resolution=self.hparams.resolution,
            redshift=self.hparams.redshift,
            x1_fields=self.hparams.x1_fields,
            x_conditioning_fields=self.hparams.x_conditioning_fields,
            file_suffix=file_suffix,
            randomize_theta=getattr(self.hparams, "randomize_theta", False),
            randomize_x_conditioning=getattr(self.hparams, "randomize_x_conditioning", False),
            randomize_x1=getattr(self.hparams, "randomize_x1", False),
            random_seed=getattr(self.hparams, "random_seed", None),
            randomization_mode=getattr(self.hparams, "randomization_mode", "dynamic"),
            norm_profile=getattr(self.hparams, "norm_profile", "fixed32"),
            clip_profile=getattr(self.hparams, "clip_profile", "none"),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )