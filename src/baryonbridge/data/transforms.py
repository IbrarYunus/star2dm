import torch
import numpy as np
import torchvision.transforms.functional as F


class Translate(object):
    """Translation transform that works with both 2D and 3D data.
    
    Args:
        ndim (int, optional): Number of spatial dimensions (2 for 2D images, 3 for 3D volumes).
                             If None, it will be inferred from the input data.
    """
    def __init__(self, ndim=None):
        self.ndim = ndim
        
    def __call__(self, sample):
        in_img, tgt_img = sample  # (C, D, H, W) for 3D, (C, H, W) for 2D, etc.
        
        # Infer ndim if not provided
        ndim = self.ndim if self.ndim is not None else len(in_img.shape) - 1
        
        # Generate random shifts for each spatial dimension
        shifts = [torch.randint(in_img.shape[-ndim + i], (1,)).item() 
                 for i in range(ndim)]
        
        # Roll along all spatial dimensions, starting from the last
        dims = tuple(range(-ndim, 0))
        in_img = torch.roll(in_img, shifts, dims=dims)
        tgt_img = torch.roll(tgt_img, shifts, dims=dims)
        
        return in_img, tgt_img


class Flip(object):
    """Random flip transform that works with both 2D and 3D data.
    
    Args:
        ndim (int, optional): Number of spatial dimensions (2 for 2D images, 3 for 3D volumes).
                             If None, it will be inferred from the input data.
    """
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.axes = None
        
    def __call__(self, sample):
        in_img, tgt_img = sample
        
        # Infer ndim if not provided
        ndim = self.ndim if self.ndim is not None else len(in_img.shape) - 1
        
        assert ndim > 1, "flipping is ambiguous for 1D scalars/vectors"
        
        # Randomly choose which axes to flip
        self.axes = torch.randint(2, (ndim,), dtype=torch.bool)
        self.axes = torch.arange(ndim)[self.axes]
        
        # Handle vector fields (if the number of channels equals spatial dimensions)
        if in_img.shape[0] == ndim:  # flip vector components
            in_img = in_img.clone()  # Create a copy to avoid modifying the original
            in_img[self.axes] = -in_img[self.axes]
        
        # Flip the spatial dimensions (adding 1 to account for the channel dimension)
        shifted_axes = (1 + self.axes).tolist()
        if shifted_axes:  # Only flip if we have axes to flip
            in_img = torch.flip(in_img, shifted_axes)
        
        # Handle target image similarly
        if tgt_img.shape[0] == ndim:  # flip vector components
            tgt_img = tgt_img.clone()  # Create a copy to avoid modifying the original
            tgt_img[self.axes] = -tgt_img[self.axes]
        
        shifted_axes = (1 + self.axes).tolist()
        if shifted_axes:  # Only flip if we have axes to flip
            tgt_img = torch.flip(tgt_img, shifted_axes)
        
        return in_img, tgt_img


class Permutate(object):
    """Random permutation transform that works with both 2D and 3D data.
    
    Args:
        ndim (int, optional): Number of spatial dimensions (2 for 2D images, 3 for 3D volumes).
                             If None, it will be inferred from the input data.
    """
    def __init__(self, ndim=None):
        self.ndim = ndim
        self.axes = None
        
    def __call__(self, sample):
        in_img, tgt_img = sample
        
        # Infer ndim if not provided
        ndim = self.ndim if self.ndim is not None else len(in_img.shape) - 1
        
        assert ndim > 1, "permutation is not necessary for 1D fields"
        
        # Generate random permutation of spatial dimensions
        self.axes = torch.randperm(ndim)
        
        # Handle vector fields (if the number of channels equals spatial dimensions)
        if in_img.shape[0] == ndim:  # permutate vector components
            in_img = in_img[self.axes]
        
        # Create the permutation including the channel dimension
        shifted_axes = [0] + (1 + self.axes).tolist()
        in_img = in_img.permute(shifted_axes)
        
        # Handle target image similarly
        if tgt_img.shape[0] == ndim:  # permutate vector components
            tgt_img = tgt_img[self.axes]
        
        tgt_img = tgt_img.permute(shifted_axes)
        
        return in_img, tgt_img


class Resize(object):
    """Resize transform that works with both 2D and 3D data.
    
    Args:
        size (tuple): Target size for each spatial dimension
        ndim (int, optional): Number of spatial dimensions (2 for 2D images, 3 for 3D volumes).
                             If None, it will be inferred from the input data.
        mode (str): Interpolation mode ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', etc.)
    """
    def __init__(self, size, ndim=None, mode='nearest'):
        self.size = size if isinstance(size, (list, tuple)) else (size, size, size)
        self.ndim = ndim
        self.mode = mode
        
    def __call__(self, sample):
        in_img, tgt_img = sample
        
        # Infer ndim if not provided
        ndim = self.ndim if self.ndim is not None else len(in_img.shape) - 1
        
        # Use appropriate size based on dimensions
        target_size = self.size[:ndim]
        
        # Resize input image
        if ndim == 2:
            in_img = F.resize(in_img, target_size, interpolation=F.InterpolationMode.NEAREST 
                             if self.mode == 'nearest' else F.InterpolationMode.BILINEAR)
            tgt_img = F.resize(tgt_img, target_size, interpolation=F.InterpolationMode.NEAREST 
                              if self.mode == 'nearest' else F.InterpolationMode.BILINEAR)
        else:  # 3D case
            # For 3D, we need to use a different approach as torchvision.transforms doesn't support 3D
            # Use the same shape as input with different sizes
            in_shape = list(in_img.shape)
            tgt_shape = list(tgt_img.shape)
            
            # Update spatial dimensions with target size
            for i in range(ndim):
                in_shape[i+1] = target_size[i]
                tgt_shape[i+1] = target_size[i]
            
            # Use torch's interpolate function which works with any number of dimensions
            in_img = torch.nn.functional.interpolate(
                in_img.unsqueeze(0), 
                size=in_shape[1:], 
                mode='nearest' if self.mode == 'nearest' else 'trilinear'
            ).squeeze(0)
            
            tgt_img = torch.nn.functional.interpolate(
                tgt_img.unsqueeze(0), 
                size=tgt_shape[1:], 
                mode='nearest' if self.mode == 'nearest' else 'trilinear'
            ).squeeze(0)
            
        return in_img, tgt_img


class RandomCrop(object):
    """Random crop transform that works with both 2D and 3D data.
    
    Args:
        size (tuple): Crop size for each spatial dimension
        ndim (int, optional): Number of spatial dimensions (2 for 2D images, 3 for 3D volumes).
                             If None, it will be inferred from the input data.
    """
    def __init__(self, size, ndim=None):
        self.size = size if isinstance(size, (list, tuple)) else (size, size, size)
        self.ndim = ndim
        
    def __call__(self, sample):
        in_img, tgt_img = sample
        
        # Infer ndim if not provided
        ndim = self.ndim if self.ndim is not None else len(in_img.shape) - 1
        
        # Use appropriate size based on dimensions
        crop_size = self.size[:ndim]
        
        # Calculate valid start positions
        starts = []
        for i in range(ndim):
            dim_size = in_img.shape[i+1]
            if dim_size <= crop_size[i]:
                starts.append(0)  # Can't crop if dimension is smaller than crop size
            else:
                starts.append(torch.randint(0, dim_size - crop_size[i] + 1, (1,)).item())
        
        # Perform cropping
        if ndim == 2:
            in_img = in_img[:, starts[0]:starts[0]+crop_size[0], starts[1]:starts[1]+crop_size[1]]
            tgt_img = tgt_img[:, starts[0]:starts[0]+crop_size[0], starts[1]:starts[1]+crop_size[1]]
        elif ndim == 3:
            in_img = in_img[:, starts[0]:starts[0]+crop_size[0], 
                        starts[1]:starts[1]+crop_size[1], 
                        starts[2]:starts[2]+crop_size[2]]
            tgt_img = tgt_img[:, starts[0]:starts[0]+crop_size[0], 
                          starts[1]:starts[1]+crop_size[1], 
                          starts[2]:starts[2]+crop_size[2]]
        
        return in_img, tgt_img


class Compose(object):
    """Composes several transforms together.
    
    Args:
        transforms (list): List of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
