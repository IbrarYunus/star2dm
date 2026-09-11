"""
BaryonBridge package for analyzing cosmological N-body simulations.
"""

from .config import (
    CLIP_PROFILE,
    CLIP_PROFILES,
    GRID_SUFFIX,
    NORM_PROFILE,
    NORM_PROFILES,
    RESOLUTION,
    VERSIONS,
    get_clip_profile,
    get_grid_suffix,
    get_norm_profile,
    get_resolution,
    get_version_info,
    merged_norm_dict,
    norm_profile_table
)

__version__ = "0.0.0"
__all__ = [
    'CLIP_PROFILE',
    'CLIP_PROFILES',
    'GRID_SUFFIX',
    'NORM_PROFILE',
    'NORM_PROFILES',
    'RESOLUTION',
    'VERSIONS',
    'get_clip_profile',
    'get_grid_suffix',
    'get_norm_profile',
    'get_resolution',
    'get_version_info',
    'merged_norm_dict',
    'norm_profile_table'
]