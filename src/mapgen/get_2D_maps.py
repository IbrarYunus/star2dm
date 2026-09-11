# get the 2D maps of the DM distribution for the training set
import numpy as np
import matplotlib.pyplot as plt
# import the helper functions
from util_func import *
# allow not using agama when other scripts using functions from this file
try:
    import agama
except ImportError:
    pass
from astropy.stats import sigma_clip
import argparse
import logging
from datetime import datetime
import os
import csv

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

# import the kdtree smoothing function
from kdtree_smoothing import *
import dreams_python

# Define the data path
main_path = '/home/trg9de/'
sim_path = '/standard/torrey-group/'
DREAMS_dm_type = 'CDM'
DREAMS_mass = 'MW_zooms'
DREAMS_dataset = 'SB5'
DREAMS_param_file = 'CDM_TNG_MW_SB5.txt'
snap_path = os.path.join(sim_path, f'DREAMS/Sims/{DREAMS_dm_type}/{DREAMS_mass}/{DREAMS_dataset}/')
group_path = os.path.join(sim_path, f'DREAMS/FOF_Subfind/{DREAMS_dm_type}/{DREAMS_mass}/{DREAMS_dataset}/')
param_path = os.path.join(sim_path, f'DREAMS/Parameters/{DREAMS_dm_type}/{DREAMS_mass}/{DREAMS_param_file}')
potential_path = os.path.join(main_path, f'ML_dm_distribution/potential_agama/')
potential_path_none_sym = os.path.join(main_path, f'ML_dm_distribution/potential_agama_none_sym/')
mvs = dreams_python.DREAMS(sim_path + '/DREAMS', suite='MW_zooms', DM_type='CDM', sobol_number=5)

# Define the data path for TNG
tng_snap_path = os.path.join(main_path, f'TNG_data/Sims/')

# Define the data path for FIRE
fire_snap_path = os.path.join(sim_path, f'FIRE/core')

# Define the hyper-parameters
# box = 1       #which simulation is used [0,1023]; will loop over all boxes
# snap = 90     #which snapshot 90 -> z=0; will work up to z~1 as written
# a = 1.0       #scale factor
# part_type = [1,4] #which particle type to load
# 0 - gas
# 1 - high res dark matter
# 2 - low res dark matter
# 3 - tracers (not used in DREAMS)
# 4 - stars
# 5 - black holes
# h = .6909     #reduced hubble constant (do not change) # defined in util func

# define the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a formatter that includes timestamp, level, and module name
formatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# create console handler and set formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# define the funcitons for getting the data
def get_snap_from_redshift(snapshot_time, redshift):
    # get the snap number from the redshift
    idx = np.argmin(np.abs(snapshot_time['z'] - redshift))
    snap = snapshot_time['snap'][idx]
    return snap

def get_redshift_from_snap(snapshot_time, snap):
    # get the redshift from the snap number
    idx = np.argmin(np.abs(snapshot_time['snap'] - snap))
    redshift = snapshot_time['z'][idx]
    return redshift

def get_a_from_snap_or_redshift(snapshot_time, snap=None, redshift=None):
    if snap is not None:
        idx = np.argmin(np.abs(snapshot_time['snap'] - snap))
    elif redshift is not None:
        idx = np.argmin(np.abs(snapshot_time['z'] - redshift))
    else:
        raise ValueError('Either snap or redshift must be provided')
    a = snapshot_time['a'][idx]
    return a


def get_data(box, snap, part_type):
    """Load DREAMS particle data. Returns (part_cat, h)."""
    path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
    part_cat, h = load_particle_data(path, ['Masses', 'Coordinates', 'Velocities', 'Potential', 'SubfindDMDensity', 'SubfindDensity'], part_type)
    return part_cat, h


def get_data_TNG(box, snap, part_type, sim_name):
    """Load TNG particle data from pre-extracted halo files. Returns (prt_cat, h, gal_pos, vels_cen, rotation_tensor)."""
    if sim_name == 'TNG50':
        path = f'{tng_snap_path}/TNG50/tng50_snap_{snap}_halo_{box}.h5'
    elif sim_name == 'TNG100':
        path = f'{tng_snap_path}/TNG100/tng100_snap_{snap}_halo_{box}.h5'
    else:
        raise ValueError(f"Unknown TNG simulation: {sim_name}")
    prt_cat, h, gal_pos, vels_cen, rotation_tensor = load_particle_data_TNG(
        path,
        ['Masses', 'Coordinates', 'Velocities', 'Potential', 'SubfindDMDensity', 'SubfindDensity'],
        part_type,
        sim_name,
    )
    return prt_cat, h, gal_pos, vels_cen, rotation_tensor

def get_MW_halo(box, snap, part_cat, h):
    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
    grp_cat = load_group_data(path, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr'])
    
    # get the MW index
    mw_idx = get_MW_idx(grp_cat, h) 

    # get the MW halo data
    prt_cat, fof_cat = get_galaxy_data(part_cat, grp_cat, mw_idx)
    return prt_cat, fof_cat

def get_MW_halo_fof(box, snap, h, pt_types):
    path = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
    grp_cat = load_group_data(path, ['GroupLenType', 'GroupFirstSub', 'GroupNsubs', 'GroupMassType', 'GroupPos', 'SubhaloLenType', 'SubhaloGrNr'])
    
    # get the MW index
    mw_idx = get_MW_idx(grp_cat, h) 

    # get the MW halo data; using the DREAMS_python load_single_halo function
    prt_cat = mvs.load_single_halo(box, snap, fof_idx=mw_idx, part_types=pt_types,
    keys=['ParticleIDs','Masses', 'Coordinates', 'Velocities', 'Potential', 'SubfindDMDensity', 'SubfindDensity'],
    _nsubs_limiter=99999)
    return prt_cat

def get_halo_coords(box, snap, rmax_ctr=10):
    """
    Read the coordinate transformation information saved by get_apot.py
    
    Parameters:
    -----------
    box : str
        Box identifier
    nsnap : int
        Snapshot number
    rmax_ctr : float, optional
        Maximum radius in kpc, defaults to 10
        
    Returns:
    --------
    gal_pos : ndarray
        Center position in kpc physical
    vels_cen : ndarray
        Center velocity in km/s physical
    rotation_tensor : ndarray
        Rotation matrix to principal-axis frame
    axis_ratios : ndarray
        Axis ratios of the system
    """
    
    # Construct the same file path as in get_apot.py
    output_stem = os.path.join(potential_path, f'{rmax_ctr:.0f}kpc', f'{box}_{snap}')
    cname = '{0}_coords.txt'.format(output_stem)
    
    # Read the file and extract information
    data = []
    with open(cname, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            data.extend([float(x) for x in line.strip().split()])
    
    # Convert to numpy arrays and reshape
    data = np.array(data)
    
    # Extract individual components
    gal_pos = data[0:3]  # 3 components for position
    vels_cen = data[3:6]  # 3 components for velocity
    rotation_tensor = data[6:15].reshape(3,3)  # 3x3 rotation matrix
    axis_ratios = data[15:18]  # 3 components for axis ratios
    
    return gal_pos, vels_cen, rotation_tensor, axis_ratios

def center_and_rescale(prt_cat, gal_pos, vels_cen, rotation_tensor, part_type, a, h):
    masses = {}
    for type in part_type:
        masses[type] = prt_cat[f'PartType{type}/Masses'] * 1e10 / h

    potentials = {}
    for type in part_type:
        try:
            potentials[type] = prt_cat[f'PartType{type}/Potential'] / a
        except KeyError:
            potentials[type] = None
            logger.warning(f"Potential not loaded for particle type {type}. Skipping.")
            continue

    densities = {}
    for type in part_type:
        try:
            densities[type] = prt_cat[f'PartType{type}/SubfindDensity'] * (1e10 / h) / (1/h)**3
        except KeyError:
            densities[type] = None
            logger.warning(f"Density not loaded for particle type {type}. Skipping.")
            continue
    
    densitiesDM = {}
    for type in part_type:
        try:
            densitiesDM[type] = prt_cat[f'PartType{type}/SubfindDMDensity'] * (1e10 / h) / (1/h)**3
        except KeyError:
            densitiesDM[type] = None
            logger.warning(f"DM density not loaded for particle type {type}. Skipping.")
            continue

    coords = {}
    for type in part_type:
        coords[type] = prt_cat[f'PartType{type}/Coordinates'] / h
        coords[type] -= gal_pos # center the galaxy 

    vels = {}
    for type in part_type:
        vels_tmp = prt_cat[f'PartType{type}/Velocities'] * np.sqrt(a) # convert to physical units
        vels[type] = vels_tmp - vels_cen

    # rotate the coordinates and velocities
    coords_rot = {}
    vels_rot = {}
    for type in part_type:
        coords_rot[type] = get_coordinates_rotated(coords[type], rotation_tensor=rotation_tensor)
        vels_rot[type] = get_coordinates_rotated(vels[type], rotation_tensor=rotation_tensor)

    return masses, potentials, densities, densitiesDM, coords_rot, vels_rot


def calculate_2d_average(coords, values, resolution, xmin, xmax, ymin, ymax, return_counts=False, fill_value=np.nan):
    """
    Calculate the average of values in 2D spatial bins.
    
    Parameters:
    -----------
    coords : array_like
        2D array of particle coordinates [N, 2] (x, y coordinates)
    values : array_like
        1D array of values to average [N]
    resolution : int
        Number of bins in each dimension
    xmin, xmax, ymin, ymax : float
        Spatial range for binning
    return_counts : bool, optional
        If True, also return the counts array
    fill_value : float, optional
        Value to fill in for bins with no particles
        
    Returns:
    --------
    maps : ndarray
        2D array of averaged values [resolution, resolution]
    counts : ndarray, optional
        2D array of particle counts per bin [resolution, resolution] (only if return_counts=True)
    """
    # Compute bin assignments for each particle
    x_bins = np.linspace(xmin, xmax, resolution+1)
    y_bins = np.linspace(ymin, ymax, resolution+1)
    
    x_indices = np.digitize(coords[:, 0], x_bins) - 1
    y_indices = np.digitize(coords[:, 1], y_bins) - 1
    
    # Filter out particles outside the bins
    valid_particles = (x_indices >= 0) & (x_indices < resolution) & (y_indices >= 0) & (y_indices < resolution)
    x_indices = x_indices[valid_particles]
    y_indices = y_indices[valid_particles]
    
    # Calculate mean values for each bin
    maps = np.zeros((resolution, resolution))
    counts = np.zeros((resolution, resolution))
    
    np.add.at(maps, (x_indices, y_indices), values[valid_particles])
    np.add.at(counts, (x_indices, y_indices), 1)
    maps = np.divide(maps, counts, where=counts > 0)
    maps[counts == 0] = fill_value
    
    if return_counts:
        return maps, counts
    else:
        return maps


def calculate_3d_average(coords, values, resolution, xmin, xmax, ymin, ymax, zmin, zmax, return_counts=False, fill_value=np.nan):
    """
    Calculate the average of values in 3D spatial bins.
    
    Parameters:
    -----------
    coords : array_like
        3D array of particle coordinates [N, 3] (x, y, z coordinates)
    values : array_like
        1D array of values to average [N]
    resolution : int
        Number of bins in each dimension
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Spatial range for binning
    return_counts : bool, optional
        If True, also return the counts array
    fill_value : float, optional
        Value to fill in for bins with no particles
    Returns:
    --------
    grids : ndarray
        3D array of averaged values [resolution, resolution, resolution]
    counts : ndarray, optional
        3D array of particle counts per bin [resolution, resolution, resolution] (only if return_counts=True)
    """
    # Compute bin assignments for each particle
    x_bins = np.linspace(xmin, xmax, resolution+1)
    y_bins = np.linspace(ymin, ymax, resolution+1)
    z_bins = np.linspace(zmin, zmax, resolution+1)
    
    x_indices = np.digitize(coords[:, 0], x_bins) - 1
    y_indices = np.digitize(coords[:, 1], y_bins) - 1
    z_indices = np.digitize(coords[:, 2], z_bins) - 1
    
    # Filter out particles outside the bins
    valid_particles = (x_indices >= 0) & (x_indices < resolution) & (y_indices >= 0) & (y_indices < resolution) & (z_indices >= 0) & (z_indices < resolution)
    x_indices = x_indices[valid_particles]
    y_indices = y_indices[valid_particles]
    z_indices = z_indices[valid_particles]
    
    # Calculate mean values for each bin
    grids = np.zeros((resolution, resolution, resolution))
    counts = np.zeros((resolution, resolution, resolution))
    
    np.add.at(grids, (x_indices, y_indices, z_indices), values[valid_particles])
    np.add.at(counts, (x_indices, y_indices, z_indices), 1)
    grids = np.divide(grids, counts, where=counts > 0)
    grids[counts == 0] = fill_value
    
    if return_counts:
        return grids, counts
    else:
        return grids


def get_2D_maps(coords_rot, vels_rot, masses, potentials, densitiesDM, densities, map_type, resolution, xmin, xmax, ymin, ymax, zmin, zmax, zmin_2d, zmax_2d, rmax_ctr, box, snap, rescale=False):
    # check if the map type is valid
    if map_type not in ['Mstar', 'MDM', 'Mtot', 'Ptot', 'Pfluc', 'Pagama', 'PagamaNone', 'PflucNone',
    'Uavg', 'Vavg', 'Wavg', 'vRavg', 'vTavg', 'rhoDM', 'rhostar', 'rhotot']:
        raise ValueError(f'Invalid map type: {map_type}')
    
    # Filter particles based on 2D z-range for all particle types
    coords_rot_2d = {}
    vels_rot_2d = {}
    masses_2d = {}
    potentials_2d = {}
    densitiesDM_2d = {}
    densities_2d = {}

    for type in coords_rot.keys():
        # Filter particles within the 2D z-range
        z_mask = (coords_rot[type][:, 2] >= zmin_2d) & (coords_rot[type][:, 2] <= zmax_2d)
        coords_rot_2d[type] = coords_rot[type][z_mask]
        vels_rot_2d[type] = vels_rot[type][z_mask]
        masses_2d[type] = masses[type][z_mask]
        if type in potentials and potentials[type] is not None:
            potentials_2d[type] = potentials[type][z_mask]
        else:
            potentials_2d[type] = None
            logger.warning(f"Potential not loaded for particle type {type}. Skipping.")
            continue
        if type in densitiesDM and densitiesDM[type] is not None:
            densitiesDM_2d[type] = densitiesDM[type][z_mask]
        else:
            densitiesDM_2d[type] = None
            logger.warning(f"DM density not loaded for particle type {type}. Skipping.")
            continue
        if type in densities and densities[type] is not None:
            densities_2d[type] = densities[type][z_mask]
        else:
            densities_2d[type] = None
            logger.warning(f"Density not loaded for particle type {type}. Skipping.")
            continue
    # get the 2D maps
    maps = np.zeros((resolution, resolution))

    # calculate the grid points for the get_map function and for the agama potential
    # Create coordinate grids using meshgrid
    # Use bin centers instead of bin edges for correct array size
    x_bins = np.linspace(xmin, xmax, resolution+1)
    y_bins = np.linspace(ymin, ymax, resolution+1)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    x_coords, y_coords = np.meshgrid(x_centers, y_centers, indexing='ij')
    z_coords = np.zeros_like(x_coords)
    
    # Reshape to get all coordinates as a single array of points
    points = np.column_stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()])
    
    # get the stellar density map if the map type is in anyway related to the stellar density
    if map_type in ['rhostar', 'Uavg', 'Vavg', 'Wavg', 'vRavg', 'vTavg']:
        rhostar_maps = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
        masses[4], points[:,0], points[:,1], points[:,2], resolution, two_d=True)
        # reshape the map to 2D
        rhostar_maps = rhostar_maps.reshape(resolution, resolution)
    
    if map_type == 'Mstar':
        for type in [4]:
            # calculate the mass in each bin
            maps += np.histogram2d(coords_rot_2d[type][:,0], coords_rot_2d[type][:,1], bins=resolution, range=[(xmin, xmax), (ymin, ymax)], weights=masses_2d[type])[0]
            # take the log of the map
            # set the minimum value to 1 to avoid log(0); 
            # this minimum value can't be too small, otherwise the sum underflow when stored with the other huge values in the maps array
        maps = np.log10(maps + 1)
    elif map_type == 'rhostar':
        # use the get_map function from Alex G. to calculate the density map
        # notice that in this case, we don't use the 2D z-range filter for the most ideal case
        # observing constraint will be added in the future
        # this should also have minimal impact on the z=0 locations since the tree is unlikely to go 
        # significant beyond the z=0 locations
        maps = rhostar_maps
        # take the log of the map
        maps = np.log10(maps + 1)
    elif map_type == 'MDM':
        for type in [1]:
            # calculate the mass in each bin
            maps += np.histogram2d(coords_rot_2d[type][:,0], coords_rot_2d[type][:,1], bins=resolution, range=[(xmin, xmax), (ymin, ymax)], weights=masses_2d[type])[0]
            # take the log of the map
            # set the minimum value to 1 to avoid log(0); 
            # this minimum value can't be too small, otherwise the sum underflow when stored with the other huge values in the maps
        maps = np.log10(maps + 1)
    elif map_type == 'Mtot':
        # sum up the masses of all the particle types
        for type in [0,1,4,5]:
            hist_tmp = np.histogram2d(coords_rot_2d[type][:,0], coords_rot_2d[type][:,1], bins=resolution, range=[(xmin, xmax), (ymin, ymax)], weights=masses_2d[type])[0]
            maps += hist_tmp
            # take the log of the map
            # set the minimum value to 1 to avoid log(0)
        maps = np.log10(maps + 1)
    elif map_type == 'Ptot' or map_type == 'Pfluc' or map_type == 'Pagama' or map_type == 'PflucNone' or map_type == 'PagamaNone':
        # Calculate mean potential for each bin; use the stellar particle potential
        maps, counts = calculate_2d_average(coords_rot_2d[4], potentials_2d[4], resolution, xmin, xmax, ymin, ymax, return_counts=True, fill_value=np.nan)
        # following is not ideal since it only works if the empty bins are isolated
        # take the average of the four nearest neighbors for the bins with no particles
        # roll the map by 1 pixel to the left and right, and up and down    
        # maps_rolled1 = np.roll(maps, 1, axis=0)
        # maps_rolled2 = np.roll(maps, -1, axis=0)
        # maps_rolled3 = np.roll(maps, 1, axis=1)
        # maps_rolled4 = np.roll(maps, -1, axis=1)
        # maps[counts == 0] = np.nanmean([maps_rolled1, maps_rolled2, maps_rolled3, maps_rolled4], axis=0)[counts == 0]
        
        if map_type == 'Pfluc' or map_type == 'Pagama' or map_type == 'PflucNone' or map_type == 'PagamaNone':
            # set up the agama potential
            if map_type.endswith('None'):
                apot1 = agama.Potential(potential_path_none_sym + f'{rmax_ctr}kpc/{box}_{snap}.bar.none.coef_cylsp')
                apot2 = agama.Potential(potential_path_none_sym + f'{rmax_ctr}kpc/{box}_{snap}.dark.none.coef_mul')
            else:
                apot1 = agama.Potential(potential_path + f'{rmax_ctr}kpc/{box}_{snap}.bar.axi.coef_cylsp')
                apot2 = agama.Potential(potential_path + f'{rmax_ctr}kpc/{box}_{snap}.dark.axi.coef_mul')
            apot = agama.Potential(apot1,apot2)
            
            if map_type == 'Pfluc' or map_type == 'PflucNone':
                # calculate the potential for each star particle
                pot_vals = apot.potential(coords_rot_2d[1])
                
                # Calculate mean agama potential for each bin
                agama_maps = calculate_2d_average(coords_rot_2d[4], pot_vals, resolution, xmin, xmax, ymin, ymax, fill_value=np.nan)
                # following is not ideal since it only works if the empty bins are isolated
                # take the average of the four nearest neighbors for the bins with no particles
                # roll the map by 1 pixel to the left and right, and up and down 
                # agama_maps_rolled1 = np.roll(agama_maps, 1, axis=0)
                # agama_maps_rolled2 = np.roll(agama_maps, -1, axis=0)
                # agama_maps_rolled3 = np.roll(agama_maps, 1, axis=1)
                # agama_maps_rolled4 = np.roll(agama_maps, -1, axis=1)
                # agama_maps[counts == 0] = np.nanmean([agama_maps_rolled1, agama_maps_rolled2, agama_maps_rolled3, agama_maps_rolled4], axis=0)[counts == 0]
                maps -= agama_maps
            elif map_type == 'Pagama' or map_type == 'PagamaNone':
                # Calculate potential for all points at once
                potentials_flat = apot.potential(points)
                
                # Reshape back to 2D grid
                agama_maps = potentials_flat.reshape(resolution, resolution)
                maps = agama_maps
                
    elif map_type == 'Uavg' or map_type == 'Vavg' or map_type == 'Wavg':
        # calculate the mass weighted average stellar velocity in each bin; essentially the same as a straight average because the masses are the same for all the particles
        if map_type == 'Uavg':
            # maps = calculate_2d_average(coords_rot_2d[4], vels_rot_2d[4][:, 0], resolution, xmin, xmax, ymin, ymax, fill_value=np.nan)
            maps = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vels_rot[4][:, 0], points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 2D
            maps = maps.reshape(resolution, resolution)
            maps = maps / rhostar_maps
        elif map_type == 'Vavg':
            # maps = calculate_2d_average(coords_rot_2d[4], vels_rot_2d[4][:, 1], resolution, xmin, xmax, ymin, ymax, fill_value=np.nan)
            maps = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vels_rot[4][:, 1], points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 2D
            maps = maps.reshape(resolution, resolution)
            maps = maps / rhostar_maps
        elif map_type == 'Wavg':
            # maps = calculate_2d_average(coords_rot_2d[4], vels_rot_2d[4][:, 2], resolution, xmin, xmax, ymin, ymax, fill_value=np.nan)
            maps = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vels_rot[4][:, 2], points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 2D
            maps = maps.reshape(resolution, resolution)
            maps = maps / rhostar_maps
    elif map_type == 'vRavg' or map_type == 'vTavg':
        # calculate the mass weighted average stellar velocity in each bin
        # calculate the velocities in the cylindrical coordinate system
        theta_tmp = np.arctan2(coords_rot[4][:,1], coords_rot[4][:,0])
        vR_tmp = vels_rot[4][:,0] * np.cos(theta_tmp) + vels_rot[4][:,1] * np.sin(theta_tmp)
        vT_tmp = -vels_rot[4][:,0] * np.sin(theta_tmp) + vels_rot[4][:,1] * np.cos(theta_tmp)
        
        if map_type == 'vRavg':
            # maps = calculate_2d_average(coords_rot_2d[4], vR_tmp, resolution, xmin, xmax, ymin, ymax, fill_value=np.nan)
            maps = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vR_tmp, points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 2D
            maps = maps.reshape(resolution, resolution)
            maps = maps / rhostar_maps
        elif map_type == 'vTavg':
            # maps = calculate_2d_average(coords_rot_2d[4], vT_tmp, resolution, xmin, xmax, ymin, ymax, fill_value=np.nan)
            maps = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vT_tmp, points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 2D
            maps = maps.reshape(resolution, resolution)
            maps = maps / rhostar_maps
    elif map_type == 'rhoDM' or map_type == 'rhotot':
        # calculate the average dark matter density in each bin
        if map_type == 'rhoDM':
            # maps = calculate_2d_average(coords_rot_2d[4], densitiesDM_2d[4], resolution, xmin, xmax, ymin, ymax, fill_value=0)
            maps = get_map(coords_rot[1][:,0], coords_rot[1][:,1], coords_rot[1][:,2], 
            masses[1], points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 2D
            maps = maps.reshape(resolution, resolution)
        elif map_type == 'rhotot':
            maps = calculate_2d_average(coords_rot_2d[4], densities_2d[4], resolution, xmin, xmax, ymin, ymax, fill_value=0)
        
        # take the log of the density map
        # set the minimum value to 1 to avoid log(0)
        maps = np.log10(maps + 1)
    
    # for debugging; check if there are any NaNs in the maps
    # if np.any(np.isnan(maps)):
    #     logger.warning('There are NaNs in the maps. Current map:')
    #     logger.warning(map_type)
    # if np.any(np.isinf(maps)):
    #     logger.warning('There are Inf in the maps. Current map:')
    #     logger.warning(map_type)

    if rescale:
        # do a sigma clipping to remove the outliers or the values that are too small
        if map_type == 'Pfluc' or map_type == 'PflucNone':
            clipped = sigma_clip(maps, sigma=3)
            valid_data = ~clipped.mask
        elif map_type == 'Ptot' or map_type == 'Pagama' or map_type == 'PagamaNone':
            clipped = np.isnan(maps)
            valid_data = ~clipped
        elif map_type == 'Mstar' or map_type == 'MDM' or map_type == 'Mtot':
            clipped = maps <= 0
            valid_data = ~clipped
        elif map_type == 'Uavg' or map_type == 'Vavg' or map_type == 'Wavg':
            clipped = np.isnan(maps)
            valid_data = ~clipped
        elif map_type == 'vRavg' or map_type == 'vTavg':
            clipped = np.isnan(maps)
            valid_data = ~clipped
        elif map_type == 'rhoDM' or map_type == 'rhotot':
            clipped = np.where(maps<=0)
            valid_data = ~clipped
        # logger.info(f'Valid data points for {map_type} map for rescaling: {np.sum(valid_data)}')
        # maps[clipped.mask] = np.nan

        # Check if we have any valid data points
        if np.sum(valid_data) > 0:
            if map_type in ['rhotot','rhoDM']: 
                # only do mean shift for density profiles
                mean_val = np.nanmean(maps[valid_data])
                logger.warning(f'No std rescaling for {map_type} grid.')
                logger.warning(f'Mean value: {mean_val}')
                maps = maps - mean_val  # Just center the data 
            else:
                mean_val = np.nanmean(maps[valid_data])
                std_val = np.nanstd(maps[valid_data])
                # Avoid division by zero
                if std_val > 0:
                    maps = (maps - mean_val) / std_val
                else:
                    logger.warning(f'Standard deviation is zero for {map_type} grid. Skipping normalization.')
                    logger.warning(f'Mean value: {mean_val}')
                    logger.warning(f'Std value: {std_val}')
                    maps = maps - mean_val  # Just center the data if std is zero
            # assign zeros to the empty bins for Pfluc
            if map_type == 'Pfluc' or map_type == 'PflucNone':
                maps[counts == 0] = 0
            return maps, mean_val, std_val
        else:
            logger.warning(f'No valid data points found for {map_type} map after sigma clipping.')
            if rescale:
                return maps, 0.0, 1.0  # Return neutral values for mean and std
    else:
        return maps

def get_3D_grids(coords_rot, vels_rot, masses, potentials, densitiesDM, densities, map_type, resolution, xmin, xmax, ymin, ymax, zmin, zmax, rmax_ctr, box, snap, rescale=False):
    # check if the map type is valid
    if map_type not in ['Mstar', 'MDM', 'Mtot', 'Ptot', 'Pfluc', 'Pagama', 'PagamaNone', 
    'PflucNone', 'Uavg', 'Vavg', 'Wavg', 'vRavg', 'vTavg', 'rhoDM', 'rhotot', 'rhostar']:
        raise ValueError(f'Invalid map type: {map_type}')
    
    # get the 3D grids
    grids = np.zeros((resolution, resolution, resolution))

    # calculate the grid points for the get_map function and for the agama potential
    # Create coordinate grids using meshgrid
    # Use bin centers instead of bin edges for correct array size
    x_bins = np.linspace(xmin, xmax, resolution+1)
    y_bins = np.linspace(ymin, ymax, resolution+1)
    z_bins = np.linspace(zmin, zmax, resolution+1)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    x_coords, y_coords, z_coords = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    
    # Reshape to get all coordinates as a single array of points
    points = np.column_stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()])
    
    # get the stellar density map if the map type is in anyway related to the stellar density
    if map_type in ['rhostar', 'Uavg', 'Vavg', 'Wavg', 'vRavg', 'vTavg']:
        rhostar_grids = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
        masses[4], points[:,0], points[:,1], points[:,2], resolution, two_d=False)
        # reshape the map to 3D
        rhostar_grids = rhostar_grids.reshape(resolution, resolution, resolution)

    if map_type == 'Mstar':
        for type in [4]:
            grids += np.histogramdd(coords_rot[type], bins=resolution, range=[(xmin, xmax), (ymin, ymax), (zmin, zmax)], weights=masses[type])[0]
            # take the log of the map
            # set the minimum value to 1 to avoid log(0)
        grids = np.log10(grids + 1)
    elif map_type == 'rhostar':
        # use the get_map function from Alex G. to calculate the density map
        grids = rhostar_grids
        # take the log of the map
        # set the minimum value to 1 to avoid log(0)
        grids = np.log10(grids + 1)
    elif map_type == 'MDM':
        for type in [1]:
            grids += np.histogramdd(coords_rot[type], bins=resolution, range=[(xmin, xmax), (ymin, ymax), (zmin, zmax)], weights=masses[type])[0]
            # take the log of the map
            # set the minimum value to 1 to avoid log(0)
        grids = np.log10(grids + 1)
    elif map_type == 'Mtot':
        for type in [0,1,4,5]:
            grids += np.histogramdd(coords_rot[type], bins=resolution, range=[(xmin, xmax), (ymin, ymax), (zmin, zmax)], weights=masses[type])[0]
            # take the log of the map
            # set the minimum value to 1 to avoid log(0)
        grids = np.log10(grids + 1)
    elif map_type == 'Ptot' or map_type == 'Pfluc' or map_type == 'Pagama' or map_type == 'PflucNone' or map_type == 'PagamaNone':
        # Calculate mean potential for each bin; use the dark matter potential
        grids, counts = calculate_3d_average(coords_rot[4], potentials[4], resolution, xmin, xmax, ymin, ymax, zmin, zmax, return_counts=True)
        # following is not ideal since it only works if the empty bins are isolated
        # take the average of the six nearest neighbors for the bins with no particles
        # roll the grid by 1 pixel to the left and right, and up and down, and in and out
        # grids_rolled1 = np.roll(grids, 1, axis=0)
        # grids_rolled2 = np.roll(grids, -1, axis=0)
        # grids_rolled3 = np.roll(grids, 1, axis=1)
        # grids_rolled4 = np.roll(grids, -1, axis=1)
        # grids_rolled5 = np.roll(grids, 1, axis=2)
        # grids_rolled6 = np.roll(grids, -1, axis=2)
        # grids[counts == 0] = np.nanmean([grids_rolled1, grids_rolled2, grids_rolled3, grids_rolled4, grids_rolled5, grids_rolled6], axis=0)[counts == 0]
        
        if map_type == 'Pfluc' or map_type == 'Pagama' or map_type == 'PflucNone' or map_type == 'PagamaNone':
            # set up the agama potential
            if map_type.endswith('None'):
                apot1 = agama.Potential(potential_path_none_sym + f'{rmax_ctr}kpc/{box}_{snap}.bar.none.coef_cylsp')
                apot2 = agama.Potential(potential_path_none_sym + f'{rmax_ctr}kpc/{box}_{snap}.dark.none.coef_mul')
            else:
                apot1 = agama.Potential(potential_path + f'{rmax_ctr}kpc/{box}_{snap}.bar.axi.coef_cylsp')
                apot2 = agama.Potential(potential_path + f'{rmax_ctr}kpc/{box}_{snap}.dark.axi.coef_mul')
            apot = agama.Potential(apot1,apot2)
            
            if map_type == 'Pfluc' or map_type == 'PflucNone':
                # calculate the potential for each star particle
                pot_vals = apot.potential(coords_rot[1])

                # Calculate mean agama potential for each bin
                agama_grids = calculate_3d_average(coords_rot[4], pot_vals, resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=np.nan)
                # following is not ideal since it only works if the empty bins are isolated
                # take the average of the six nearest neighbors for the bins with no particles
                # roll the grid by 1 pixel to the left and right, and up and down, and in and out
                # agama_grids_rolled1 = np.roll(agama_grids, 1, axis=0)
                # agama_grids_rolled2 = np.roll(agama_grids, -1, axis=0)
                # agama_grids_rolled3 = np.roll(agama_grids, 1, axis=1)
                # agama_grids_rolled4 = np.roll(agama_grids, -1, axis=1)
                # agama_grids_rolled5 = np.roll(agama_grids, 1, axis=2)
                # agama_grids_rolled6 = np.roll(agama_grids, -1, axis=2)
                # agama_grids[counts == 0] = np.nanmean([agama_grids_rolled1, agama_grids_rolled2, agama_grids_rolled3, agama_grids_rolled4, agama_grids_rolled5, agama_grids_rolled6], axis=0)[counts == 0]

                grids -= agama_grids
            elif map_type == 'Pagama' or map_type == 'PagamaNone':
                # Calculate potential for all points at once
                potentials_flat = apot.potential(points)
                
                # Reshape back to 3D grid
                agama_grids = potentials_flat.reshape(resolution, resolution, resolution)
                grids = agama_grids


    elif map_type == 'Uavg' or map_type == 'Vavg' or map_type == 'Wavg':
        # calculate the mass weighted average stellar velocity in each bin; essentially the same as a straight average because the masses are the same for all the particles
        if map_type == 'Uavg':
            # grids = calculate_3d_average(coords_rot[4], vels_rot[4][:, 0], resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=np.nan)
            grids = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vels_rot[4][:, 0], points[:,0], points[:,1], points[:,2], resolution, two_d=False)
            # reshape the map to 3D
            grids = grids.reshape(resolution, resolution, resolution)
            grids = grids / rhostar_grids
        elif map_type == 'Vavg':
            # grids = calculate_3d_average(coords_rot[4], vels_rot[4][:, 1], resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=np.nan)
            grids = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vels_rot[4][:, 1], points[:,0], points[:,1], points[:,2], resolution, two_d=False)
            # reshape the map to 3D
            grids = grids.reshape(resolution, resolution, resolution)
            grids = grids / rhostar_grids
        elif map_type == 'Wavg':
            # grids = calculate_3d_average(coords_rot[4], vels_rot[4][:, 2], resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=np.nan)
            grids = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vels_rot[4][:, 2], points[:,0], points[:,1], points[:,2], resolution, two_d=False)
            # reshape the map to 3D
            grids = grids.reshape(resolution, resolution, resolution)
            grids = grids / rhostar_grids
    elif map_type == 'vRavg' or map_type == 'vTavg':
        # calculate the mass weighted average stellar velocity in each bin
        # calculate the velocities in the cylindrical coordinate system
        theta_tmp = np.arctan2(coords_rot[4][:,1], coords_rot[4][:,0])
        vR_tmp = vels_rot[4][:,0] * np.cos(theta_tmp) + vels_rot[4][:,1] * np.sin(theta_tmp)
        vT_tmp = -vels_rot[4][:,0] * np.sin(theta_tmp) + vels_rot[4][:,1] * np.cos(theta_tmp)
        if map_type == 'vRavg':
            # grids = calculate_3d_average(coords_rot[4], vR_tmp, resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=np.nan)
            grids = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vR_tmp, points[:,0], points[:,1], points[:,2], resolution, two_d=False)
            # reshape the map to 3D
            grids = grids.reshape(resolution, resolution, resolution)
            grids = grids / rhostar_grids
        elif map_type == 'vTavg':
            # grids = calculate_3d_average(coords_rot[4], vT_tmp, resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=np.nan)
            grids = get_map(coords_rot[4][:,0], coords_rot[4][:,1], coords_rot[4][:,2], 
            masses[4]*vT_tmp, points[:,0], points[:,1], points[:,2], resolution, two_d=False)
            # reshape the map to 3D
            grids = grids.reshape(resolution, resolution, resolution)
            grids = grids / rhostar_grids
    elif map_type == 'rhoDM' or map_type == 'rhotot':
        # calculate the average dark matter density in each bin
        if map_type == 'rhoDM':
            # grids = calculate_3d_average(coords_rot[4], densitiesDM[4], resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=0)
            grids = get_map(coords_rot[1][:,0], coords_rot[1][:,1], coords_rot[1][:,2], 
            masses[1], points[:,0], points[:,1], points[:,2], resolution, two_d=True)
            # reshape the map to 3D
            grids = grids.reshape(resolution, resolution, resolution)
        elif map_type == 'rhotot':
            grids = calculate_3d_average(coords_rot[4], densities[4], resolution, xmin, xmax, ymin, ymax, zmin, zmax, fill_value=0)
        # take the log of the density map
        # set the minimum value to 1 to avoid log(0)
        grids = np.log10(grids + 1)
        

    if rescale:
        # do a sigma clipping to remove the outliers or the values that are too small
        if map_type == 'Pfluc' or map_type == 'PflucNone':
            clipped = sigma_clip(grids, sigma=3) 
            valid_data = ~clipped.mask
        elif map_type == 'Ptot' or map_type == 'Pagama' or map_type == 'PagamaNone':
            clipped = np.isnan(grids)
            valid_data = ~clipped
        elif map_type == 'Mstar' or map_type == 'MDM' or map_type == 'Mtot':
            clipped = grids <= 0
            valid_data = ~clipped
        elif map_type == 'Uavg' or map_type == 'Vavg' or map_type == 'Wavg':
            clipped = np.isnan(grids)
            valid_data = ~clipped
        elif map_type == 'vRavg' or map_type == 'vTavg':
            clipped = np.isnan(grids)
            valid_data = ~clipped
        elif map_type == 'rhoDM' or map_type == 'rhotot':
            clipped = np.where(grids<=0)
            valid_data = ~clipped
        # logger.info(f'Valid data points for {map_type} grid: {np.sum(valid_data)}')
        # grids[clipped.mask] = np.nan
        
        # Check if we have any valid data points
        if np.sum(valid_data) > 0:
            if map_type in ['rhotot','rhoDM']: 
                # only do mean shift for density profiles
                mean_val = np.nanmean(grids[valid_data])
                logger.warning(f'No std rescaling for {map_type} grid.')
                logger.warning(f'Mean value: {mean_val}')
                grids = grids - mean_val  # Just center the data 
            else:
                mean_val = np.nanmean(grids[valid_data])
                std_val = np.nanstd(grids[valid_data])
                # Avoid division by zero
                if std_val > 0:
                    grids = (grids - mean_val) / std_val
                else:
                    logger.warning(f'Standard deviation is zero for {map_type} grid. Skipping normalization.')
                    logger.warning(f'Mean value: {mean_val}')
                    logger.warning(f'Std value: {std_val}')
                    grids = grids - mean_val  # Just center the data if std is zero
            # assign zeros to the empty bins for Pfluc
            if map_type == 'Pfluc' or map_type == 'PflucNone':
                grids[counts == 0] = 0
            return grids, mean_val, std_val
        else:
            logger.warning(f'No valid data points found for {map_type} grid after sigma clipping.')
            if rescale:
                return grids, 0.0, 1.0  # Return neutral values for mean and std
    else:
        return grids
    


            
def read_redshift_csv(csv_file):
    """
    Read all redshifts and physical FOVs from CSV file generated by sample_redshifts_spatial_scales.py
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file
    
    Returns
    -------
    redshifts : list
        List of redshifts (observed redshifts, not simulation redshifts)
    physical_fovs_kpc : list
        List of physical FOVs in kpc (one per row/box)
    physical_resolutions_kpc : list
        List of physical resolutions in kpc (one per row/box)
    """
    redshifts = []
    physical_fovs_kpc = []
    physical_resolutions_kpc = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            redshifts.append(float(row['redshift']))
            physical_fovs_kpc.append(float(row['physical_fov_kpc']))
            physical_resolutions_kpc.append(float(row['physical_resolution_kpc']))
    
    return redshifts, physical_fovs_kpc, physical_resolutions_kpc


if __name__ == '__main__':
    ## Example usage:
    # python get_2D_maps.py --box-ini 0 --box-end 1023 --redshift 0 --redshift-csv redshift_sample_0p05_0p65.csv --map-type rhoDM --resolution 128 --get-maps --save-map --sim-name DREAMS --snap 90
    # python get_2D_maps.py --box-ini 0 --box-end 1023 --redshift 0 --redshift-csv redshift_sample_0p65.csv --map-type rhoDM --resolution 128 --get-maps --save-map --sim-name TNG50 --snap 99
    parser = argparse.ArgumentParser()
    parser.add_argument('--box-ini', type=int, default=0)
    parser.add_argument('--box-end', type=int, default=1023)
    parser.add_argument('--box-list-file', type=str, default=None,
                        help='File containing custom box IDs (one per row). If provided, overrides --box-ini and --box-end')
    parser.add_argument('--sim-name', type=str, default='DREAMS', choices=['TNG50', 'TNG100', 'DREAMS', 'DREAMS_VM', 'FIRE'], help='Simulation name to use')
    parser.add_argument('--rescale', action='store_true')
    parser.add_argument('--resolution', type=int, default=128)
    # provide either snap or redshift
    parser.add_argument('--snap', type=int, default=None)
    parser.add_argument('--redshift', type=float, default=None)
    parser.add_argument('--redshift-csv', type=str, default=None, 
                        help='Path to CSV file with redshift and physical FOV data. Each row corresponds to a box (row 0 = box 0, etc.). The redshift in CSV is the observed redshift (determines FOV), not the simulation redshift.')
    parser.add_argument('--part-type', type=int, default=[0,1,4], nargs='+')
    parser.add_argument('--rmax-ctr', type=float, default=10)
    parser.add_argument('--map-type', type=str, nargs='+', default=['Mstar', 'Mtot', 'Ptot', 'Pfluc', 'Pagama'])
    parser.add_argument('--xmin', type=float, default=None, help='X minimum (overridden by CSV if provided)')
    parser.add_argument('--xmax', type=float, default=None, help='X maximum (overridden by CSV if provided)')
    parser.add_argument('--ymin', type=float, default=None, help='Y minimum (overridden by CSV if provided)')
    parser.add_argument('--ymax', type=float, default=None, help='Y maximum (overridden by CSV if provided)')
    parser.add_argument('--zmin', type=float, default=None, help='Z minimum (overridden by CSV if provided)')
    parser.add_argument('--zmax', type=float, default=None, help='Z maximum (overridden by CSV if provided)')
    parser.add_argument('--zmin-2d', type=float, default=None, help='Z-range for 2D map generation (controls which particles are included). If None and CSV provided, uses same as zmin.')
    parser.add_argument('--zmax-2d', type=float, default=None, help='Z-range for 2D map generation (controls which particles are included). If None and CSV provided, uses same as zmax.')
    parser.add_argument('--save-map', action='store_true')
    parser.add_argument('--get-maps', action='store_true', help='Generate 2D maps')
    parser.add_argument('--get-grids', action='store_true', help='Generate 3D grids')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output root directory. Keeps default simulation subdirectory layout under this path.')
    args = parser.parse_args()

    # Set DREAMS paths for varied_mass when using DREAMS_VM (same layout as synthesizer_get_images.py)
    if args.sim_name == 'DREAMS_VM':
        DREAMS_dm_type = 'CDM'
        DREAMS_mass = 'varied_mass'
        DREAMS_dataset = 'SB6'
        DREAMS_param_file = 'TNG_SB6.txt'
        snap_path = os.path.join(sim_path, f'DREAMS/Sims/{DREAMS_dm_type}/{DREAMS_mass}/{DREAMS_dataset}/')
        group_path = os.path.join(sim_path, f'DREAMS/FOF_Subfind/{DREAMS_dm_type}/{DREAMS_mass}/{DREAMS_dataset}/')
        param_path = os.path.join(sim_path, f'DREAMS/Parameters/{DREAMS_dm_type}/{DREAMS_mass}/{DREAMS_param_file}')
        mvs = dreams_python.DREAMS(sim_path + '/DREAMS', suite='varied_mass', DM_type='CDM', sobol_number=6)
        params_tmp, _ = mvs.read_param_file(DREAMS_param_file)
        target_masses = params_tmp[:, 0]
        logger.info(f"DREAMS_VM target masses (first 10): {target_masses[:10]}")

    # Determine which boxes to process
    if args.box_list_file is not None:
        if not os.path.exists(args.box_list_file):
            raise FileNotFoundError(f"Box list file not found: {args.box_list_file}")
        box_ids = []
        with open(args.box_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    box_ids.append(int(line))
                except ValueError:
                    if args.sim_name == 'FIRE':
                        box_ids.append(str(line))
                        logger.info(f"Adding FIRE simulation name: {line}")
                    else:
                        logger.warning(f"Skipping invalid line in box list file: {line}")
        if len(box_ids) == 0:
            raise ValueError(f"No valid box IDs found in file: {args.box_list_file}")
        box_ids = sorted(set(box_ids))
        logger.info(f"Loaded {len(box_ids)} box IDs from file: {args.box_list_file}")
    else:
        box_ids = list(range(args.box_ini, args.box_end + 1))

    # import the snapshot time file
    if args.sim_name.startswith('TNG'):
        snapshot_time = np.loadtxt(
            os.path.join(main_path, 'ML_dm_distribution/training_prep/snapshot_time_TNG.txt'),
            comments='#', delimiter=' ', dtype=[('snap', 'i4'), ('a', 'f8'), ('z', 'f8')]
        )
    elif args.sim_name == 'FIRE':
        # default to take the snapshot_times.txt file from the first sim in the list
        snapshot_time = np.loadtxt(
            os.path.join(fire_snap_path, f'{box_ids[0]}/snapshot_times.txt'),
            comments='#', dtype=[('snap', 'i4'), ('a', 'f8'), ('z', 'f8'), ('time', 'f8'), ('lookback_time', 'f8'), ('time_width', 'f8')]
        )
    else:
        snapshot_time = np.loadtxt(
            os.path.join(main_path, 'ML_dm_distribution/training_prep/snapshot_time.txt'),
            comments='#', delimiter=' ', dtype=[('z', 'f8'), ('a', 'f8'), ('snap', 'i4')]
        )

    # Handle CSV file input for automatic FOV calculation per box
    csv_data = None
    if args.redshift_csv is not None:
        if not os.path.exists(args.redshift_csv):
            raise ValueError(f'CSV file not found: {args.redshift_csv}')
        
        # Read all redshifts and FOVs from CSV (each row corresponds to a box)
        csv_redshifts, csv_fovs, csv_resolutions = read_redshift_csv(args.redshift_csv)
        csv_data = {
            'redshifts': csv_redshifts,
            'fovs_kpc': csv_fovs,
            'resolutions_kpc': csv_resolutions
        }
        
        # Check that CSV has enough rows for the box range
        if args.sim_name == 'FIRE':
            max_box = len(box_ids) # FIRE has no box IDs, so use the number of simulation names
        else:
            max_box = max(box_ids)
        if len(csv_redshifts) <= max_box:
            raise ValueError(f'CSV file has {len(csv_redshifts)} rows but need at least {max_box + 1} rows for box IDs up to {max_box}')
        
        logger.info(f"Loaded CSV with {len(csv_redshifts)} entries. Each box will use its corresponding row's FOV.")
        logger.info(f"CSV redshift range: {min(csv_redshifts):.3f} to {max(csv_redshifts):.3f}")
        logger.info(f"CSV FOV range: {min(csv_fovs):.3f} to {max(csv_fovs):.3f} kpc")
    
    # Set default values if not provided and CSV not used
    # When CSV is used, these will be calculated per box
    default_xmin = args.xmin if args.xmin is not None else -32
    default_xmax = args.xmax if args.xmax is not None else 32
    default_ymin = args.ymin if args.ymin is not None else -32
    default_ymax = args.ymax if args.ymax is not None else 32
    default_zmin = args.zmin if args.zmin is not None else -32
    default_zmax = args.zmax if args.zmax is not None else 32
    default_zmin_2d = args.zmin_2d if args.zmin_2d is not None else default_zmin
    default_zmax_2d = args.zmax_2d if args.zmax_2d is not None else default_zmax

    # Simulation redshift is always z=0 (snap 90) - CSV redshifts are for FOV calculation only
    # confirm that either snap or redshift is provided (for simulation snapshot)
    if args.snap is None and args.redshift is None:
        # Default to z=0 (snap 90) for simulation
        if args.sim_name == 'FIRE':
            args.snap = 600
            logger.info("No snap or redshift provided, defaulting to snap 600 (z=0)")
        else:
            args.snap = 90
        logger.info("No snap or redshift provided, defaulting to snap 90 (z=0)")
    
    if args.snap is not None and args.redshift is not None:
        # confirm that the snap number is consistent with the redshift
        if args.snap != get_snap_from_redshift(snapshot_time, args.redshift):
            raise ValueError(f'Snap number is inconsistent with redshift: {args.snap} != {get_snap_from_redshift(snapshot_time, args.redshift)}')
    
    # confirm that either 2D or 3D maps must be generated
    if not args.get_maps and not args.get_grids:
        raise ValueError('Either 2D or 3D maps must be generated')

    # get the snap number and scale factor (simulation redshift, always z=0)
    if args.redshift is not None:
        snap = get_snap_from_redshift(snapshot_time, args.redshift)
        sim_redshift = args.redshift
        a = get_a_from_snap_or_redshift(snapshot_time, snap=snap)
    else:
        snap = args.snap
        sim_redshift = get_redshift_from_snap(snapshot_time, snap)
        a = get_a_from_snap_or_redshift(snapshot_time, snap=snap)

    # print the snap number and redshift
    logger.info(f"Starting analysis with parameters:")
    logger.info(f"Simulation: {args.sim_name}")
    logger.info(f"Box range: {box_ids[0]} to {box_ids[-1]} ({len(box_ids)} boxes)" if args.box_list_file is None
                else f"Box list: {len(box_ids)} boxes from {args.box_list_file}")
    logger.info(f"Resolution: {args.resolution}")
    if csv_data is not None:
        logger.info(f"Using CSV file: FOV will be calculated per box from CSV")
        logger.info(f"CSV redshift range: {min(csv_data['redshifts']):.3f} to {max(csv_data['redshifts']):.3f}")
        logger.info(f"CSV FOV range: {min(csv_data['fovs_kpc']):.3f} to {max(csv_data['fovs_kpc']):.3f} kpc")
    else:
        logger.info(f"Grid range: x=[{default_xmin}, {default_xmax}], y=[{default_ymin}, {default_ymax}], z=[{default_zmin_2d}, {default_zmax_2d}] (2D map z-range)")
        logger.info(f"3D grid z-range: z=[{default_zmin}, {default_zmax}] (for 3D grids only)")
    logger.info(f"Snap number: {snap}, Simulation redshift: {sim_redshift:.3f}, Scale factor: {a:.3f}")
    logger.info(f"Particle types: {args.part_type}")
    logger.info(f"Map types: {args.map_type}")
    logger.info(f"2D maps: {args.get_maps}, 3D grids: {args.get_grids}")

    # set the grid parameters
    resolution = args.resolution
    map_type = args.map_type

    # At the beginning of main, create directories if they don't exist
    # Note: suffix will be calculated per box when CSV is used
    if csv_data is None:
        suffix_2d = f"x{default_xmin:.0f}_{default_xmax:.0f}_y{default_ymin:.0f}_{default_ymax:.0f}_z{default_zmin_2d:.0f}_{default_zmax_2d:.0f}"
        suffix_3d = f"x{default_xmin:.0f}_{default_xmax:.0f}_y{default_ymin:.0f}_{default_ymax:.0f}_z{default_zmin:.0f}_{default_zmax:.0f}"
    else:
        # Use suffix with redshift range from CSV (since each box has different FOV)
        z_min = min(csv_data['redshifts'])
        z_max = max(csv_data['redshifts'])
        # Format redshift range for filename (e.g., z0p3_0p4 for 0.3-0.4)
        z_min_str = f"{z_min:.3f}".replace('.', 'p')
        z_max_str = f"{z_max:.3f}".replace('.', 'p')
        suffix_2d = f"csv_fov_z{z_min_str}_{z_max_str}"
        suffix_3d = f"csv_fov_z{z_min_str}_{z_max_str}"
    # Output base paths depend on simulation
    if args.sim_name == 'DREAMS':
        rel_output_base_2d = 'DREAMS_data/2D_maps/CDM/MW_zooms/SB5'
        rel_output_base_3d = 'DREAMS_data/3D_grids/CDM/MW_zooms/SB5'
    elif args.sim_name == 'DREAMS_VM':
        rel_output_base_2d = 'DREAMS_data/2D_maps/CDM/varied_mass/SB6'
        rel_output_base_3d = 'DREAMS_data/3D_grids/CDM/varied_mass/SB6'
    elif args.sim_name == 'TNG50':
        rel_output_base_2d = 'TNG_data/2D_maps/TNG50'
        rel_output_base_3d = 'TNG_data/3D_grids/TNG50'
    elif args.sim_name == 'TNG100':
        rel_output_base_2d = 'TNG_data/2D_maps/TNG100'
        rel_output_base_3d = 'TNG_data/3D_grids/TNG100'
    elif args.sim_name == 'FIRE':
        rel_output_base_2d = 'FIRE_data/2D_maps/'
        rel_output_base_3d = 'FIRE_data/3D_grids/'
    else:
        raise ValueError(f"Unknown simulation: {args.sim_name}")
    output_root = args.output_dir if args.output_dir is not None else main_path
    output_base_2d = os.path.join(output_root, rel_output_base_2d)
    output_base_3d = os.path.join(output_root, rel_output_base_3d)
    output_sim_label = args.sim_name  # for filenames: DREAMS, DREAMS_VM, TNG50, or TNG100
    logger.info(f"Output root directory: {output_root}")

    if args.save_map:
        for mt in map_type:
            if args.get_maps:
                os.makedirs(f'{output_base_2d}/temp', exist_ok=True)
            if args.get_grids:
                os.makedirs(f'{output_base_3d}/temp', exist_ok=True)

    # loop over the boxes (missing halo data is skipped; boxes_processed tracks which were written)
    boxes_processed = []
    for idx, box in enumerate(box_ids):
        logger.info(f"Processing box {box} ({idx + 1}/{len(box_ids)})")

        # Calculate xyz ranges for this box
        if csv_data is not None:
            # Use CSV data: box index corresponds to CSV row index
            if args.sim_name == 'FIRE':
                box_idx = idx
                box_fov_kpc = csv_data['fovs_kpc'][box_idx]
                box_obs_redshift = csv_data['redshifts'][box_idx]
            else:
                box_fov_kpc = csv_data['fovs_kpc'][box]
                box_obs_redshift = csv_data['redshifts'][box]
            fov_half = box_fov_kpc / 2.0
            xmin = -fov_half
            xmax = fov_half
            ymin = -fov_half
            ymax = fov_half
            zmin = -fov_half
            zmax = fov_half
            zmin_2d = zmin
            zmax_2d = zmax
            logger.info(f"Box {box}: Observed redshift={box_obs_redshift:.3f}, FOV={box_fov_kpc:.3f} kpc, Range=±{fov_half:.3f} kpc")
        else:
            # Use default or manually specified ranges
            xmin = default_xmin
            xmax = default_xmax
            ymin = default_ymin
            ymax = default_ymax
            zmin = default_zmin
            zmax = default_zmax
            zmin_2d = default_zmin_2d
            zmax_2d = default_zmax_2d

        # Skip box if required halo/snapshot data is missing
        if args.sim_name == 'DREAMS':
            snap_file = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
            group_file = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
            coords_file = f'{potential_path}{args.rmax_ctr:.0f}kpc/{box}_{snap:d}_coords.txt'
            if not os.path.exists(snap_file):
                logger.warning(f"Box {box}: snapshot not found ({snap_file}). Skipping.")
                continue
            if not os.path.exists(group_file):
                logger.warning(f"Box {box}: group catalog not found ({group_file}). Skipping.")
                continue
            if not os.path.exists(coords_file):
                logger.warning(f"Box {box}: halo coords not found ({coords_file}). Skipping.")
                continue
        elif args.sim_name == 'DREAMS_VM':
            snap_file = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
            group_file = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
            if not os.path.exists(snap_file):
                logger.warning(f"Box {box}: snapshot not found ({snap_file}). Skipping.")
                continue
            if not os.path.exists(group_file):
                logger.warning(f"Box {box}: group catalog not found ({group_file}). Skipping.")
                continue
        elif args.sim_name.startswith('TNG'):
            if args.sim_name == 'TNG50':
                halo_file = f'{tng_snap_path}/TNG50/tng50_snap_{snap}_halo_{box}.h5'
            else:
                halo_file = f'{tng_snap_path}/TNG100/tng100_snap_{snap}_halo_{box}.h5'
            if not os.path.exists(halo_file):
                logger.warning(f"Box {box}: halo data not found ({halo_file}). Skipping.")
                continue
        elif args.sim_name == 'FIRE':
            snap_file = f'{fire_snap_path}/{box}/output/snapdir_{snap:03}/snapshot_{snap:03}.0.hdf5'
            if not os.path.exists(snap_file):
                logger.warning(f"Sim {box}: snapshot not found ({snap_file}). Skipping.")
                continue
        
        # loop over map types once before loading data, and skip fully processed boxes
        if csv_data is not None:
            box_suffix_2d_pre = f"x{xmin:.1f}_{xmax:.1f}_y{ymin:.1f}_{ymax:.1f}_z{zmin_2d:.1f}_{zmax_2d:.1f}"
            box_suffix_3d_pre = f"x{xmin:.1f}_{xmax:.1f}_y{ymin:.1f}_{ymax:.1f}_z{zmin:.1f}_{zmax:.1f}"
        else:
            box_suffix_2d_pre = suffix_2d
            box_suffix_3d_pre = suffix_3d

        if args.save_map:
            box_already_processed = True
            for mt in map_type:
                maps_exist = True
                grids_exist = True
                if args.get_maps:
                    maps_exist = os.path.exists(
                        f'{output_base_2d}/temp/Maps_{mt}_box_{box}_res_{resolution}_{box_suffix_2d_pre}.npy'
                    )
                if args.get_grids:
                    grids_exist = os.path.exists(
                        f'{output_base_3d}/temp/Grids_{mt}_box_{box}_res_{resolution}_{box_suffix_3d_pre}.npy'
                    )
                if not (maps_exist and grids_exist):
                    box_already_processed = False
                    break
            if box_already_processed:
                logger.info(
                    f"Box {box}: all requested temp outputs already exist for all map types. Skipping data load."
                )
                boxes_processed.append(box)
                continue


        # get the data (DREAMS: full snapshot + group catalog + halo coords; DREAMS_VM: load_particle_data_DREAMS with target FOF; TNG: pre-extracted halo file)
        if args.sim_name == 'DREAMS':
            part_cat, h = get_data(box, snap, args.part_type)
            # prt_cat, fof_cat = get_MW_halo(box, snap, part_cat, h)
            prt_cat = get_MW_halo_fof(box, snap, h, args.part_type) # some repeated loading of the data here but oh well
            gal_pos, vels_cen, rotation_tensor, axis_ratios = get_halo_coords(box, snap, args.rmax_ctr)
        elif args.sim_name == 'DREAMS_VM':
            path = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
            gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
            fof_idx = mvs.get_target_fof_index(box, snap, target_masses[box])
            logger.info(f"Box {box}: FOF index {fof_idx}")
            prt_cat, h, gal_pos, vels_cen, rotation_tensor = load_particle_data_DREAMS(
                path,
                ['Masses', 'Coordinates', 'Velocities', 'Potential', 'SubfindDMDensity', 'SubfindDensity'],
                args.part_type,
                gpath,
                fof_idx,
            )
            prt_cat = mvs.load_single_halo(box, snap, fof_idx=fof_idx, part_types=args.part_type,
                keys=['ParticleIDs','Masses', 'Coordinates', 'Velocities', 'Potential', 'SubfindDMDensity', 'SubfindDensity'],
                _nsubs_limiter=99999) # some repeated loading of the data here but oh well
        elif args.sim_name.startswith('TNG'):
            prt_cat, h, gal_pos, vels_cen, rotation_tensor = get_data_TNG(box, snap, args.part_type, args.sim_name)
        elif args.sim_name == 'FIRE':
            prt_cat, h, gal_pos, vels_cen, rotation_tensor = load_particle_data_FIRE(f'{fire_snap_path}/{box}', 
            snap, ['Masses', 'Coordinates', 'Velocities'], args.part_type)
            h = 1 # FIRE loading does the conversion from cosmological to physical units (all a and h factors are taken care of) already; but masses are still in units of 1e10 M_sun
        else:
            raise ValueError(f"Unknown simulation: {args.sim_name}")

        # center and rescale the data
        masses, potentials, densities, densitiesDM, coords_rot, vels_rot = center_and_rescale(prt_cat, gal_pos, vels_cen, rotation_tensor, args.part_type, a, h)

        # Calculate suffix for this box (for file naming)
        # Use separate variables to avoid overwriting the original suffix
        if csv_data is not None:
            # Include box-specific FOV in suffix for temporary files
            box_suffix_2d = f"x{xmin:.1f}_{xmax:.1f}_y{ymin:.1f}_{ymax:.1f}_z{zmin_2d:.1f}_{zmax_2d:.1f}"
            box_suffix_3d = f"x{xmin:.1f}_{xmax:.1f}_y{ymin:.1f}_{ymax:.1f}_z{zmin:.1f}_{zmax:.1f}"
        else:
            # Use the original suffix when CSV is not used
            box_suffix_2d = suffix_2d
            box_suffix_3d = suffix_3d

        for mt in map_type:
            logger.info(f"Processing {mt} maps for box {box}")
            # check if the box is already processed in temp folder
            if args.save_map:
                if args.get_maps and not args.get_grids:
                    if os.path.exists(f'{output_base_2d}/temp/Maps_{mt}_box_{box}_res_{resolution}_{box_suffix_2d}.npy'):
                        logger.info(f"Box {box} {mt} 2D maps with resolution {resolution} and 2d range {box_suffix_2d} already processed. Skipping...")
                        continue
                if args.get_grids and not args.get_maps:
                    if os.path.exists(f'{output_base_3d}/temp/Grids_{mt}_box_{box}_res_{resolution}_{box_suffix_3d}.npy'):
                        logger.info(f"Box {box} {mt} 3D grids with resolution {resolution} and 3d range {box_suffix_3d} already processed. Skipping...")
                        continue
                if args.get_maps and args.get_grids:
                    if os.path.exists(f'{output_base_2d}/temp/Maps_{mt}_box_{box}_res_{resolution}_{box_suffix_2d}.npy') and os.path.exists(f'{output_base_3d}/temp/Grids_{mt}_box_{box}_res_{resolution}_{box_suffix_3d}.npy'):
                        logger.info(f"Box {box} {mt} 2D maps and 3D grids with resolution {resolution} and 2d range {box_suffix_2d} and 3d range {box_suffix_3d} already processed. Skipping...")
                        continue
            start_time = datetime.now()
            # get the 2D maps
            if args.rescale:
                if args.get_maps:
                    maps_tmp, mean, std = get_2D_maps(coords_rot, vels_rot, masses, potentials, densitiesDM, densities, mt, resolution, xmin, xmax, ymin, ymax, zmin, zmax, zmin_2d, zmax_2d, args.rmax_ctr, box, snap, args.rescale)
                if args.get_grids:
                    grids_tmp, gmean, gstd = get_3D_grids(coords_rot, vels_rot, masses, potentials, densitiesDM, densities, mt, resolution, xmin, xmax, ymin, ymax, zmin, zmax, args.rmax_ctr, box, snap, args.rescale)
                
                # Save statistics
                if args.save_map:
                    if args.get_maps:
                        # Use per-box suffix when CSV is used
                        with open(f'{output_base_2d}/temp/stats_{mt}_res_{resolution}_{box_suffix_2d}.txt', 'a') as f:
                            f.write(f"{box},{mean},{std}\n")
                    if args.get_grids:
                        # Use per-box suffix when CSV is used
                        with open(f'{output_base_3d}/temp/stats_{mt}_res_{resolution}_{box_suffix_3d}.txt', 'a') as f:
                            f.write(f"{box},{gmean},{gstd}\n")
            else:
                if args.get_maps:
                    maps_tmp = get_2D_maps(coords_rot, vels_rot, masses, potentials, densitiesDM, densities, mt, resolution, xmin, xmax, ymin, ymax, zmin, zmax, zmin_2d, zmax_2d, args.rmax_ctr, box, snap, args.rescale)
                if args.get_grids:
                    grids_tmp = get_3D_grids(coords_rot, vels_rot, masses, potentials, densitiesDM, densities, mt, resolution, xmin, xmax, ymin, ymax, zmin, zmax, args.rmax_ctr, box, snap, args.rescale)

            # Save temporary files
            if args.save_map:
                if args.get_maps:
                    np.save(f'{output_base_2d}/temp/Maps_{mt}_box_{box}_res_{resolution}_{box_suffix_2d}.npy', maps_tmp)
                if args.get_grids:
                    np.save(f'{output_base_3d}/temp/Grids_{mt}_box_{box}_res_{resolution}_{box_suffix_3d}.npy', grids_tmp)

            # for debugging; do imshow of the maps to check if the x-y indexing is correct
            # limit the range of the colorbar to -3 to 3 for better visualization if rescale is true
            # plot only a selected few boxes to reduce the number of plots
            if box in [0, 1, 86, 100, 1000]:
                if args.rescale:
                    if args.get_maps:
                        plt.imshow(maps_tmp, vmin=-4, vmax=4)
                        plt.colorbar()
                        plt.savefig(f'{output_base_2d}/Maps_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{box_suffix_2d}_box={box}_rescaled.png')
                        plt.close()
                else:
                    if args.get_maps:
                        plt.imshow(maps_tmp)
                        plt.colorbar()
                        plt.savefig(f'{output_base_2d}/Maps_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{box_suffix_2d}_box={box}.png')
                        plt.close()

            processing_time = datetime.now() - start_time
            logger.info(f"Completed {mt} maps for box {box}. Processing time: {processing_time}")

            if args.get_maps:
                del maps_tmp
            if args.get_grids:
                del grids_tmp

        boxes_processed.append(box)

    # After the loop, combine all files (only boxes that had data and were processed)
    logger.info("Combining temporary files into final arrays...")
    if args.save_map:
        if not boxes_processed:
            logger.warning("No boxes were processed (all missing or skipped). Nothing to combine.")
        if boxes_processed:
            for mt in map_type:
                maps_combined = []
                grids_combined = []
                means = []
                stds = []
                
                for box in boxes_processed:
                    # Calculate suffix for this box (same as in main loop)
                    if csv_data is not None:
                        if args.sim_name == 'FIRE':
                            box_idx = idx
                            box_fov_kpc = csv_data['fovs_kpc'][box_idx]
                            box_obs_redshift = csv_data['redshifts'][box_idx]
                        else:
                            box_fov_kpc = csv_data['fovs_kpc'][box]
                            box_obs_redshift = csv_data['redshifts'][box]
                        fov_half = box_fov_kpc / 2.0
                        box_xmin = -fov_half
                        box_xmax = fov_half
                        box_ymin = -fov_half
                        box_ymax = fov_half
                        box_zmin = -fov_half
                        box_zmax = fov_half
                        box_zmin_2d = box_zmin
                        box_zmax_2d = box_zmax
                        box_suffix_2d = f"x{box_xmin:.1f}_{box_xmax:.1f}_y{box_ymin:.1f}_{box_ymax:.1f}_z{box_zmin_2d:.1f}_{box_zmax_2d:.1f}"
                        box_suffix_3d = f"x{box_xmin:.1f}_{box_xmax:.1f}_y{box_ymin:.1f}_{box_ymax:.1f}_z{box_zmin:.1f}_{box_zmax:.1f}"
                    else:
                        box_suffix_2d = suffix_2d
                        box_suffix_3d = suffix_3d
                    
                    # Load temporary files
                    if args.get_maps:
                        map_file = f'{output_base_2d}/temp/Maps_{mt}_box_{box}_res_{resolution}_{box_suffix_2d}.npy'
                    if args.get_grids:
                        grid_file = f'{output_base_3d}/temp/Grids_{mt}_box_{box}_res_{resolution}_{box_suffix_3d}.npy'
                    
                    if args.get_maps:
                        maps_combined.append(np.load(map_file))
                    if args.get_grids:
                        grids_combined.append(np.load(grid_file))
                    
                    # Clean up temporary files
                    if args.get_maps:
                        os.remove(map_file)
                    if args.get_grids:
                        os.remove(grid_file)
                
                # Stack arrays
                if args.get_maps:
                    maps_combined = np.stack(maps_combined, axis=0)
                if args.get_grids:
                    grids_combined = np.stack(grids_combined, axis=0)
                
                # Save combined arrays
                # Use suffix with redshift range when CSV is used (since FOV varies per box)
                # The suffix already includes redshift range from earlier calculation
                final_suffix_2d = suffix_2d
                final_suffix_3d = suffix_3d
                
                if args.rescale:
                    if args.get_maps:
                        np.save(f'{output_base_2d}/Maps_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_2d}_rescaled.npy', maps_combined)
                    if args.get_grids:
                        np.save(f'{output_base_3d}/Grids_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_3d}_rescaled.npy', grids_combined)
                    
                    # Combine statistics
                    # Note: stats files use per-box suffixes, so we need to collect them
                    if args.get_maps:
                        if csv_data is not None:
                            # Collect stats from all boxes (each has different suffix)
                            all_stats = []
                            for box in boxes_processed:
                                if args.sim_name == 'FIRE':
                                    box_idx = idx
                                    box_fov_kpc = csv_data['fovs_kpc'][box_idx]
                                    box_obs_redshift = csv_data['redshifts'][box_idx]
                                else:
                                    box_fov_kpc = csv_data['fovs_kpc'][box]
                                    box_obs_redshift = csv_data['redshifts'][box]
                                fov_half = box_fov_kpc / 2.0
                                box_suffix_2d = f"x{-fov_half:.1f}_{fov_half:.1f}_y{-fov_half:.1f}_{fov_half:.1f}_z{-fov_half:.1f}_{fov_half:.1f}"
                                stats_file = f'{output_base_2d}/temp/stats_{mt}_res_{resolution}_{box_suffix_2d}.txt'
                                if os.path.exists(stats_file):
                                    box_stats = np.loadtxt(stats_file, delimiter=',')
                                    all_stats.append(box_stats)
                                    os.remove(stats_file)
                            if all_stats:
                                stats = np.vstack(all_stats)
                                np.save(f'{output_base_2d}/Maps_mean_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_2d}.npy', stats[:,1])
                                np.save(f'{output_base_2d}/Maps_std_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_2d}.npy', stats[:,2])
                        else:
                            stats = np.loadtxt(f'{output_base_2d}/temp/stats_{mt}_res_{resolution}_{final_suffix_2d}.txt', delimiter=',')
                            np.save(f'{output_base_2d}/Maps_mean_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_2d}.npy', stats[:,1])
                            np.save(f'{output_base_2d}/Maps_std_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_2d}.npy', stats[:,2])
                            os.remove(f'{output_base_2d}/temp/stats_{mt}_res_{resolution}_{final_suffix_2d}.txt')
                    if args.get_grids:
                        if csv_data is not None:
                            # Collect stats from all boxes (each has different suffix)
                            all_stats = []
                            for box in boxes_processed:
                                if args.sim_name == 'FIRE':
                                    box_idx = idx
                                    box_fov_kpc = csv_data['fovs_kpc'][box_idx]
                                    box_obs_redshift = csv_data['redshifts'][box_idx]
                                else:
                                    box_fov_kpc = csv_data['fovs_kpc'][box]
                                    box_obs_redshift = csv_data['redshifts'][box]
                                fov_half = box_fov_kpc / 2.0
                                box_suffix_3d = f"x{-fov_half:.1f}_{fov_half:.1f}_y{-fov_half:.1f}_{fov_half:.1f}_z{-fov_half:.1f}_{fov_half:.1f}"
                                stats_file = f'{output_base_3d}/temp/stats_{mt}_res_{resolution}_{box_suffix_3d}.txt'
                                if os.path.exists(stats_file):
                                    box_stats = np.loadtxt(stats_file, delimiter=',')
                                    all_stats.append(box_stats)
                                    os.remove(stats_file)
                            if all_stats:
                                stats = np.vstack(all_stats)
                                np.save(f'{output_base_3d}/Grids_mean_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_3d}.npy', stats[:,1])
                                np.save(f'{output_base_3d}/Grids_std_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_3d}.npy', stats[:,2])
                        else:
                            stats = np.loadtxt(f'{output_base_3d}/temp/stats_{mt}_res_{resolution}_{final_suffix_3d}.txt', delimiter=',')
                            np.save(f'{output_base_3d}/Grids_mean_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_3d}.npy', stats[:,1])
                            np.save(f'{output_base_3d}/Grids_std_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_3d}.npy', stats[:,2])
                            os.remove(f'{output_base_3d}/temp/stats_{mt}_res_{resolution}_{final_suffix_3d}.txt')
                else:
                    if args.get_maps:
                        np.save(f'{output_base_2d}/Maps_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_2d}.npy', maps_combined)
                    if args.get_grids:
                        np.save(f'{output_base_3d}/Grids_{mt}_{output_sim_label}_{resolution}_z={sim_redshift:.2f}_{final_suffix_3d}.npy', grids_combined)

            # Clean up temporary directories (once after all map types)
            import shutil
            if args.get_maps:
                shutil.rmtree(f'{output_base_2d}/temp')
            if args.get_grids:
                shutil.rmtree(f'{output_base_3d}/temp')

        logger.info("All processing complete!")
        