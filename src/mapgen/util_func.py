# some utility functions for loading DREAMS data

import numpy as np
import h5py, os
from scipy.spatial import KDTree
import constant as constant
try:
    from gizmopy.load_fire_snap import load_fire_snap
    import gizmo_analysis as ga
    HAS_GIZMO = True
except ImportError:
    HAS_GIZMO = False
    print("gizmopy or gizmo_analysis not found. Skipping FIRE data loading.")

# Define the path to the data
# main_path = '/Users/xwou/Documents/Research/MIT/ML_dm_distribution/DREAMS_data'
# # join the path to the data
# snap_path = os.path.join(main_path,'Sims/CDM/MW_zooms/SB5/')
# group_path = os.path.join(main_path,'FOF_Subfind/CDM/MW_zooms/SB5/')
# param_path = os.path.join(main_path,'Parameters/CDM/MW_zooms/WDM_TNG_MW_SB5.txt')

# important parameters (define them in the actual analysis script)
# box = 0       #which simulation is used [0,1023]
# snap = 90     #which snapshot 90 -> z=0; will work up to z~1 as written
# part_type = 1 #which particle type to calculate the density
# h = .6909     #reduced hubble constant (do not change) for DREAMS
# rmin = 0.4    #the radius at which to start calculating the density
# rmax = 50     #the radius at which to stop calculating the density
# sphere_samples = 50 #number of measurement samples for each radius
# radial_samples = 50  #number of radii to measure density 
# DesNgb = 32   #how many particles to use when estimating density

def load_particle_data(path, keys, part_types):
    """
    Read particle data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
    """
    cat = dict()
    with h5py.File(path) as ofile:
        h = ofile['Header'].attrs['HubbleParam']
        if type(part_types) == type(0):
            part_types = [part_types]
        
        for pt in part_types:
            for key in keys:
                if pt == 1 and key == 'Masses':
                    cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                else:
                    dataset_path = f'PartType{pt}/{key}'
                    if dataset_path in ofile:
                        try:
                            cat[dataset_path] = np.array(ofile[dataset_path])
                        except (KeyError, OSError, RuntimeError) as e:
                            # Handle corrupted datasets gracefully
                            # The error message often contains "Unable to open object" or "unable to determine object type"
                            import warnings
                            warnings.warn(f"Could not read {dataset_path} from {path}: {e}. Skipping this dataset.")
                            # Don't add it to the catalog - downstream code checks for existence anyway
    return cat, h

def load_particle_data_TNG(path, keys, part_types, resolution_tag):
    """
    Read particle data from the TNG simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
      h - the Hubble parameter
      gal_pos - the position of the galaxy
      vels_cen - the velocity of the galaxy
      rotation_tensor - the rotation tensor of the galaxy
    """
    cat = dict()
    with h5py.File(path) as ofile:
        h = ofile['Header'].attrs['HubbleParam']
        scale_factor = ofile['Header'].attrs['Time']
        # Copy to numpy arrays; otherwise dataset refs become invalid when file closes
        gal_pos = np.asarray(ofile['CenterPosition']) * scale_factor / h # convert to physical units
        vels_cen = np.asarray(ofile['CenterVelocity']) / scale_factor # convert to physical units
        rotation_tensor = np.asarray(ofile['RotationMatrix'])

        if type(part_types) == type(0):
            part_types = [part_types]
        
        for pt in part_types:
            for key in keys:
                if pt == 1 and key == 'Masses':
                    if resolution_tag == 'TNG50':
                        cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                    elif resolution_tag == 'TNG100':
                        cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                else:
                    dataset_path = f'PartType{pt}/{key}'
                    if dataset_path in ofile:
                        try:
                            cat[dataset_path] = np.array(ofile[dataset_path])
                        except (KeyError, OSError, RuntimeError) as e:
                            # Handle corrupted datasets gracefully
                            # The error message often contains "Unable to open object" or "unable to determine object type"
                            import warnings
                            warnings.warn(f"Could not read {dataset_path} from {path}: {e}. Skipping this dataset.")
                            # Don't add it to the catalog - downstream code checks for existence anyway
    return cat, h, gal_pos, vels_cen, rotation_tensor

def load_particle_data_DREAMS(path, keys, part_types, group_path, fof_idx):
    """
    Read particle data from the DREAMS simulations when need to provide center position and velocity of the galaxy
    and rotation tensor of the galaxy, and the FOF index of the target halo
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in DREAMS)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
      h - the Hubble parameter
      gal_pos - the position of the galaxy
      vels_cen - the velocity of the galaxy
      rotation_tensor - the rotation tensor of the galaxy
    """
    cat = dict()
    with h5py.File(path) as ofile:
        h = ofile['Header'].attrs['HubbleParam']
        scale_factor = ofile['Header'].attrs['Time']
        # Copy to numpy arrays; otherwise dataset refs become invalid when file closes
        # gal_pos = np.asarray(ofile['CenterPosition']) / h # convert to physical units
        # vels_cen = np.asarray(ofile['CenterVelocity']) * np.sqrt(scale_factor) / h # convert to physical units
        # rotation_tensor = np.asarray(ofile['RotationMatrix'])
        # above are only available for TNG simulations as they are saved when split from the main snapshot file
        # load the group catalog data from the group catalog path
        group_cat = load_group_data(group_path, ['GroupPos', 'GroupVel'])
        gal_pos = group_cat['GroupPos'][fof_idx] * scale_factor / h 
        vels_cen = group_cat['GroupVel'][fof_idx] / scale_factor

        if type(part_types) == type(0):
            part_types = [part_types]
        
        for pt in part_types:
            for key in keys:
                if pt == 1 and key == 'Masses':
                    cat[f'PartType{pt}/{key}'] = np.ones(ofile['PartType1/ParticleIDs'].shape)*ofile['Header'].attrs['MassTable'][1]
                else:
                    dataset_path = f'PartType{pt}/{key}'
                    if dataset_path in ofile:
                        try:
                            cat[dataset_path] = np.array(ofile[dataset_path])
                        except (KeyError, OSError, RuntimeError) as e:
                            # Handle corrupted datasets gracefully
                            # The error message often contains "Unable to open object" or "unable to determine object type"
                            import warnings
                            warnings.warn(f"Could not read {dataset_path} from {path}: {e}. Skipping this dataset.")
                            # Don't add it to the catalog - downstream code checks for existence anyway
        
        # compute the rotation tensor on the fly
        coords = cat['PartType4/Coordinates'] * scale_factor / h - gal_pos
        vels = cat['PartType4/Velocities'] * np.sqrt(scale_factor) - vels_cen
        masses = cat['PartType4/Masses'] * 1e10 / h
        rmax_mask = np.sqrt(np.sum(coords**2, axis=1)) < (10.0)
        # print(len(coords), len(vels), len(masses), len(rmax_mask))
        rotation_tensor = get_rot_mat_angular_momentum_vertical(coords, vels, masses, masks=rmax_mask)
    return cat, h, gal_pos, vels_cen, rotation_tensor

def load_particle_data_FIRE(sdir, snum, keys, part_types):
    """
    Read particle data from the FIRE simulations
    
    Inputs
      sdir - the directory of the FIRE simulation
      snum - the snapshot number of the FIRE simulation
      keys - the data that you want to read from the simulation 
      part_types - which particle types to load.
                   0 - gas
                   1 - high res dark matter
                   2 - low res dark matter
                   3 - tracers (not used in FIRE)
                   4 - stars
                   5 - black holes
      
    Returns
      cat - a dictionary that contains all of the particle information for the specified keys and particle types
      h - the Hubble parameter
      gal_pos - the position of the galaxy
      vels_cen - the velocity of the galaxy
      rotation_tensor - the rotation tensor of the galaxy
    """
    if not HAS_GIZMO:
        raise ImportError("gizmopy or gizmo_analysis not found. Skipping FIRE data loading.")

    cat = dict()
    sdir_output = os.path.join(sdir, 'output')
    h = load_fire_snap('HubbleParam', -1, sdir_output,snum)
    scale_factor = load_fire_snap('Time', -1, sdir_output,snum)

    if type(part_types) == type(0):
            part_types = [part_types]

    for pt in part_types:
        for key in keys:
            dataset_path = f'PartType{pt}/{key}'
            try:
                cat[dataset_path] = np.array(load_fire_snap(key, pt, sdir_output, snum))
            except (KeyError, OSError, RuntimeError) as e:
                # Handle corrupted datasets gracefully
                # The error message often contains "Unable to open object" or "unable to determine object type"
                import warnings
                warnings.warn(f"Could not read {dataset_path} from {sdir_output}/{snum}: {e}. Skipping this dataset.")
                # Don't add it to the catalog - downstream code checks for existence anyway
        
    # compute the center of mass position and velocity of the galaxy and the rotation tensor on the fly
    # read in the snapshot
    part = ga.io.Read.read_snapshots(species=['star'],
                                    snapshot_value_kind='index',
                                    snapshot_values=snum,
                                    simulation_directory=sdir,
                                    properties=['position', 'velocity','mass'],
                                    assign_hosts=True)
    gal_pos = np.asarray(part.host['position'], dtype=np.float64).reshape(-1, 3)[0] * scale_factor
    vels_cen = np.asarray(part.host['velocity'], dtype=np.float64).reshape(-1, 3)[0]
    coords = cat['PartType4/Coordinates'] - gal_pos
    vels = cat['PartType4/Velocities'] - vels_cen
    masses = cat['PartType4/Masses']
    rmax_mask = np.sqrt(np.sum(coords**2, axis=1)) < (10.0)
    # print(len(coords), len(vels), len(masses), len(rmax_mask))
    rotation_tensor = get_rot_mat_angular_momentum_vertical(coords, vels, masses, masks=rmax_mask)
    return cat, h, gal_pos, vels_cen, rotation_tensor

def get_MW_idx(cat, h):
    """
    Selects the corrent MW-mass galaxy from each simulation given the group catalog.
    This function only works for z~0
    It selects the least contaminated halo with a mass within current uncertainties of the MW's mass
    
    Inputs 
     - cat - a dictionary containing the 'GroupMassType' field from the FOF catalogs
     
    Returns
     - mw_idx - the index into the group catalog for the target MW-mass galaxy
    """
    masses = cat['GroupMassType'] * 1e10 / h
    
    tot_masses = np.sum(masses,axis=1)
    mcut = (tot_masses > 5e11) & (tot_masses < 2.5e12)
    
    contamination = masses[:,2] / tot_masses
    idx = np.argmin(contamination[mcut])
    
    mw_idx = np.arange(len(masses))[mcut][idx]
    return mw_idx


def load_group_data(path, keys):
    """
    Read Group Data from the DREAMS simulations
    
    Inputs
      path - the absolute or relative path to the hdf5 file you want to read from
      keys - the data that you want to read from the simulation 
             see https://www.tng-project.org/data/docs/specifications/ for a list of available data)
      
    Returns
      cat - a dictionary that contains all of the group and subhalo information for the specified keys
    """
    cat = dict()
    with h5py.File(path) as ofile:
        for key in keys:
            if 'Group' in key:
                cat[key] = np.array(ofile[f'Group/{key}'])
            if 'Subhalo' in key:
                cat[key] = np.array(ofile[f'Subhalo/{key}'])
    return cat

def get_galaxy_data(part_cat, group_cat, fof_idx=-1, sub_idx=-1):
    """
    Given particle and group catalogs, return a new catalog that only contains data for a specified galaxy.
    If fof_idx is given but sub_idx is not, data for the FOF group and all satellites are returned
    If fof_idx is given and sub_idx is given, data for just the specified subhalo of that group
      e.g. fof_idx=3 sub_idx=5 will provide data for the fifth subhalo of group three
    If only sub_idx is supplied, the data for that subfind galaxy is returned, can be in any FOF group
    
    Inputs
      part_cat  - a dictionary containing particle data make from load_particle_data
      group_cat - a dictionary containing group data make from load_particle_data
                  must contain these fields: GroupLenType, GroupFirstSub, GroupNsubs, SubhaloLenType, SubhaloGrNr
      fof_idx   - the FOF group that you want data for
      sub_idx   - the Subfind galaxy that you want data for
      
    Returns
      new_part_cat  - a new dictionary with keys from part_cat but data only for the specified galaxy
      new_group_cat - a new dictionary with keys from group_cat but data only for the specified galaxy
    """
    
    if fof_idx < 0 and sub_idx < 0:
        return part_cat, group_cat
    
    if fof_idx < 0 and sub_idx >= 0:
        fof_idx = group_cat['SubhaloGrNr'][sub_idx]

    if sub_idx >= 0:
        if sub_idx < group_cat['GroupFirstSub'][fof_idx]:
            sub_idx += group_cat['GroupFirstSub'][fof_idx]
    
    offsets = np.sum(group_cat['GroupLenType'][:fof_idx],axis=0)
    
    if sub_idx >= 1:
        start_sub = group_cat['GroupFirstSub'][fof_idx]
        offsets += np.sum(group_cat['SubhaloLenType'][start_sub:sub_idx], axis=0)
    
    if sub_idx < 0:
        num_parts = group_cat['GroupLenType'][fof_idx]
        nsubs = group_cat['GroupNsubs'][fof_idx]
        sub_start = group_cat['GroupFirstSub'][fof_idx]
    else:
        num_parts = group_cat['SubhaloLenType'][sub_idx]
        nsubs = 1
        sub_start = sub_idx
    
    new_part_cat = dict()
    for key in part_cat:
        pt = int(key.split("/")[0][-1])
        new_part_cat[key] = part_cat[key][offsets[pt]:offsets[pt]+num_parts[pt]]
    
    new_group_cat = dict()
    for key in group_cat:
        if 'Group' in key:
            new_group_cat[key] = group_cat[key][fof_idx]
        else:
            new_group_cat[key] = group_cat[key][sub_start:sub_start+nsubs]
    
    return new_part_cat, new_group_cat

def get_group_data(group_cat, fof_idx=-1, sub_idx=-1):
    """
    Given group catalog, return a new catalog that only contains data for a specified subhalo or FOF group.
    If fof_idx is given but sub_idx is not, data for the FOF group and all satellites are returned
    If fof_idx is given and sub_idx is given, data for just the specified subhalo of that group
      e.g. fof_idx=3 sub_idx=5 will provide data for the fifth subhalo of group three
    If only sub_idx is supplied, the data for that subfind galaxy is returned, can be in any FOF group
    
    Inputs
      group_cat - a dictionary containing group data make from load_particle_data
                  must contain these fields: GroupLenType, GroupFirstSub, GroupNsubs, SubhaloLenType, SubhaloGrNr
      fof_idx   - the FOF group that you want data for
      sub_idx   - the Subfind galaxy that you want data for
      
    Returns
      new_group_cat - a new dictionary with keys from group_cat but data only for the specified galaxy
    """
    
    if fof_idx < 0 and sub_idx < 0:
        return group_cat
    
    if fof_idx < 0 and sub_idx >= 0:
        fof_idx = group_cat['SubhaloGrNr'][sub_idx]

    if sub_idx >= 0:
        if sub_idx < group_cat['GroupFirstSub'][fof_idx]:
            sub_idx += group_cat['GroupFirstSub'][fof_idx]
    
    if sub_idx < 0:
        nsubs = group_cat['GroupNsubs'][fof_idx]
        sub_start = group_cat['GroupFirstSub'][fof_idx]
    else:
        nsubs = 1
        sub_start = sub_idx
    
    new_group_cat = dict()
    for key in group_cat:
        if 'Group' in key:
            new_group_cat[key] = group_cat[key][fof_idx]
        else:
            new_group_cat[key] = group_cat[key][sub_start:sub_start+nsubs]
    
    return new_group_cat

def fibonacci_sphere(samples, r):
    """
    This function returns a set of points equally spaced on the surface of a sphere.
    This function was adapted from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    
    Inputs
      samples - the number of points on the sphere
      r       - the radius of the sphere that is sampled
      
    Returns
      points  - the coordinates of the points of the sphere with shape (samples,3)
    """
    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / (samples - 1)) * 2  
        radius = np.sqrt(1 - y * y) 

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    points = np.array(points) * r
        
    return points

def calc_density(matter_coords, matter_mass, sample_coords, DesNgb):
    """
    This function takes particle data and calculates the density around a given set of points.
    The points are assumed to be for one radius and are averaged together before returned.
    This function will work for any particle type given to it.
    
    Inputs
      matter_coords - the coordinates for the simulation particles used to calculate the density
      matter_mass   - the mass of the simulation particles, must be the same shape as matter_coords
      sample_coords - the coordinates where the density is calculated
      DesNgb        - the number of simulation particles used to calculate the density (32 is standard)
      
    Returns
      density - the average density for the supplied sample coordinates
    """
    tree = KDTree(matter_coords)
    distance, idx = tree.query(sample_coords, DesNgb)
    hsml = distance[:,-1]
    mass_enclosed = np.sum(matter_mass[idx], axis=1)
    density =  mass_enclosed / (4 / 3 * np.pi * np.power(hsml,3))
    
    density = np.average(density)
    return density

# --------------------------------------------------------------------------------------------------
# position distances
# --------------------------------------------------------------------------------------------------
def get_positions_periodic(positions, periodic_length=None):
    '''
    Get position in range [0, periodic_length).

    Parameters
    ----------
    positions : float or array
    periodic_length : float
        periodicity length (if none, return array as is)
    '''
    if periodic_length is None:
        return positions

    if np.isscalar(positions):
        if positions >= periodic_length:
            positions -= periodic_length
        elif positions < 0:
            positions += periodic_length
    else:
        positions[positions >= periodic_length] -= periodic_length
        positions[positions < 0] += periodic_length

    return positions


def get_position_differences(position_difs, periodic_length=None):
    '''
    Get distance / separation vector, in range [-periodic_length/2, periodic_length/2).

    Parameters
    ----------
    position_difs : array
        position difference[s]
    periodic_length : float
        periodicity length (if none, return array as is)
    '''
    if not periodic_length:
        return position_difs
    else:
        if np.isscalar(periodic_length) and periodic_length <= 1:
            print(f'! got unusual periodic_length = {periodic_length}')

    if np.isscalar(position_difs):
        if position_difs >= 0.5 * periodic_length:
            position_difs -= periodic_length
        elif position_difs < -0.5 * periodic_length:
            position_difs += periodic_length
    else:
        position_difs[position_difs >= 0.5 * periodic_length] -= periodic_length
        position_difs[position_difs < -0.5 * periodic_length] += periodic_length

    return position_difs


def get_distances(
    positions_1=None, positions_2=None, periodic_length=None, scalefactor=None, total_distance=False
):
    '''
    Get vector or total/scalar distance[s] between input position vectors.
    If input scale-factors, will convert distance from comoving to physical.

    Parameters
    ----------
    positions_1 : array
        position[s]
    positions_2 : array
        position[s]
    periodic_length : float
        periodic length (if none, not use periodic)
    scalefactor : float or array
        expansion scale-factor (to convert comoving to physical)
    total : bool
        whether to compute total/scalar (instead of vector) distance

    Returns
    -------
    distances : array (object number x dimension number, or object number)
        vector or total/scalar distance[s]
    '''
    if not isinstance(positions_1, np.ndarray):
        positions_1 = np.array(positions_1)
    if not isinstance(positions_2, np.ndarray):
        positions_2 = np.array(positions_2)

    if len(positions_1.shape) == 1 and len(positions_2.shape) == 1:
        shape_pos = 0
    else:
        shape_pos = 1

    distances = get_position_differences(positions_1 - positions_2, periodic_length)

    if total_distance:
        distances = np.sqrt(np.sum(distances**2, shape_pos))

    if scalefactor is not None:
        if scalefactor > (1 + 1e-4) or scalefactor <= 0:
            print(f'! got unusual scalefactor = {scalefactor}')
        distances *= scalefactor

    return distances


def get_distances_angular(positions_1=None, positions_2=None, sphere_angle=360):
    '''
    Get angular separation[s] between input positions, valid for small separations.

    Parameters
    ----------
    positions_1, positions_2 : arrays
        positions in [RA, dec]
    sphere_angle : float
        angular size of sphere 360 [degrees], 2 * pi [radians]

    Returns
    -------
    angular distances : array (object number x angular dimension number)
    '''
    if sphere_angle == 360:
        angle_scale = constant.radian_per_degree
    elif sphere_angle == 2 * np.pi:
        angle_scale = 1
    else:
        raise ValueError(f'angle of sphere = {sphere_angle} does not make sense')

    if np.ndim(positions_1) == 1 and positions_1.size == 2:
        ras_1, decs_1 = positions_1[0], positions_1[1]
    else:
        ras_1, decs_1 = positions_1[:, 0], positions_1[:, 1]

    if np.ndim(positions_2) == 1 and positions_2.size == 2:
        ras_2, decs_2 = positions_2[0], positions_2[1]
    else:
        ras_2, decs_2 = positions_2[:, 0], positions_2[:, 1]

    return np.sqrt(
        (
            get_position_differences(ras_1 - ras_2, sphere_angle)
            * np.cos(angle_scale * 0.5 * (decs_1 + decs_2))
        )
        ** 2
        + (decs_1 - decs_2) ** 2
    )


# --------------------------------------------------------------------------------------------------
# velocity conversion
# --------------------------------------------------------------------------------------------------
def get_velocity_differences(
    velocity_vectors_1=None,
    velocity_vectors_2=None,
    position_vectors_1=None,
    position_vectors_2=None,
    periodic_length=None,
    scalefactor=None,
    hubble_time=None,
    total_velocity=False,
):
    '''
    Get relative velocity[s] [km/s] between input velocity vectors.
    If input positions as well, add Hubble flow to velocities.

    Parameters
    ----------
    velocity_vectors_1 : array (object number x dimension number)
        velocity[s] [km/s]
    velocity_vectors_2 : array (object number x dimension number)
        velocity[s]  [km/s]
    position_vectors_1 : array (object number x dimension number)
        position[s] associated with velocity_vector_1 [kpc comoving]
    position_vectors_2 : array (object number x dimension number)
        position[s] associated with velocity_vector_2 [kpc comoving]
    periodic_length : float
        periodicity length [kpc comoving]
    scalefactor : float
        expansion scale-factor
    hubble_time : float
        1 / H(z) [Gyr]
    total_velocity : bool
        whether to compute total/scalar (instead of vector) velocity

    Returns
    -------
    velocity_difs  : array (object number x dimension number, or object number)
        velocity differences [km/s]
    '''
    if np.ndim(velocity_vectors_1) == 1 and np.ndim(velocity_vectors_1) == 1:
        dimension_shape = 0
    else:
        dimension_shape = 1

    velocity_difs = velocity_vectors_1 - velocity_vectors_2  # [km/s]

    if position_vectors_1 is not None and position_vectors_2 is not None:
        # add hubble flow: dr/dt = a * dx/dt + da/dt * x = a(t) * dx/dt + r * H(t)
        # [kpc / Gyr]
        vels_hubble = (
            scalefactor
            / hubble_time
            * get_distances(position_vectors_1, position_vectors_2, periodic_length)
        )
        vels_hubble *= constant.km_per_kpc / constant.sec_per_Gyr  # [km/s]
        velocity_difs += vels_hubble

    if total_velocity:
        velocity_difs = np.sqrt(np.sum(velocity_difs**2, dimension_shape))

    return velocity_difs


# --------------------------------------------------------------------------------------------------
# center of mass: position and velocity
# --------------------------------------------------------------------------------------------------
def get_center_position(
    positions,
    weights=None,
    periodic_length=None,
    position_number_min=32,
    center_position=None,
    distance_max=np.inf,
):
    '''
    Get position of center of mass, using iterative zoom-in.

    Parameters
    ----------
    positions : array (particle number x dimension number)
        position[s]
    weights : array
        weight for each position (usually mass) - if None, assume all have same weight
    periodic_length : float
        periodic box length
    position_number_min : int
        minimum number of positions within distance to keep zooming in
    center_position : array
        initial center position to use
    distance_max : float
        maximum distance to consider initially

    Returns
    -------
    center_position : array
        position vector of center of mass
    '''
    distance_bins = np.array(
        [
            np.inf,
            1000,
            700,
            500,
            300,
            200,
            150,
            100,
            70,
            50,
            30,
            20,
            15,
            10,
            7,
            5,
            3,
            2,
            1.5,
            1,
            0.7,
            0.5,
            0.3,
            0.2,
            0.15,
            0.1,
            0.07,
            0.05,
            0.03,
            0.02,
            0.015,
            0.01,
            0.007,
            0.005,
            0.003,
            0.002,
            0.0015,
            0.001,
        ]
    )
    distance_bins = distance_bins[distance_bins <= distance_max]

    if weights is not None:
        assert positions.shape[0] == weights.size
        # normalize weights by median, improves numerical stability
        weights = np.asarray(weights) / np.median(weights)

    if center_position is None or len(center_position) == 0:
        center_position = np.zeros(positions.shape[1], positions.dtype)
    else:
        center_position = np.array(center_position, positions.dtype)

    if positions.shape[0] > 2147483647:
        idtype = np.int64
    else:
        idtype = np.int32
    part_indices = np.arange(positions.shape[0], dtype=idtype)

    for dist_i, dist_max in enumerate(distance_bins):
        # direct method ----------
        distance2s = (
            get_position_differences(positions[part_indices] - center_position, periodic_length)
            ** 2
        )
        distance2s = np.sum(distance2s, 1)

        # get particles within distance max
        masks = distance2s < dist_max**2
        part_indices_dist = part_indices[masks]

        # store particles slightly beyond distance max for next interation
        masks = distance2s < (1.5 * dist_max) ** 2
        part_indices = part_indices[masks]

        # kd-tree method ----------
        # if dist_i == 0:
        # create tree if this is the first distance bin
        #    KDTree = spatial.KDTree(positions, boxsize=part.info['box.length'])
        #    particle_number_max = positions.shape[0]

        # distances, indices = KDTree.query(
        #    center_position, particle_number_max, distance_upper_bound=dist_max, workers=2)

        # masks = (distances < dist_max)
        # part_indices_dist = indices[masks]
        # particle_number_max = part_indices_dist.size

        # check whether reached minimum total number of particles within distance
        # but force at least one loop over distance bins to get *a* center
        if part_indices_dist.size <= position_number_min and dist_i > 0:
            return center_position

        if weights is None:
            weights_use = weights
        else:
            weights_use = weights[part_indices_dist]

        # ensure that np.average uses 64-bit internally for accuracy, but returns as input dtype
        center_position = np.average(
            positions[part_indices_dist].astype(np.float64), 0, weights_use
        ).astype(positions.dtype)

    return center_position

def get_center_velocity(
    velocities,
    weights=None,
    positions=None,
    center_position=None,
    distance_max=20,
    periodic_length=None,
):
    '''
    Get velocity of center of mass.
    If no input masses, assume all masses are the same.

    Parameters
    ----------
    velocities : array (particle number x dimension_number)
        velocity[s]
    weights : array
        weight for each position (usually mass) - if None, assume all have same weight
    positions : array (particle number x dimension number)
        positions, if want to select by this
    center_position : array
        center position, if want to select by this
    distance_max : float
        maximum position difference from center to use particles
    periodic_length : float
        periodic box length

    Returns
    -------
    center_velocity : array
        velocity vector of center of mass
    '''
    masks = np.full(velocities.shape[0], True)

    # ensure that use only finite values
    for dimen_i in range(velocities.shape[1]):
        masks *= np.isfinite(velocities[:, dimen_i])

    if positions is not None and center_position is not None and len(center_position) > 0:
        assert velocities.shape == positions.shape
        distance2s = np.sum(
            get_position_differences(positions - center_position, periodic_length) ** 2, 1
        )
        masks *= distance2s < distance_max**2

    if weights is not None:
        assert velocities.shape[0] == weights.size
        # normalizing weights by median seems to improve numerical stability
        weights = weights[masks] / np.median(weights[masks])

    if not masks.any():
        print('! cannot compute host/center velocity')
        print('  no positions within distance_max = {:.3f} kpc comoving'.format(distance_max))
        print('  nearest = {:.3f} kpc comoving'.format(np.sqrt(distance2s.min())))
        return np.r_[np.nan, np.nan, np.nan]

    # ensure that np.average uses 64-bit internally for accuracy, but returns as input dtype
    return np.average(velocities[masks].astype(np.float64), 0, weights).astype(velocities.dtype)

# --------------------------------------------------------------------------------------------------
# rotation of position or velocity
# --------------------------------------------------------------------------------------------------
def get_coordinates_rotated(coordinate_vectors, rotation_tensor=None, rotation_angles=None):
    '''
    Get 3-D coordinate[s] (distance or velocity vector[s]) that are rotated by input rotation
    vectors or input rotation angles.
    If rotation_tensor, need to input vectors that are orthogonal.
    If rotation_angles, rotate by rotation_angles[0] about x-axis, then by rotation_angles[1] about
    y-axis, then by rotation_angles[2] about z-axis.

    Parameters
    ----------
    coordinate_vectors : array (object number x dimension number)
        coordinate[s] (distance[s] or velocity[s]) wrt a center of rotation
    rotation_tensor : array
        *orthogonal* rotation vectors (such as max, med, min eigen-vectors)
    rotation_angles : array
        rotation angles about x-axis, y-axis, z-axis [radians]

    Returns
    -------
    coordinate_vectors_rotated : array (object number x dimension number)
        coordinate[s] (distance[s] or velocity[s]) in rotated basis
    '''
    dtype = coordinate_vectors.dtype
    # sanity check in case input coordinate_vectors as int
    if dtype == np.int32 or dtype == np.int64:
        dtype = np.float64

    if rotation_tensor is not None:
        # sanity check - ensure input rotation vectors are orthogonal
        tolerance = 1e-6
        if (
            np.abs(np.dot(rotation_tensor[0], rotation_tensor[1])) > tolerance
            or np.abs(np.dot(rotation_tensor[0], rotation_tensor[2])) > tolerance
            or np.abs(np.dot(rotation_tensor[1], rotation_tensor[2])) > tolerance
        ):
            raise ValueError('input rotation_tensor is not orthogonal')

    elif rotation_angles is not None:
        m11 = np.cos(rotation_angles[1]) * np.cos(rotation_angles[2])
        m12 = np.cos(rotation_angles[0]) * np.sin(rotation_angles[2]) + np.sin(
            rotation_angles[0]
        ) * np.sin(rotation_angles[1]) * np.cos(rotation_angles[2])
        m13 = np.sin(rotation_angles[0]) * np.sin(rotation_angles[2]) - np.cos(
            rotation_angles[0]
        ) * np.sin(rotation_angles[1]) * np.cos(rotation_angles[2])
        m21 = -np.cos(rotation_angles[1]) * np.sin(rotation_angles[2])
        m22 = np.cos(rotation_angles[0]) * np.cos(rotation_angles[2]) - np.sin(
            rotation_angles[0]
        ) * np.sin(rotation_angles[1]) * np.sin(rotation_angles[2])
        m23 = np.sin(rotation_angles[0]) * np.cos(rotation_angles[2]) + np.cos(
            rotation_angles[0]
        ) * np.sin(rotation_angles[1]) * np.sin(rotation_angles[2])
        m31 = np.sin(rotation_angles[1])
        m32 = -np.sin(rotation_angles[0]) * np.cos(rotation_angles[1])
        m33 = np.cos(rotation_angles[0]) * np.cos(rotation_angles[1])

        rotation_tensor = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]], dtype=dtype)

    else:
        raise ValueError('need to input either rotation angles or rotation vectors')

    # have to do this way
    coordinate_vectors_rotated = np.asarray(
        np.dot(coordinate_vectors, rotation_tensor.transpose()), dtype=dtype
    )

    return coordinate_vectors_rotated

def get_velocities_in_coordinate_system(
    velocity_vectors, position_vectors, system_from='cartesian', system_to='cylindrical'
):
    '''
    Convert input 3-D velocity vectors from (cartesian, cylindrical, spherical) to:
        cartesian : x, y, z
        cylindrical : R (major axes), angle phi, Z (minor axis)
        spherical : r , angle theta, angle phi

    Parameters
    ----------
    velocity_vectors : array (object number x 3)
        velocity[s] wrt a center
    position_vectors : array (object number x 3)
        position[s]/distance[s] wrt a center

    Returns
    -------
    velocity_vectors_new : array (object number x 3)
        velocity[s] in new coordiante system
    '''
    assert system_from in ('cartesian', 'cylindrical', 'spherical')
    assert system_to in ('cartesian', 'cylindrical', 'spherical')

    if system_from == system_to:
        return velocity_vectors

    velocity_vectors = np.asarray(velocity_vectors)
    if np.ndim(velocity_vectors) == 1:
        velocity_vectors = np.asarray([velocity_vectors])

    position_vectors = np.asarray(position_vectors)
    if np.ndim(position_vectors) == 1:
        position_vectors = np.asarray([position_vectors])

    assert np.shape(velocity_vectors)[1] == 3 and np.shape(position_vectors)[1] == 3

    velocities_new = np.zeros(velocity_vectors.shape, dtype=velocity_vectors.dtype)

    if system_from == 'cartesian':
        # convert position vectors
        # R = {x,y}
        r = position_vectors[:, [0, 1]]
        r_norm = np.zeros(r.shape, position_vectors.dtype)
        # R_total = sqrt(x^2 + y^2)
        r_total = np.sqrt(np.sum(r**2, 1))
        masks = np.where(r_total > 0)[0]
        # need to do this way
        r_norm[masks] = np.transpose(r[masks].transpose() / r_total[masks])

        if system_to == 'cylindrical':
            # v_R = dot(v_{x,y}, R_norm)
            velocities_new[:, 0] = np.sum(velocity_vectors[:, [0, 1]] * r_norm, 1)
            # v_phi = cross(R_norm, v_{x,y})
            velocities_new[:, 1] = np.cross(r_norm, velocity_vectors[:, [0, 1]])
            # v_Z = v_z
            velocities_new[:, 2] = velocity_vectors[:, 2]
        elif system_to == 'spherical':
            # convert position vectors
            position_vectors_norm = np.zeros(position_vectors.shape, position_vectors.dtype)
            position_vectors_total = np.sqrt(np.sum(position_vectors**2, 1))
            masks = np.where(position_vectors_total > 0)[0]
            # need to do this way
            position_vectors_norm[masks] = np.transpose(
                position_vectors[masks].transpose() / position_vectors_total[masks]
            )

            # v_r = dot(v, r)
            velocities_new[:, 0] = np.sum(velocity_vectors * position_vectors_norm, 1)
            # v_theta
            a = np.transpose(
                [
                    r_norm[:, 0] * position_vectors_norm[:, 2],
                    r_norm[:, 1] * position_vectors_norm[:, 2],
                    -r_total / position_vectors_total,
                ]
            )
            velocities_new[:, 1] = np.sum(velocity_vectors * a, 1)
            # v_phi = cross(R_norm, v_{x,y})
            velocities_new[:, 2] = np.cross(r_norm, velocity_vectors[:, [0, 1]])

    elif system_from == 'cylindrical':
        raise ValueError(f'not yet support conversion from {system_from} to {system_to}')

    elif system_from == 'spherical':
        raise ValueError(f'not yet support conversion from {system_from} to {system_to}')

    # if only one velocity vector, return as 1-D array
    if len(velocities_new) == 1:
        velocities_new = velocities_new[0]

    return velocities_new

def get_principal_axes(position_vectors, velocity_vectors=None, masks=None, weights=None, use_moi=False, verbose=True):
    '''
    Compute principal axes of input position_vectors (which should be wrt a center),
    defined via the moment of inertia tensor.
    Get reverse-sorted rotation_tensor and axis ratios of these principal axes.

    Parameters
    ----------
    position_vectors : array (object number x dimension number)
        position[s] or distance[s] wrt a center
    velocity_vectors : array (object number x dimension number)
        velocity[s] wrt a center
    masks : array
        mask to apply to position_vectors and velocity_vectors
    weights : array
        weight for each position (usually mass) - if None, assume all have same weight
    use_moi : bool
        whether to use the moment of inertia tensor, instead of the second moment of the
        mass distribution, for the diagonal components input to get the rotation tensor
        this choice only affect the resultant axis ratios, not the resultant rotation tensor
    verbose : bool
        whether to print axis ratios

    Returns
    -------
    rotation_tensor : array
        max, med, min eigen-vectors that define the rotation tensor
    axis_ratios : array
        ratios of principal axes
    '''
    if weights is None or len(weights) == 0:
        weights = 1
    else:
        weights = weights / np.median(weights)

    if masks is not None:
        position_vectors = position_vectors[masks]
        if velocity_vectors is not None and len(velocity_vectors) > 0:
            velocity_vectors = velocity_vectors[masks]
        if weights is not None and len(weights) > 0:
            weights = weights[masks]

    if position_vectors.shape[1] == 3:
        # 3-D
        if use_moi:
            # use moment of inertia to define for diagonal terms
            xx = np.sum(weights * (position_vectors[:, 1] ** 2 + position_vectors[:, 2] ** 2))
            yy = np.sum(weights * (position_vectors[:, 0] ** 2 + position_vectors[:, 2] ** 2))
            zz = np.sum(weights * (position_vectors[:, 0] ** 2 + position_vectors[:, 1] ** 2))
            xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])
            xz = zx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 2])
            yz = zy = np.sum(weights * position_vectors[:, 1] * position_vectors[:, 2])

            moi_tensor = [[xx, -xy, -xz], [-yx, yy, -yz], [-zx, -zy, zz]]
        else:
            # default: use second moment of mass distribution for diagonal terms
            xx = np.sum(weights * position_vectors[:, 0] ** 2)
            yy = np.sum(weights * position_vectors[:, 1] ** 2)
            zz = np.sum(weights * position_vectors[:, 2] ** 2)
            xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])
            xz = zx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 2])
            yz = zy = np.sum(weights * position_vectors[:, 1] * position_vectors[:, 2])

            moi_tensor = [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]

    elif position_vectors.shape[1] == 2:
        # 2-D
        xx = np.sum(weights * position_vectors[:, 0] ** 2)
        yy = np.sum(weights * position_vectors[:, 1] ** 2)
        xy = yx = np.sum(weights * position_vectors[:, 0] * position_vectors[:, 1])

        moi_tensor = [[xx, xy], [yx, yy]]

    eigen_values, rotation_tensor = np.linalg.eig(moi_tensor)

    # order eigen-vectors by eigen-values, from largest to smallest
    eigen_indices_sorted = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[eigen_indices_sorted]
    # eigen_values /= eigen_values.max()  # renormalize to 1
    # make rotation_tensor[0, 1, 2] be eigen_vectors that correspond to eigen_values[0, 1, 2]
    rotation_tensor = rotation_tensor.transpose()[eigen_indices_sorted]
    # ensure that rotation tensor satisfies right-hand rule
    rotation_tensor[2] = np.cross(rotation_tensor[0], rotation_tensor[1])

    if position_vectors.shape[1] == 3:
        axis_ratios = np.sqrt(
            [
                eigen_values[2] / eigen_values[0],
                eigen_values[2] / eigen_values[1],
                eigen_values[1] / eigen_values[0],
            ]
        )

        if verbose:
            print(
                '* principal axes:  min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
                    axis_ratios[0], axis_ratios[1], axis_ratios[2]
                )
            )

    elif position_vectors.shape[1] == 2:
        axis_ratios = eigen_values[1] / eigen_values[0]

        if verbose:
            print('* principal axes:  min/maj = {:.3f}'.format(axis_ratios))

    if velocity_vectors is not None and len(velocity_vectors) > 0:
        # test if need to flip a principal axis to ensure that median v_phi > 0
        velocity_vectors_rot = get_coordinates_rotated(
            velocity_vectors, rotation_tensor
        )
        distance_vectors_rot = get_coordinates_rotated(
            position_vectors, rotation_tensor
        )
        velocity_vectors_cyl = get_velocities_in_coordinate_system(
            velocity_vectors_rot, distance_vectors_rot, 'cartesian', 'cylindrical'
        )

        # ensure that disk rotates in right-handed direction with median v_phi > 0
        if np.median(velocity_vectors_cyl[:, 1]) < 0:
            rotation_tensor[1] *= -1
            rotation_tensor[2] *= -1
    else:
        print('! warning: no center velocities input to get_principal_axes()')
        print('so direction/sign of minor axis rotation is ambiguous')

    return rotation_tensor, axis_ratios

def get_rot_mat_angular_momentum_vertical(position_vectors, velocity_vectors, masses, masks=None, rmax_ctr=10.0):
    '''
    Compute the rotation tensor that would maximize the angular momentum of the galaxy in the z-axis direction
    Inputs position_vectors, velocity_vectors, and masses are all centered at the galaxy center
    Inputs masks is a mask to apply to the position_vectors, velocity_vectors, and masses
    Inputs rmax_ctr is the radius within which to compute the angular momentum
    Returns the rotation tensor that would maximize the angular momentum of the galaxy in the z-axis direction
    '''
    if masks is not None:
        pos_s = position_vectors[masks]
        if velocity_vectors is not None and len(velocity_vectors) > 0:
            vel_s = velocity_vectors[masks]
        if masses is not None and len(masses) > 0:
            mass_s = masses[masks]

    r_s = np.sqrt(np.sum(pos_s**2, axis=1))
    # idx_sort = np.argsort(r_s)
    # r_sorted = r_s[idx_sort]
    # m_cum = np.cumsum(mass_s[idx_sort])
    # m_tot = m_cum[-1]
    # i_half = np.searchsorted(m_cum, 0.5 * m_tot)
    # r_half_star = r_sorted[i_half] if i_half < len(r_sorted) else r_sorted[-1]

    mask_10rh = r_s < (rmax_ctr)
    r_in = pos_s[mask_10rh]
    v_in = vel_s[mask_10rh]
    m_in = mass_s[mask_10rh]
    L = np.sum(m_in[:, np.newaxis] * np.cross(r_in, v_in), axis=0)
    L_norm = np.linalg.norm(L)
    if L_norm < 1e-30:
        # raise ValueError('negligible stellar angular momentum')
        # return the identity matrix as the rotation tensor
        return np.eye(3)
    L_hat = L / L_norm

    z_axis = np.array([0., 0., 1.], dtype=np.float64)
    angle = np.arccos(np.clip(np.dot(L_hat, z_axis), -1.0, 1.0))
    if np.sin(angle) < 1e-10:
        R = np.diag([1., 1., 1. if np.dot(L_hat, z_axis) > 0 else -1.])
    else:
        axis = np.cross(L_hat, z_axis)
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    return R