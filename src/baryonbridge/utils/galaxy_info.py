"""
Utility functions for tracking and identifying galaxies in the DREAMS dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Which parameter table describes which suite.
#
# Without this, a dwarf run reaches load_dreams_parameters with the DREAMS defaults,
# reads the 1024-row MW table, finds galaxy id 0-395 comfortably in range, and prints
# Milky Way TNG parameters for a FIRE dwarf. Nothing raises -- the ids simply overlap.
PARAM_TABLES = {
    "DREAMS": {
        "path": "Parameters/CDM/MW_zooms/CDM_{suite}_MW_{dataset}.txt",
        "columns": ["Om", "s8", "SN1", "SN2", "BHFF"],
        "sep": " ",
    },
    # Five FIRE feedback parameters, raw (unlogged) values. The loader conditions on
    # the log10-transformed table instead; these are for human-readable printing.
    "DWARFFIRE": {
        "path": "Parameters/CDM/dwarfs/CDM_FIRE_Dwarfs_SB5_n396.txt",
        "columns": [
            "SNeIIEnergyFrac",
            "WindMomentumLoading",
            "CritPhysDensity",
            "zreion",
            "AlphaCrit",
        ],
        "sep": r"\s+",
    },
}


def load_parameters(root: str = "./data/DREAMS_data",
                    suite_key: str = "DREAMS",
                    suite: str = "TNG",
                    dataset: str = "SB5") -> pd.DataFrame:
    """
    Load the conditioning parameters for one suite, indexed by galaxy id.

    Args:
        root: Root path to the DREAMS data
        suite_key: Key into PARAM_TABLES ('DREAMS' or 'DWARFFIRE')
        suite: Simulation suite, substituted into the DREAMS path template
        dataset: Dataset name, substituted into the DREAMS path template
    """
    if suite_key not in PARAM_TABLES:
        raise ValueError(
            f"Unknown suite_key {suite_key!r}. Available: {sorted(PARAM_TABLES)}"
        )
    spec = PARAM_TABLES[suite_key]
    params_path = Path(root) / spec["path"].format(suite=suite, dataset=dataset)
    params = pd.read_csv(
        params_path,
        sep=spec["sep"],
        names=spec["columns"],
        comment="#",
    )
    params.index.name = "galaxy_id"
    return params


def load_dreams_parameters(root: str = "./data/DREAMS_data",
                           suite: str = 'TNG',
                           dataset: str = 'SB5') -> pd.DataFrame:
    """
    Load the cosmological parameters for the DREAMS MW-zoom dataset.

    Kept as the original entry point; new callers should use load_parameters with an
    explicit suite_key.
    """
    return load_parameters(root=root, suite_key="DREAMS", suite=suite, dataset=dataset)


def get_galaxy_info(galaxy_id: int,
                    root: str = "./data/DREAMS_data",
                    suite: str = 'TNG',
                    dataset: str = 'SB5',
                    suite_key: str = "DREAMS") -> Dict:
    """
    Get information about a specific galaxy.

    Args:
        galaxy_id: The galaxy ID
        root: Root path to the DREAMS data
        suite: Simulation suite
        dataset: Dataset name
        suite_key: Which parameter table to read (see PARAM_TABLES)

    Returns:
        Dictionary with galaxy information
    """
    params_df = load_parameters(root, suite_key, suite, dataset)

    if galaxy_id >= len(params_df):
        raise ValueError(
            f"Galaxy ID {galaxy_id} is out of range for suite {suite_key}. "
            f"Max ID: {len(params_df) - 1}"
        )

    galaxy_params = params_df.iloc[galaxy_id]

    return {
        'galaxy_id': galaxy_id,
        'dataset': dataset,
        'suite': suite,
        'suite_key': suite_key,
        'cosmological_parameters': galaxy_params.to_dict(),
        # Built from the table's own columns rather than a hardcoded five, so a suite
        # with different parameter names reports its own.
        'parameter_summary': {
            name: galaxy_params[name] for name in PARAM_TABLES[suite_key]["columns"]
        },
    }


def print_galaxy_summary(galaxy_id: int,
                         root: str = "./data/DREAMS_data",
                         suite: str = 'TNG',
                         dataset: str = 'SB5',
                         suite_key: str = "DREAMS"):
    """
    Print a summary of galaxy information.

    Args:
        galaxy_id: The galaxy ID
        root: Root path to the DREAMS data
        suite: Simulation suite
        dataset: Dataset name
        suite_key: Which parameter table to read (see PARAM_TABLES)
    """
    try:
        info = get_galaxy_info(galaxy_id, root, suite, dataset, suite_key)

        print(f"=== Galaxy {galaxy_id} Summary ===")
        print(f"Suite table: {info['suite_key']}")
        print(f"Dataset: {info['dataset']}")
        print("\nParameters:")
        for param, value in info['parameter_summary'].items():
            print(f"  {param}: {value:.4f}")

    except Exception as e:
        print(f"Error getting galaxy info: {e}")


def get_validation_galaxy_ids(n_test: int = 20, n_val: int = 20) -> List[int]:
    """
    Get the galaxy IDs that are used in the validation set.
    
    Args:
        n_test: Number of test samples
        n_val: Number of validation samples
        
    Returns:
        List of galaxy IDs in the validation set
    """
    val_start = n_test
    val_end = n_test + n_val
    return list(range(val_start, val_end))

def get_training_galaxy_ids(n_test: int = 20, n_val: int = 20, n_train: int = 216) -> List[int]:
    """
    Get the galaxy IDs that are used in the training set.
    
    Args:
        n_test: Number of test samples
        n_val: Number of validation samples
        n_train: Number of training samples
        
    Returns:
        List of galaxy IDs in the training set
    """
    train_start = n_test + n_val
    train_end = train_start + n_train
    return list(range(train_start, train_end)) 