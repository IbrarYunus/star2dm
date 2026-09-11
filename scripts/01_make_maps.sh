#!/bin/bash
#SBATCH --job-name=s2dm_maps
#SBATCH --output=logs/maps_%j.out
#SBATCH --error=logs/maps_%j.err
#SBATCH --partition=RM
#SBATCH --time=06:00:00
#
# Build the paired stellar/dark-matter density maps from raw simulation snapshots.
#
#   sbatch --export=ALL,SUITE=DREAMS  scripts/01_make_maps.sh
#   sbatch --export=ALL,SUITE=TNG50   scripts/01_make_maps.sh
#   sbatch --export=ALL,SUITE=FIREm12 scripts/01_make_maps.sh
#
# MOST READERS DO NOT NEED THIS. It requires the raw DREAMS / TNG50 / FIRE snapshots and
# group catalogs -- terabytes, not redistributable here -- plus a different environment from
# the rest of the pipeline (environment-mapgen.yml: astropy and agama, which the training
# environment does not carry). The finished maps are 340 MB and can be downloaded instead:
#
#   python scripts/fetch_data.py --tier maps
#
# What it produces. For each galaxy, the stellar and dark-matter density fields are estimated
# at the mid-plane of the galaxy -- a slice, NOT a line-of-sight integral -- on a 128x128 grid
# in the face-on frame (stellar angular momentum within 10 kpc aligned to the line of sight).
# Density at each grid point comes from a KD-tree adaptive kernel estimate: smoothing length
# set by the 128th nearest neighbour, masses weighted by a cubic spline. Maps are stored as
# log10 density and normalized by the data loader.
#
# The field of view is NOT a flag here -- it is read per galaxy from
# src/mapgen/redshift_sample_0p35.csv, which holds physical_fov_kpc = 65.192930 for all 1024
# rows, i.e. +/-32.596465 kpc. That is deliberately not the +/-32 kpc the analysis code uses
# for its radial axis; see docs/CAVEATS.md before changing either.
#
# Paths to the raw snapshots are set at the top of src/mapgen/get_2D_maps.py and point at the
# cluster where the published maps were built. Edit them for your own layout.

set -euo pipefail
cd "$(dirname "$0")/../src/mapgen"
source ../../scripts/env.sh

SUITE=${SUITE:?set SUITE=DREAMS|TNG50|FIREm12}
RES=${RES:-128}
FOV_CSV=${FOV_CSV:-redshift_sample_0p35.csv}
require_file "${FOV_CSV}"

case "${SUITE}" in
    DREAMS)  SNAP=90;  BOX_END=1023 ;;   # snapshot 90 is z=0 for the DREAMS MW zooms
    TNG50)   SNAP=99;  BOX_END=263  ;;   # snapshot 99 is z=0 for TNG
    FIREm12) SNAP=600; BOX_END=7    ;;   # FIRE m12 z=0 snapshot; 8 galaxies
    *) echo "ERROR: SUITE must be DREAMS, TNG50 or FIREm12, got '${SUITE}'" >&2; exit 1 ;;
esac

for MAP_TYPE in rhostar rhoDM; do
    echo "########## ${SUITE} ${MAP_TYPE} at ${RES}px, FOV from ${FOV_CSV}"
    date
    python -u get_2D_maps.py \
        --box-ini 0 --box-end "${BOX_END}" \
        --redshift 0 \
        --redshift-csv "${FOV_CSV}" \
        --map-type "${MAP_TYPE}" \
        --resolution "${RES}" \
        --sim-name "${SUITE}" \
        --snap "${SNAP}" \
        --get-maps --save-map
done

# get_2D_maps.py writes per-box files under 2D_maps/.../temp/ and then combines them itself
# into the single Maps_<field>_<suite>_128_z=0.00_csv_fov_z0p350_0p350.npy the data loaders
# read. Row order in that stack IS the galaxy index used everywhere downstream, so a partial
# run that is resumed with a different --box-ini will produce a stack whose indices no longer
# match the parameter table. Run the full box range in one pass.
#
# src/mapgen/stack_npy_list.py is a separate utility for concatenating finished stacks from
# different suites (python stack_npy_list.py --list files.txt --output combined.npy); it is
# not part of this chain.
echo "########## done"
date
