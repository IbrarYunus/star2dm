#!/bin/bash
#SBATCH --job-name=s2dm_sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --time=24:00:00
#
# Simulation-parameter sweeps -- Appendix C. Five independent jobs, one per parameter.
#
#   for S in om s8 sn1 sn2 bhff; do
#       sbatch --export=ALL,TAG=v95,SWEEP=${S}_sweep_5 scripts/05_infer_sweeps.sh
#   done
#
# Each grid has 5 points; 10 draws per point gives one 50-draw file per galaxy
# (galaxy_N_v95_pred_draws_50_heun100_<sweep>.npz), 1024 galaxies, ~16.5 h per sweep.
#
# These stay at 50 draws while everything else moved to 100. Converting them would cost
# ~165 GPU-h, and the ddof=0 ensemble-std bias at n=50 is 0.986, so the residual MEANS these
# figures report move by under 2%. The main-text calibration numbers, which are the ones
# sensitive to that bias, come from the n=100 runs instead.
#
# Only grid points 0 and 4 (the extremes) appear in the paper -- the figure stage reads
# draws 0:10 and 40:50 out of each file. The middle three are computed anyway because the
# sweep driver evaluates the whole CSV in one pass.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

TAG=${TAG:-v95}
SWEEP=${SWEEP:?set SWEEP=om_sweep_5|s8_sweep_5|sn1_sweep_5|sn2_sweep_5|bhff_sweep_5}
STEPS=${STEPS:-100}
GAL_START=${GAL_START:-0}
GAL_END=${GAL_END:-1024}

CKPT=$(require_ckpt "${TAG}")
CONFIG_CSV="param_sweep_configs/${SWEEP}.csv"
require_file "${CONFIG_CSV}"

echo "########## sweep ${SWEEP}, ${TAG}, galaxies [${GAL_START}, ${GAL_END})"
date
python -u scripts/infer_sweep.py \
    --checkpoint-path "${CKPT}" \
    --param-config-csv "${CONFIG_CSV}" \
    --output-dir "${STARS2DM_PRED_ROOT}/" \
    --output-suffix "heun${STEPS}_${SWEEP}" \
    --start-idx "${GAL_START}" --end-idx "${GAL_END}" \
    --version "${TAG}" \
    --x-conditioning-fields rhostar --x1-fields rhoDM \
    --num-samples 10 \
    --overwrite-draws \
    --no-plots \
    --sampler heun \
    --n-sampling-steps "${STEPS}"
echo "########## done"
date
