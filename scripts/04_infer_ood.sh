#!/bin/bash
#SBATCH --job-name=s2dm_infer_ood
#SBATCH --output=logs/infer_ood_%j.out
#SBATCH --error=logs/infer_ood_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --time=08:00:00
#
# Out-of-domain inference on TNG50 and FIRE m12.
#
#   sbatch --export=ALL,TAG=v95,SUITE=TNG50,START_IDX=0,END_IDX=132 scripts/04_infer_ood.sh
#   sbatch --export=ALL,TAG=v95,SUITE=FIREm12                       scripts/04_infer_ood.sh
#
# The two suites are NOT run the same way, and the difference matters when reading the
# figures:
#
#   TNG50    theta is well defined and is not an arbitrary choice -- the loader reads
#            TNG's own cosmology from its parameter file, and the feedback parameters are
#            the TNG fiducial values the DREAMS grid is centred on. So TNG50 sits on exactly
#            the same footing as DREAMS: 100 stochastic draws at fixed theta.
#
#   FIREm12  has no counterpart to the DREAMS SN1/SN2/BHFF parametrization, so its ensemble
#            marginalizes over those under a flat (log-uniform) prior: 100 configurations x
#            1 realization, with Om/s8 left at FIRE's true values. Its spread therefore
#            contains parameter ignorance that DREAMS' and TNG50's do not. The DREAMS-derived
#            calibration factor is an UPPER BOUND on the correction FIRE needs, not the right
#            divisor for it, and Delta^{A2} built on this ensemble is biased toward zero.

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

TAG=${TAG:-v95}
SUITE=${SUITE:?set SUITE=TNG50 or SUITE=FIREm12}
STEPS=${STEPS:-100}
CKPT=$(require_ckpt "${TAG}")

echo "########## ${SUITE} OOD, ${TAG}, heun-${STEPS}"
date

if [ "${SUITE}" = "TNG50" ]; then
    NDRAWS=${NDRAWS:-100}
    START_IDX=${START_IDX:-0}
    END_IDX=${END_IDX:-264}
    require_file "${STARS2DM_TNG_ROOT}/2D_maps/TNG50/Maps_rhostar_TNG50_128_z=0.00_csv_fov_z0p350_0p350.npy"
    python -u scripts/infer_ood.py \
        --checkpoint-path "${CKPT}" \
        --output-dir "${STARS2DM_PRED_ROOT}/" \
        --start-idx "${START_IDX}" --end-idx "${END_IDX}" \
        --version "${TAG}" \
        --x-conditioning-fields rhostar --x1-fields rhoDM \
        --xmin -32 --xmax 32 --ymin -32 --ymax 32 \
        --save-full-trajectory \
        --data-source TNG50 \
        --data-root "${STARS2DM_TNG_ROOT}" \
        --no-plots \
        --num-samples "${NDRAWS}" \
        --sampler heun \
        --n-sampling-steps "${STEPS}" \
        --output-suffix "heun${STEPS}_n${NDRAWS}"

elif [ "${SUITE}" = "FIREm12" ]; then
    CONFIG_CSV=param_sweep_configs/FIREm12_flat_prior_baryon_100.csv
    OUT_SUFFIX=FIREm12_flat_prior_baryon_100
    require_file "${STARS2DM_FIRE_ROOT}/2D_maps/FIREm12/Maps_rhostar_FIREm12_128_z=0.00_csv_fov_z0p350_0p350.npy"
    # The grid ships with the repository so the published draws are reproducible; regenerate
    # it only if it is missing, and note that a different seed gives a different ensemble.
    if [ ! -f "${CONFIG_CSV}" ]; then
        echo "WARNING: regenerating ${CONFIG_CSV}; the published runs used the shipped file" >&2
        python scripts/make_param_csv.py \
            --output "${CONFIG_CSV}" \
            --flat-prior-samples 100 \
            --flat-prior-params SN1 SN2 BHFF \
            --seed 42
    fi
    python -u scripts/infer_sweep.py \
        --checkpoint-path "${CKPT}" \
        --param-config-csv "${CONFIG_CSV}" \
        --output-dir "${STARS2DM_PRED_ROOT}/" \
        --output-suffix "heun${STEPS}_${OUT_SUFFIX}" \
        --data-source FIREm12 \
        --data-root "${STARS2DM_FIRE_ROOT}" \
        --start-idx 0 --end-idx 8 \
        --version "${TAG}" \
        --x-conditioning-fields rhostar --x1-fields rhoDM \
        --num-samples 1 \
        --overwrite-draws \
        --no-plots \
        --sampler heun \
        --n-sampling-steps "${STEPS}"
else
    echo "ERROR: SUITE must be TNG50 or FIREm12, got '${SUITE}'" >&2
    exit 1
fi
echo "########## done"
date
