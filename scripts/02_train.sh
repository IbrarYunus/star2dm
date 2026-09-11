#!/bin/bash
#SBATCH --job-name=s2dm_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:2
#SBATCH --time=12:00:00
#
# Train one of the two paper models from scratch.
#
#   sbatch --export=ALL,MODEL=v95 scripts/02_train.sh   # fiducial, theta-conditioned
#   sbatch --export=ALL,MODEL=v97 scripts/02_train.sh   # Appendix D, no-theta ablation
#
# 12 h on 2x v100-16 reaches the configured max_steps=10000; both published runs stopped at
# step 9054 with val_loss 0.065 (v95) and 0.064 (v97). The ablation gets the SAME budget as
# the fiducial deliberately, so any performance gap is attributable to removing conditioning
# rather than to undertraining.
#
# Checkpoints land in ./checkpoints/<MODEL>/ (config `dirpath`). Each model writes to its own
# directory: two full-budget jobs sharing a dirpath collide on the save_last=True filename and
# one run's final checkpoint is silently lost.
#
# To use the trained checkpoint for inference, copy it to where the rest of the pipeline
# looks:  cp checkpoints/v95_fiducial/last.ckpt "$STARS2DM_CKPT_ROOT/last-v95.ckpt"

set -eo pipefail

# Locate the repository. Slurm copies the batch script to a spool directory before running
# it, so "$0" is /var/spool/slurm/.../slurm_script and dirname "$0" finds nothing. Prefer an
# explicit STARS2DM_REPO, then the directory sbatch was called from, then the script's own
# location for a direct (non-Slurm) run -- and check the answer before trusting it.
for candidate in "${STARS2DM_REPO}" "${SLURM_SUBMIT_DIR}" "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; do
    if [ -n "${candidate}" ] && [ -d "${candidate}/src/paperfigs" ]; then
        STARS2DM_REPO="${candidate}"
        break
    fi
done
if [ ! -d "${STARS2DM_REPO}/src/paperfigs" ]; then
    echo "ERROR: cannot locate the stars2dm repository." >&2
    echo "       Submit from the repository root, or set STARS2DM_REPO explicitly:" >&2
    echo "       sbatch --export=ALL,STARS2DM_REPO=/path/to/stars2dm,... $0" >&2
    exit 1
fi
export STARS2DM_REPO

cd "${STARS2DM_REPO}"
source "${STARS2DM_REPO}/scripts/env.sh"
set -u

MODEL=${MODEL:?set MODEL=v95 or MODEL=v97}
case "${MODEL}" in
    v95) CONFIG=configs/v95_fiducial.yaml ;;
    v97) CONFIG=configs/v97_no_theta.yaml ;;
    *)   echo "ERROR: MODEL must be v95 or v97, got '${MODEL}'" >&2; exit 1 ;;
esac
require_file "${CONFIG}"
require_file "${STARS2DM_DREAMS_ROOT}/2D_maps/CDM/MW_zooms/SB5/Maps_rhostar_TNG_SB5_128_z=0.00_csv_fov_z0p350_0p350.npy"

scrub_slurm_ntasks

echo "########## training ${MODEL} from ${CONFIG}"
date
python -m baryonbridge.train fit \
    --config "${CONFIG}" \
    --data.init_args.root "${STARS2DM_DREAMS_ROOT}/"
echo "########## done"
date
