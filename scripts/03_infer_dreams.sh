#!/bin/bash
#SBATCH --job-name=s2dm_infer_dreams
#SBATCH --output=logs/infer_dreams_%j.out
#SBATCH --error=logs/infer_dreams_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --time=08:00:00
#
# In-domain inference over the 1024 DREAMS galaxies. One shard per job.
#
#   sbatch --export=ALL,TAG=v95,START_IDX=0,END_IDX=102  scripts/03_infer_dreams.sh
#   bash scripts/submit_all.sh                            # every shard, both models
#
# Why 100 draws. The profile-level calibration factor k = sigma(Delta) and the per-pixel
# model uncertainty are built from an ensemble std with ddof=0, which is biased low by
# sqrt((n-1)/n)*c4(n) = 0.923 at n=10 and inflates every Delta derived from it. The same
# estimator on a PERFECTLY calibrated Gaussian ensemble returns k = 1.27 at n=10 against
# 1.03 at n=100, so a 10-draw number carries an estimator artifact the paper would have to
# explain. At n=100 the measured values are the genuine miscalibration. Over the held-out
# split, k falls from 1.48 -> 2.13 (n=10) to 1.28 -> 1.93 (n=100) across r = 3.5 -> 24.7 kpc.
#
# Cost, measured twice independently: 1.17 s per galaxy-draw on one v100-16 at heun-100, so
# a 115-galaxy shard is ~3.7 h -- inside the 8 h request. Output is ~5.8 MB per galaxy.
#
# NDRAWS=10 additionally produces the 10-draw set that the stamp-grid figures (Appendix A)
# and the v97 DREAMS panels (Appendix D) read.

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

TAG=${TAG:-v95}
NDRAWS=${NDRAWS:-100}
STEPS=${STEPS:-100}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-1024}
CKPT=$(require_ckpt "${TAG}")

# The suffix is part of every output filename and is what the figure scripts glob on, so it
# must match what the analysis expects: heun100 for the 10-draw set, heun100_n100 for 100.
if [ "${NDRAWS}" = "100" ]; then
    SUFFIX="heun${STEPS}_n${NDRAWS}"
else
    SUFFIX="heun${STEPS}"
fi

echo "########## DREAMS ${TAG}: galaxies [${START_IDX}, ${END_IDX})  draws=${NDRAWS}  heun-${STEPS}"
date
python -u scripts/infer.py \
    --checkpoint-path "${CKPT}" \
    --output-dir "${STARS2DM_PRED_ROOT}/" \
    --version "${TAG}" \
    --start-idx "${START_IDX}" --end-idx "${END_IDX}" \
    --save-full-trajectory \
    --no-plots \
    --x-conditioning-fields rhostar --x1-fields rhoDM \
    --xmin -32 --xmax 32 --ymin -32 --ymax 32 \
    --num-samples "${NDRAWS}" \
    --sampler heun \
    --n-sampling-steps "${STEPS}" \
    --output-suffix "${SUFFIX}"
echo "########## done"
date
