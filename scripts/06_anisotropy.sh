#!/bin/bash
#SBATCH --job-name=s2dm_anisotropy
#SBATCH --output=logs/anisotropy_%j.out
#SBATCH --error=logs/anisotropy_%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
# One galaxy at a time: a 6 MB draws npz promoted to 13 MB float64, then 20 annuli x 4 modes
# of small matmuls. Nothing like the whole-population load in 07_figures.sh, so 8 cores is
# plenty.
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#
# Azimuthal anisotropy metric (m=2 Fourier amplitude + major-axis orientation), for the
# fiducial model, its no-conditioning ablation, and both out-of-domain suites.
#
#   sbatch --export=ALL,TAG=v95 scripts/06_anisotropy.sh
#
# Writes small CSVs and npz to data/anisotropy_<tag>/ -- a few MB, not gigabytes. The figure
# stage reads only those, so restyling a panel costs seconds instead of re-reading 6.7 GB of
# draws. Set ABLATION_TAG="" to skip the v97 pass.
#
# Stage 1 runs FIRST and aborts the job on failure: no result should be produced by an
# operator that does not pass its own analytic checks.

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

cd "${STARS2DM_REPO}/src/paperfigs"
source "${STARS2DM_REPO}/scripts/env.sh"
set -u

TAG=${TAG:-v95}
ABLATION_TAG=${ABLATION_TAG-v97}
DRAWS_SUFFIX=${DRAWS_SUFFIX:-heun100_n100}
NUM_DRAWS=${NUM_DRAWS:-100}
# 820/102/102 split: the held-out set is [0, 102).
TEST_RANGE=${TEST_RANGE:-0:102}
FEATURED=${FEATURED:-19}
OUT_BASE="${STARS2DM_DATA_ROOT}"

echo "########## TAG=${TAG}  ABLATION_TAG=${ABLATION_TAG}  suffix=${DRAWS_SUFFIX}"
date

echo "########## 1/5 operator selftest"
python -u analyze_anisotropy.py selftest

echo "########## 2/5 crosscheck against the particle-based bar strength"
python -u analyze_anisotropy.py crosscheck --output-dir "${OUT_BASE}/anisotropy_${TAG}"

echo "########## 3/5 analyze ${TAG} (held-out DREAMS)"
python -u analyze_anisotropy.py analyze \
    --version "${TAG}" \
    --pred-root "${STARS2DM_PRED_ROOT}" \
    --num-draws "${NUM_DRAWS}" \
    --draws-suffix "${DRAWS_SUFFIX}" \
    --galaxy-indices "${TEST_RANGE}" \
    --per-galaxy-indices "${FEATURED}" \
    --output-dir "${OUT_BASE}/anisotropy_${TAG}"

echo "########## 4/5 analyze TNG50 out of domain"
# 280 prediction directories against 264 truth maps; the analyzer drops the excess and says so.
python -u analyze_anisotropy.py analyze \
    --version "TNG50_${TAG}_heun100_n100" \
    --pred-root "${STARS2DM_PRED_ROOT}" \
    --num-draws 100 \
    --draws-suffix "" \
    --draws-pattern "*_TNG50_${TAG}_heun100_n100_pred_draws_100.npz" \
    --galaxy-indices 0:264 \
    --per-galaxy-indices "" \
    --output-dir "${OUT_BASE}/anisotropy_TNG50_${TAG}"

echo "########## 5/5 analyze FIREm12 out of domain"
# The FIRE ensemble is 100 flat-prior parameter configurations x 1 realization, NOT 100
# stochastic draws at fixed conditioning. Its spread therefore carries the parameter
# marginalization, which inflates the denominator of Delta^{A2} and biases it toward zero.
# The alignment statistic is normalized and is unaffected; for the amplitude, compare
# A2_pred_median / A2_true_median from the population CSV instead.
python -u analyze_anisotropy.py analyze \
    --version "FIREm12_${TAG}" \
    --pred-root "${STARS2DM_PRED_ROOT}" \
    --num-draws 100 \
    --draws-suffix "" \
    --draws-pattern "*_FIREm12_${TAG}_pred_draws_100_heun100_FIREm12_flat_prior_baryon_100.npz" \
    --galaxy-indices 0:8 \
    --per-galaxy-indices "" \
    --output-dir "${OUT_BASE}/anisotropy_FIREm12_${TAG}"

if [ -n "${ABLATION_TAG}" ]; then
    echo "########## analyze ${ABLATION_TAG} (no-conditioning ablation)"
    python -u analyze_anisotropy.py analyze \
        --version "${ABLATION_TAG}" \
        --pred-root "${STARS2DM_PRED_ROOT}" \
        --num-draws "${NUM_DRAWS}" \
        --draws-suffix "${DRAWS_SUFFIX}" \
        --galaxy-indices "${TEST_RANGE}" \
        --per-galaxy-indices "${FEATURED}" \
        --output-dir "${OUT_BASE}/anisotropy_${ABLATION_TAG}"
fi
echo "########## done"
date
