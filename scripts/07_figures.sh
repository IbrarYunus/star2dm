#!/bin/bash
#SBATCH --job-name=s2dm_figures
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
# 32 cores = ~60 GB. STAGE=dreams loads 1024 x 100 x 128^2, which is 6.7 GB as float32
# before any float64 promotion; an 8-core request was OOM-killed doing exactly this. Keep 32
# for dreams and sweeps.
#SBATCH --ntasks-per-node=32
#SBATCH --time=08:00:00
#
# Regenerate the paper's figures as PDF. Five independent stages, submitted separately so
# they run in parallel and a failure is isolated:
#
#   sbatch --export=ALL,STAGE=dreams  scripts/07_figures.sh   # Figs 1-4a, App B(a)
#   sbatch --export=ALL,STAGE=ood     scripts/07_figures.sh   # Fig 4b, App B(b)
#   sbatch --export=ALL,STAGE=support scripts/07_figures.sh   # App A, Figs 5-6, App D
#   sbatch --export=ALL,STAGE=sweeps  scripts/07_figures.sh   # App C
#   sbatch --export=ALL,STAGE=aniso   scripts/07_figures.sh   # anisotropy panels (seconds)
#
# Output goes to $STARS2DM_FIG_DIR (default data/figures/), with a PNG preview of each figure
# alongside. Compare against reference_figures/ to confirm a reproduction.
#
# The fiducial prediction is combined over draws with the MEAN, not the median, and annulus
# aggregation is also the mean. Both matter: the profile-level Delta needs the same
# convention in numerator and denominator, and Eq. 1 already defines sigma_pred with the
# mean. Changing either moves published numbers -- see src/paperfigs/pred_data_loaders.py.

set -euo pipefail
cd "$(dirname "$0")/../src/paperfigs"
source ../../scripts/env.sh

STAGE=${STAGE:?set STAGE=dreams|ood|support|sweeps|aniso}
PAPER_GRID=${PAPER_GRID:-1}
export PAPER_GRID
TAG=${TAG:-v95}
ABLATION_TAG=${ABLATION_TAG:-v97}
FIG="${STARS2DM_FIG_DIR}"
PRED="${STARS2DM_PRED_ROOT}"

DREAMS_N100="*_${TAG}_pred_draws_100_heun100_n100.npz"
TNG_N100="*TNG50_${TAG}_heun100_n100_pred_draws_100.npz"
FIRE_N100="*_FIREm12_${TAG}_pred_draws_100_heun100_FIREm12_flat_prior_baryon_100.npz"

echo "########## STAGE=${STAGE}  PAPER_GRID=${PAPER_GRID}  ->  ${FIG}"
date

# =========================================================================================
if [ "${STAGE}" = "dreams" ]; then

# How far the mean/mean convention moves the numbers the Results section quotes. Runs first
# so the answer is in the log before any prose is rewritten. It draws nothing, so the no-grid
# pass skips it rather than spending ~20 min reproducing the same table.
if [ "${PAPER_GRID}" = "1" ]; then
    echo "########## combiner comparison (mean vs median), DREAMS held-out"
    python -u compare_combiners.py
fi

echo "########## DREAMS: all 1024, the 102 held-out highlighted -- Figs 1, 2, 3, 4a, App B(a)"
# All three DREAMS-all renderings come out of one data load; re-running the script per
# variant would re-read 6.7 GB of draws for a figure that takes seconds to draw.
python -u plot_radial_profile_figures.py \
    --version "${TAG}_heun100_n100" \
    --pred-root "${PRED}" \
    --data-layout per_galaxy \
    --output-dir "${FIG}" \
    --num-draws 100 \
    --draws-pattern "${DREAMS_N100}" \
    --highlight-test-galaxies \
    --test-galaxy-range 0:102 \
    --dreams-all-style points,contours,hide

fi

# =========================================================================================
if [ "${STAGE}" = "ood" ]; then

echo "########## TNG50 with the FIREm12 flat-prior overlay -- Fig 4b, App B(b)"
python -u plot_radial_profile_figures.py \
    --version "TNG50_${TAG}_heun100_n100" \
    --version2 "FIREm12_${TAG}" \
    --pred-root "${PRED}" \
    --data-layout per_galaxy \
    --output-dir "${FIG}" \
    --num-draws 100 \
    --draws-pattern "${TNG_N100}" \
    --num-draws-2 100 \
    --draws-pattern-2 "${FIRE_N100}" \
    --draws-output-label-2 heun100_flat_prior_baryon

fi

# =========================================================================================
if [ "${STAGE}" = "support" ]; then

echo "########## stamp grids, the 8 residual/map panel types -- Appendix A"
# --panel-type residual, not all: 'all' also covers the anisotropy panels, and those must run
# on the 100-draw ensemble (next block), not this 10-draw one.
python -u plot_paper_stamp_figures.py \
    --version "${TAG}" \
    --panel-type residual \
    --n-galaxies 16 \
    --seed 42 \
    --num-draws 10 \
    --draws-pattern "*_${TAG}_pred_draws_10_heun100.npz" \
    --output-dir "${FIG}"

echo "########## stamp grids, anisotropy panels (100 draws) -- Appendix A"
# Same 16 galaxies as the block above: the sampling pool is the test split [0, 102), every
# member of which has both a 10-draw and a 100-draw file, so seed 42 draws the same set.
# The 100-draw ensemble is required here -- Delta^{A2}'s denominator is an ensemble std and
# the 16-84th percentile band is meaningless from ten samples. The version tag carries the
# ensemble size into the filenames so these cannot be confused with the 10-draw set above.
python -u plot_paper_stamp_figures.py \
    --version "${TAG}_heun100_n100" \
    --panel-type anisotropy \
    --n-galaxies 16 \
    --seed 42 \
    --num-draws 100 \
    --draws-pattern "${DREAMS_N100}" \
    --output-dir "${FIG}"

echo "########## calibration + e_w sweep -- Figs 5 and 6"
python -u plot_calibration_figures.py --outdir "${FIG}"

echo "########## ${ABLATION_TAG} ablation: DREAMS held-out -- App D1, D2"
python -u plot_radial_profile_figures.py \
    --version "${ABLATION_TAG}_heun100" \
    --pred-root "${PRED}" \
    --data-layout per_galaxy \
    --output-dir "${FIG}" \
    --num-draws 10 \
    --draws-pattern "*_${ABLATION_TAG}_pred_draws_10_heun100.npz" \
    --highlight-test-galaxies \
    --test-galaxy-range 0:102 \
    --dreams-all-style points,contours,hide

# TNG50 v97 exists only at 10 draws; FIRE v97 has the 100-draw set. The mix is deliberate and
# is stated in the caption: a ddof=0 ensemble std is biased low at n=10, so TNG50's Delta
# denominators run ~5% small relative to FIRE's in the same panel.
echo "########## ${ABLATION_TAG} ablation: TNG50 (n=10) with FIREm12 (n=100) -- App D3, D4"
python -u plot_radial_profile_figures.py \
    --version "TNG50_${ABLATION_TAG}_heun100" \
    --version2 "FIREm12_${ABLATION_TAG}_heun100_n100" \
    --pred-root "${PRED}" \
    --data-layout per_galaxy \
    --output-dir "${FIG}" \
    --num-draws 10 \
    --draws-pattern "*TNG50_${ABLATION_TAG}_heun100_pred_draws_10.npz" \
    --num-draws-2 100 \
    --draws-pattern-2 "*FIREm12_${ABLATION_TAG}_heun100_n100_pred_draws_100.npz" \
    --draws-output-label-2 FIRE

fi

# =========================================================================================
if [ "${STAGE}" = "sweeps" ]; then

# Appendix C shows the minimum and maximum grid point of each parameter, i.e. config 0 and
# config 4 of the 5-point sweep, sliced out of the config-major 50-draw file.
# The histogram x-range is per-parameter: at the default -5 5 the kappa_w (SN2) sweep leaves
# ~38% of its distribution off-plot, and the caption quotes a mean that would then sit on the
# axis edge.
for SWEEP in om_sweep_5 s8_sweep_5 sn1_sweep_5 sn2_sweep_5 bhff_sweep_5; do
    CONFIG_CSV="${STARS2DM_REPO}/param_sweep_configs/${SWEEP}.csv"
    if [ ! -f "${CONFIG_CSV}" ]; then
        echo "ERROR: sweep grid ${CONFIG_CSV} not found" >&2
        exit 1
    fi
    case "${SWEEP}" in
        sn1_sweep_5|sn2_sweep_5) HIST_XLIM="-12 18" ;;
        *)                       HIST_XLIM="-8 8"   ;;
    esac
    mapfile -t LABELS < <(tail -n +2 "${CONFIG_CSV}" | cut -d',' -f2 \
        | tr '[:upper:]' '[:lower:]' | sed 's/=/_/; s/\./p/')
    for i in 0 4; do
        echo "########## sweep ${SWEEP} config ${i} (${LABELS[$i]}), xlim ${HIST_XLIM}"
        python -u plot_radial_profile_figures.py \
            --version "${TAG}_heun100" \
            --pred-root "${PRED}" \
            --data-layout per_galaxy \
            --output-dir "${FIG}" \
            --galaxy-indices 0:1024 \
            --num-draws 50 \
            --draws-pattern "*_${TAG}_pred_draws_50_heun100_${SWEEP}.npz" \
            --draws-slice $((i * 10)):$(((i + 1) * 10)) \
            --deviation-hist-xlim ${HIST_XLIM} \
            --draws-output-label "${LABELS[$i]}"
    done
done

fi

# =========================================================================================
if [ "${STAGE}" = "aniso" ]; then

# Reads only the CSVs and small npz that 06_anisotropy.sh wrote, so this stage is seconds,
# not hours -- run that job first if data/anisotropy_<tag>/ does not exist.
echo "########## anisotropy: galaxy-19 panels, population panels, ${TAG}-vs-${ABLATION_TAG} histogram"
python -u plot_anisotropy_figures.py \
    --data-dir "${STARS2DM_DATA_ROOT}/anisotropy_${TAG}" \
    --version "${TAG}_heun100_n100" \
    --suite "TNG50:${STARS2DM_DATA_ROOT}/anisotropy_TNG50_${TAG}:TNG50_${TAG}_heun100_n100" \
    --suite "FIRE:${STARS2DM_DATA_ROOT}/anisotropy_FIREm12_${TAG}:FIREm12_${TAG}" \
    --data-dir-2 "${STARS2DM_DATA_ROOT}/anisotropy_${ABLATION_TAG}" \
    --version-2 "${ABLATION_TAG}_heun100_n100" \
    --label "With conditioning (fiducial)" \
    --label-2 "Without conditioning" \
    --galaxy 19 \
    --output-dir "${FIG}"

fi

echo "########## done"
date
ls -l "${FIG}"/*.pdf 2>/dev/null | tail -40
