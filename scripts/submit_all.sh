#!/bin/bash
# The full chain, in order. Run from the repository root. Each stage consumes the previous
# stage's output, so wait for one to finish before submitting the next.
#
#   bash scripts/submit_all.sh <stage>
#
# Stages: train | infer | sweeps | aniso | figures
#
# Skip straight to `figures` if you downloaded the predictions
# (python scripts/fetch_data.py --tier predictions), or to nothing at all if you downloaded
# the figure cache -- see docs/REPRODUCE.md for the three entry points.

set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?usage: bash scripts/submit_all.sh train|infer|sweeps|aniso|figures}

# sbatch inherits the submitting shell's SLURM_* variables, which makes Lightning think it is
# already under srun and silently hangs multi-GPU jobs. Scrub them from every submission.
scrub() {
    env $(env | grep -oE '^SLURM[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ') "$@"
}

case "${STAGE}" in

train)
    for M in v95 v97; do
        printf "  train %s  " "$M"
        scrub sbatch --export="ALL,MODEL=${M}" scripts/02_train.sh
    done
    ;;

infer)
    # 1024 DREAMS galaxies in 9 shards of ~115. Shard boundaries are not arbitrary: the
    # held-out split is [0, 102), so keeping it as its own shard means the calibration
    # numbers can be regenerated without re-running the other 922 galaxies.
    DREAMS_BOUNDS=(0 102 217 332 447 562 677 792 907 1024)
    echo "=== DREAMS v95, 100 draws ==="
    for ((i = 0; i < ${#DREAMS_BOUNDS[@]} - 1; i++)); do
        s=${DREAMS_BOUNDS[$i]}; e=${DREAMS_BOUNDS[$((i + 1))]}
        printf "  [%4d, %4d)  %3d galaxies  " "$s" "$e" "$((e - s))"
        scrub sbatch --export="ALL,TAG=v95,NDRAWS=100,START_IDX=${s},END_IDX=${e}" scripts/03_infer_dreams.sh
    done

    # The 10-draw sets: Appendix A stamp grids (v95) and Appendix D DREAMS panels (v97).
    echo "=== DREAMS 10-draw sets ==="
    for TAG in v95 v97; do
        printf "  %s  " "$TAG"
        scrub sbatch --export="ALL,TAG=${TAG},NDRAWS=10,START_IDX=0,END_IDX=1024" scripts/03_infer_dreams.sh
    done

    echo "=== out of domain ==="
    for ((s = 0; s < 264; s += 132)); do
        printf "  TNG50 v95 [%3d, %3d)  " "$s" "$((s + 132))"
        scrub sbatch --export="ALL,TAG=v95,SUITE=TNG50,START_IDX=${s},END_IDX=$((s + 132))" scripts/04_infer_ood.sh
    done
    printf "  TNG50 v97 (10 draws)  "
    scrub sbatch --export="ALL,TAG=v97,SUITE=TNG50,NDRAWS=10,START_IDX=0,END_IDX=264" scripts/04_infer_ood.sh
    for TAG in v95 v97; do
        printf "  FIREm12 %s  " "$TAG"
        scrub sbatch --export="ALL,TAG=${TAG},SUITE=FIREm12" scripts/04_infer_ood.sh
    done
    ;;

sweeps)
    for S in om s8 sn1 sn2 bhff; do
        printf "  %-5s  " "$S"
        scrub sbatch --export="ALL,TAG=v95,SWEEP=${S}_sweep_5" scripts/05_infer_sweeps.sh
    done
    ;;

aniso)
    scrub sbatch --export="ALL,TAG=v95" scripts/06_anisotropy.sh
    ;;

figures)
    for ST in dreams ood support sweeps aniso; do
        printf "  %-8s  " "$ST"
        scrub sbatch --export="ALL,STAGE=${ST}" scripts/07_figures.sh
    done
    ;;

*)
    echo "ERROR: unknown stage '${STAGE}'" >&2
    exit 1
    ;;
esac
