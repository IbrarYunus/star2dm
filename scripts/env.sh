# Sourced by every job script. The only file in this repository you must edit to run it
# somewhere other than where it was developed.
#
#   1. ACTIVATE     how to get the stars2dm environment onto PATH
#   2. STARS2DM_*   where the data lives (see src/stars2dm_paths.py for what each holds)
#
# Everything else derives from these. Values already set in the environment win, so a single
# job can be redirected without editing the file:
#   STARS2DM_DREAMS_ROOT=/somewhere/else sbatch scripts/03_infer_dreams.sh

# --- 1. environment ---------------------------------------------------------------------
# On the development cluster anaconda is not loaded by ~/.bashrc (it slows non-interactive
# SSH), so a separate file must be sourced before `conda activate` works in a batch job.
# Replace both lines with whatever puts your environment on PATH.
if [ -f ~/.bashrc_conda ]; then
    source ~/.bashrc_conda
fi
conda activate "${STARS2DM_ENV:-stars2dm}"

# --- 2. this repository -------------------------------------------------------------------
export STARS2DM_REPO="${STARS2DM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# src/ goes on PYTHONPATH ahead of anything else. With `pip install -e .` this is redundant;
# without it the code still runs. It also settles an ambiguity that is otherwise silent and
# nasty: if another copy of `baryonbridge` is installed in the environment -- an editable
# install pointing at a development checkout, say -- an unqualified `import baryonbridge`
# finds that one, and the whole run exercises different code than the repository holds, with
# no error anywhere. Prepending makes this repository win.
export PYTHONPATH="${STARS2DM_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

# --- 3. data roots ------------------------------------------------------------------------
export STARS2DM_DATA_ROOT="${STARS2DM_DATA_ROOT:-${STARS2DM_REPO}/data}"
export STARS2DM_DREAMS_ROOT="${STARS2DM_DREAMS_ROOT:-${STARS2DM_DATA_ROOT}/DREAMS_data}"
export STARS2DM_TNG_ROOT="${STARS2DM_TNG_ROOT:-${STARS2DM_DATA_ROOT}/TNG_data}"
export STARS2DM_FIRE_ROOT="${STARS2DM_FIRE_ROOT:-${STARS2DM_DATA_ROOT}/FIRE_data}"
export STARS2DM_PRED_ROOT="${STARS2DM_PRED_ROOT:-${STARS2DM_DREAMS_ROOT}/Pred/CDM/MW_zooms/SB5}"
export STARS2DM_CKPT_ROOT="${STARS2DM_CKPT_ROOT:-${STARS2DM_DATA_ROOT}/checkpoints}"
export STARS2DM_CACHE_ROOT="${STARS2DM_CACHE_ROOT:-${STARS2DM_DATA_ROOT}/figure_cache}"
export STARS2DM_FIG_DIR="${STARS2DM_FIG_DIR:-${STARS2DM_DATA_ROOT}/figures}"

# Absolute, so a script that cd's into a subdirectory still writes logs in one place.
mkdir -p "${STARS2DM_REPO}/logs" "${STARS2DM_FIG_DIR}"

# --- helpers ------------------------------------------------------------------------------
# Resolve a version tag to its checkpoint, refusing to start a multi-hour job without one.
require_ckpt() {
    CKPT="${STARS2DM_CKPT_ROOT}/last-${1}.ckpt"
    if [ ! -f "${CKPT}" ]; then
        echo "ERROR: checkpoint not found: ${CKPT}" >&2
        echo "       fetch it with: python scripts/fetch_data.py --tier checkpoints" >&2
        exit 1
    fi
    echo "${CKPT}"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required input not found: $1" >&2
        exit 1
    fi
}

# Lightning picks SLURMEnvironment iff SLURM_NTASKS is set (fabric/plugins/environments/
# slurm.py::_is_srun_used). That environment reports creates_processes_externally=True, so
# _SubprocessScriptLauncher never spawns rank 1 and multi-GPU DDP hangs in rendezvous until
# the TCPStore times out -- no error, just a job that burns its whole allocation doing
# nothing. sbatch inherits SLURM_NTASKS from the submitting shell, so submitting a training
# job from inside an interactive allocation triggers it. Scrub the variables.
scrub_slurm_ntasks() {
    unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE SLURM_NPROCS
}
