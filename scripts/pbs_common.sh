#!/bin/bash
# Shared setup for every PBS job in this repo (Aurora). Adapted from mini-embed-filip.
# Every job script starts with:
#   cd "${PBS_O_WORKDIR:?submit with qsub from the repo root}" || exit 1
#   source ./pbs_common.sh
# THE cd MUST COME FIRST: PBS runs a spooled copy with $HOME as cwd, so `source ./pbs_common.sh`
# without the cd looks in $HOME and fails. config.py owns all model/dataset paths; job scripts
# must NOT export PROTGEN_* (that would override the config silently).
set -o pipefail

if [ ! -f config.py ] || [ ! -d src ]; then
    echo "FATAL: $(pwd) is not the protein_generation repo root (need config.py and src/)." >&2
    exit 1
fi

module load frameworks                                     # provides torch + IPEX + oneCCL on Aurora
# Create once: python -m venv --system-site-packages, then pip install -e deps. Edit to your env.
source /flare/NLDesignProtein/bryan/envs/protein-gen-env/bin/activate

# --- topology: 12 tiles/node, one rank per tile ---
RANKS_PER_NODE=${RANKS_PER_NODE:-12}
NNODES=$(wc -l < "$PBS_NODEFILE" | tr -d " ")
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# --- device selector + rendezvous (set AFTER module load frameworks) ---
# frameworks defaults ONEAPI_DEVICE_SELECTOR to opencl+level_zero, which double-enumerates tiles and
# breaks one-rank-per-tile. Force Level-Zero only. See dist.py's warning.
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- oneCCL / fabric (do KVS exchange over MPI/PMIx, not CCL's internal TCP KVS which melts at scale) ---
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=pidfd

MPI_LAUNCH=(mpiexec -n "${NRANKS}" -ppn "${RANKS_PER_NODE}" --pmi=pmix --cpu-bind depth -d 8)

job_banner() {
    local title="$1"; shift
    echo "================================================================"
    echo "  ${title}"
    echo "  job        : ${PBS_JOBID:-<interactive>}  started $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "  repo       : $(pwd)   commit $(git rev-parse --short HEAD 2>/dev/null || echo '<no git>')"
    echo "  topology   : ${NNODES} nodes x ${RANKS_PER_NODE} ranks/node = ${NRANKS} ranks"
    local e
    for e in ONEAPI_DEVICE_SELECTOR CCL_PROCESS_LAUNCHER CCL_KVS_MODE FI_PROVIDER OMP_NUM_THREADS; do
        printf '    %-24s = %s\n' "$e" "${!e}"
    done
    echo "  paths resolved by config.py:"
    python config.py 2>&1 | sed 's/^/    /'
    local v
    for v in "$@"; do printf '    %-24s = %s\n' "$v" "${!v}"; done
    echo "================================================================"
}
