#!/bin/bash
#SBATCH --job-name=latbench
#SBATCH --array=0-129
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/latbench_%A_%a.out
#SBATCH --error=logs/latbench_%A_%a.err
#SBATCH --partition=hmem,compute,stats

# ============================================================
#  Array layout (130 tasks):
#    marg : 10 seeds x  1 chunk  =  10 tasks   (indices   0.. 9)
#    logit: 10 seeds x 12 chunks = 120 tasks   (indices  10..129)
#  n_tasks = SEEDS*(1 + LOGIT_CHUNKS). Adjust both together.
# ============================================================
SEEDS=10
LOGIT_CHUNKS=12

# -- Environment (ADJUST to your cluster) --
module load PyTorch/2.0-Miniconda3-4.12.0-Python-3.11.3
source /storage/maths/strdnj/env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -- Repo root (ADJUST) --
cd /storage/maths/strdnj/latticecp

TASK=$SLURM_ARRAY_TASK_ID
N_MARG=$SEEDS
if [ "$TASK" -lt "$N_MARG" ]; then
    FORECASTER=marg; SEED=$TASK; CHUNK=0; NCHUNKS=1
else
    L=$((TASK - N_MARG))
    FORECASTER=logit
    SEED=$((L / LOGIT_CHUNKS))
    CHUNK=$((L % LOGIT_CHUNKS))
    NCHUNKS=$LOGIT_CHUNKS
fi

echo "task=$TASK forecaster=$FORECASTER seed=$SEED chunk=$CHUNK/$NCHUNKS node=$(hostname) start=$(date)"
python stages/s03_run_benchmark.py --task $SEED $FORECASTER $CHUNK $NCHUNKS
echo "finished: $(date)"
