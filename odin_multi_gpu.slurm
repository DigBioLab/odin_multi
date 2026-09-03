#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch --array=0-N odin_multi_gpu.slurm {design|evaluate} RUN_DIR [options...]" >&2
    echo "Pass setup options after RUN_DIR for the first design submission." >&2
    exit 2
fi

command=$1
run_dir=$2
shift 2
if [[ "$command" != "design" && "$command" != "evaluate" ]]; then
    echo "GPU wrapper supports only design and evaluate" >&2
    exit 2
fi

shard_index=${SLURM_ARRAY_TASK_ID:-0}
num_shards=${SLURM_ARRAY_TASK_COUNT:-1}
array_min=${SLURM_ARRAY_TASK_MIN:-0}
array_max=${SLURM_ARRAY_TASK_MAX:-0}
array_step=${SLURM_ARRAY_TASK_STEP:-1}
if [[ "$array_min" -ne 0 || "$array_step" -ne 1 || "$array_max" -ne $((num_shards - 1)) ]]; then
    echo "Slurm array must be contiguous and zero-based (for example --array=0-7)" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python -u "$script_dir/odin_multi.py" "$command" \
    --run-dir "$run_dir" "$@" \
    --shard-index "$shard_index" --num-shards "$num_shards"
