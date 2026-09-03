#!/bin/bash
# Run or continue Odin-Multi design locally.
# Usage: ./submit.sh GPU_ID RUN_DIR [odin_multi design options...]

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: ./submit.sh GPU_ID RUN_DIR [odin_multi design options...]" >&2
    exit 2
fi

gpu_id=$1
run_dir=$2
shift 2

export CUDA_VISIBLE_DEVICES=$gpu_id
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python -u "$script_dir/odin_multi.py" design --run-dir "$run_dir" "$@"
