#!/bin/bash
#SBATCH --cpus-per-task=1

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch odin_multi_cpu.slurm {preprocess-af3|select|summarize} [arguments...]" >&2
    exit 2
fi

command=$1
shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "$command" == "preprocess-af3" ]]; then
    exec python -u "$script_dir/odin_multi.py" "$command" "$@"
fi

if [[ "$command" != "select" && "$command" != "summarize" ]]; then
    echo "CPU wrapper supports preprocess-af3, select, and summarize" >&2
    exit 2
fi
if [[ $# -lt 1 ]]; then
    echo "$command requires RUN_DIR" >&2
    exit 2
fi

run_dir=$1
shift
exec python -u "$script_dir/odin_multi.py" "$command" --run-dir "$run_dir" "$@"
