#!/usr/bin/env bash
set -euo pipefail
archive_log_root=$1
python_bin=$2
job_name=$3
module=$4
shift 4
mkdir -p "$archive_log_root/logs"
exec >> "$archive_log_root/logs/$job_name.log" 2>&1
trap 'status=$?; echo "$status" > "$archive_log_root/logs/$job_name.exit"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
date -Iseconds
git rev-parse HEAD
nice -n 10 ionice -c 3 "$python_bin" -u -m "$module" "$@"
date -Iseconds
