#!/usr/bin/env bash
set -euo pipefail
diagnostic_root=$1
python_bin=$2
job_name=$3
shift 3
mkdir -p "$diagnostic_root/logs"
exec >> "$diagnostic_root/logs/$job_name.log" 2>&1
trap 'status=$?; echo "$status" > "$diagnostic_root/logs/$job_name.exit"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
date -Iseconds
git rev-parse HEAD
"$python_bin" -u -m cocogen_eval.long_sampling_diagnostic --root "$diagnostic_root" "$@"
date -Iseconds
