#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
job=$1
module=$2
shift 2
exec >> "$study/logs/${job}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/${job}.exit"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
date -Iseconds
git rev-parse HEAD
"$python" -u -m "$module" "$@" --study "$study"
date -Iseconds
