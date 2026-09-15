#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
job=$1
module=$2
shift 2
mkdir -p "$study/logs"
exec >> "$study/logs/${job}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/${job}.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$study/code"
date -Iseconds
git rev-parse HEAD
"$python" -m "$module" "$@" --study "$study"
date -Iseconds
