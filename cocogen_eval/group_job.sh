#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
job=$1
shift
mkdir -p "$study/logs"
exec >> "$study/logs/${job}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/${job}.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$study/code"
date -Iseconds
git rev-parse HEAD
bash cocogen_eval/calibrate_group.sh "$@"
date -Iseconds
