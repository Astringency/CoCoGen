#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
group=$1
shift
mkdir -p "$study/logs"
exec >> "$study/logs/prepare_${group}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/prepare_${group}.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
cd "$study/code"
date -Iseconds
git rev-parse HEAD
for pde in "$@"; do
  "$python" -u -m cocogen_eval.prepare --pde "$pde" --study "$study"
done
date -Iseconds
