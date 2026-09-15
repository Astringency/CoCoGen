#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
group=$1
device=$2
shift 2
exec >> "$study/logs/calibrate_cost_${group}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/calibrate_cost_${group}.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$study/code_cost"
date -Iseconds
git rev-parse HEAD
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
for pde in "$@"; do
  if [[ ! -f "$study/validation/${pde}.json" ]]; then
    "$python" -u -m cocogen_eval.validate --pde "$pde" --device "$device" --study "$study"
  fi
  "$python" -u -m cocogen_eval.cost_calibrate --pde "$pde" --device "$device" --batch-size 32 --max-nfe 500 --study "$study"
done
date -Iseconds
