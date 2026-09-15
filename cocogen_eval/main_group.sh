#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
group=$1
device=$2
shift 2
exec >> "$study/logs/main_${group}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/main_${group}.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$study/code"
date -Iseconds
git rev-parse HEAD
echo 'Waiting for successful calibration and frozen-input audit'
while [[ ! -f "$study/logs/calibrate_${group}.exit" || ! -f "$study/logs/validate_inputs_v2.exit" || ! -f "$study/logs/validate_evaluator_v2.exit" ]]; do
  sleep 20
done
[[ $(cat "$study/logs/calibrate_${group}.exit") == 0 ]]
[[ $(cat "$study/logs/validate_inputs_v2.exit") == 0 ]]
[[ $(cat "$study/logs/validate_evaluator_v2.exit") == 0 ]]
for pde in "$@"; do
  "$python" -u -m cocogen_eval.evaluate --pde "$pde" --device "$device" --batch-size 32 --study "$study"
done
date -Iseconds
