#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
exec >> "$study/logs/aggregate60.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/aggregate60.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$study/code_phase1"
date -Iseconds
echo 'Waiting for all four PDE main evaluations'
while [[ ! -f "$study/logs/main_ph.exit" || ! -f "$study/logs/main_dn.exit" ]]; do
  sleep 20
done
[[ $(cat "$study/logs/main_ph.exit") == 0 ]]
[[ $(cat "$study/logs/main_dn.exit") == 0 ]]
"$python" -u -m cocogen_eval.evaluate --aggregate --study "$study"
date -Iseconds
