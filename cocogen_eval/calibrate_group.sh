#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
device=$1
shift
for pde in "$@"; do
  "$python" -u -m cocogen_eval.validate --pde "$pde" --device "$device" --study "$study"
  "$python" -u -m cocogen_eval.calibrate --pde "$pde" --device "$device" --batch-size 32 --study "$study"
done
