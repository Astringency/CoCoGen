#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
gpu=$1
shift
exec >> "$study/logs/main24_gpu${gpu}.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/main24_gpu${gpu}.exit"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
date -Iseconds
git rev-parse HEAD
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
free -h
for pde in "$@"; do
  "$python" -u -m cocogen_eval.evaluate --pde "$pde" --device "cuda:$gpu" --batch-size 32 --study "$study"
done
date -Iseconds
