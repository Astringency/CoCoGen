#!/usr/bin/env bash
set -euo pipefail
study=/research_data/users/zhangxifeng/C01Python/FM4PDE/outputs/cocogen_main_20260915
python=/research_data/users/zhangxifeng/.conda/envs/fm4pde/bin/python
exec >> "$study/logs/schedule24_v2.log" 2>&1
trap 'status=$?; echo "$status" > "$study/logs/schedule24_v2.exit"' EXIT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
date -Iseconds
git rev-parse HEAD
"$python" -u -m cocogen_eval.schedule_24h --hours-limit 24 --margin 1.15 --study "$study"
date -Iseconds
