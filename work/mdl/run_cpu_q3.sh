#!/bin/zsh
cd ./work/mdl
while pgrep -f run_cpu_q2.sh >/dev/null; do sleep 10; done
OMP_NUM_THREADS=2 ../../venv/bin/python -I base_v2_ref.py > logs/base_v2_ref.log 2>&1
