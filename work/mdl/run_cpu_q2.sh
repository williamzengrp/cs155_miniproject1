#!/bin/zsh
cd ./work/mdl
while pgrep -f run_cpu_q1.sh >/dev/null; do sleep 10; done
P1='{"feats": "all", "h": [256, 128], "drop": 0.4, "wd": 0.001, "seeds": 5, "ple": 16, "epochs": 40}'
P2='{"feats": "v2", "h": [256, 128], "drop": 0.4, "wd": 0.001, "seeds": 5, "ple": 16, "epochs": 40}'
NJ=2 OMP_NUM_THREADS=2 ./venv/bin/python -I torch_mlp.py std "$P1" mdl_mlp_ple > logs/mlp_std.log 2>&1
NJ=2 OMP_NUM_THREADS=2 ./venv/bin/python -I torch_mlp.py std "$P2" mdl_mlp_ple_v2 > logs/mlp_std_v2.log 2>&1
