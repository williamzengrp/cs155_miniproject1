#!/bin/zsh
cd ./work/mdl
NJ=2 OMP_NUM_THREADS=2 ./venv/bin/python -I torch_mlp.py tune > logs/torch_tune.log 2>&1
NJ=2 OMP_NUM_THREADS=2 ./venv/bin/python -I ebm_cv.py tune > logs/ebm_tune.log 2>&1
