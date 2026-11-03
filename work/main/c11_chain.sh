#!/bin/zsh
W=./work
cd $W/main
NREP=3 DO_FULL=0 ../../venv/bin/python -I final_cv.py 3 c11 $W/main/final_params.json 11000 lgbm xgb cat et rf > c11_trees.out 2>&1
NREP=3 SEED0=11000 PREFIX=c11_ch ../../venv/bin/python -I conf_ch.py et > c11_ch_et.out 2>&1
NREP=3 SEED0=11000 PREFIX=c11_ch ../../venv/bin/python -I conf_ch.py lgbm > c11_ch_lgbm.out 2>&1
echo C11_TREES_CH_DONE >> c11_trees.out
