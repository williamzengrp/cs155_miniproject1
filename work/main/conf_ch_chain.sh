#!/bin/zsh
W=./work
cd $W/main
while [ ! -f $W/results/conf_base_rf.npz ]; do sleep 15; done
NREP=3 SEED0=9000 PREFIX=conf_ch ../../venv/bin/python -I conf_ch.py et > conf_ch_et.out 2>&1
NREP=3 SEED0=9000 PREFIX=conf_ch ../../venv/bin/python -I conf_ch.py lgbm > conf_ch_lgbm.out 2>&1
