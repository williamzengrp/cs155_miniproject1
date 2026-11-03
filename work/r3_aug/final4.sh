#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=1 OMP_NUM_THREADS=1
$S/venv/bin/python -I aug2.py lgbm "pl=chg:val:sharp0.5:1;noise=0.1:2:0.5;mix=1:0.5:0.5" 5 r3_aug_lgbm_plgnm 2>&1 | grep -vi warning
