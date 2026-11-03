#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=2 OMP_NUM_THREADS=2
rm -f cache/runs2/lgbm____s0.npz
$S/venv/bin/python -I aug2.py lgbm "noise=0.1:2:0.5;mix=1:0.5:0.5" 5 r3_aug_lgbm_nm 2>&1 | grep -vi warning
