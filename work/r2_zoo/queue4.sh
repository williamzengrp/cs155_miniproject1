#!/bin/zsh
# after queue3: mini screen B (monotone year constraint, HGB, GOSS, depth-1 boosting, bagged trees)
Z=./work/r2_zoo
PY=./venv/bin/python
cd $Z
until grep -q Q3DONE logs/q3.out 2>/dev/null; do sleep 3; done
export OMP_NUM_THREADS=2 NJ=2
$PY -I zoo.py cv 1000 5 0 lgb_mono lgb_d1 lgb_goss hgb bag_dt > logs/screenB.out 2>&1
echo QBDONE >> logs/screenB.out
