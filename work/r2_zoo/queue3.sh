#!/bin/zsh
# after queue2: 3-seed lgb_linear vs 3-seed lgbm_ref (fair comparison of linear leaves)
Z=./work/r2_zoo
PY=./venv/bin/python
cd $Z
until grep -q Q2DONE logs/q2.out 2>/dev/null; do sleep 3; done
export OMP_NUM_THREADS=2 NJ=2
$PY -I zoo.py cv 1000 5 100 lgb_linear > logs/q3.out 2>&1
$PY -I zoo.py cv 1000 5 200 lgb_linear >> logs/q3.out 2>&1
$PY -I zoo.py cv 1000 5 100 lgbm_ref >> logs/q3.out 2>&1
$PY -I zoo.py cv 1000 5 200 lgbm_ref >> logs/q3.out 2>&1
echo Q3DONE >> logs/q3.out
