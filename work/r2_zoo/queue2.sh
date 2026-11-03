#!/bin/zsh
# wait for screen A, then 3-seed cat_alb vs 3-seed cat_ref (fair comparison of the categorical keys)
Z=./work/r2_zoo
PY=./venv/bin/python
cd $Z
while pgrep -f "zoo.py cv 1000 5 0 cat_alb" > /dev/null; do sleep 3; done
export OMP_NUM_THREADS=2 NJ=2
$PY -I zoo.py cv 1000 5 100 cat_alb > logs/q2.out 2>&1
$PY -I zoo.py cv 1000 5 200 cat_alb >> logs/q2.out 2>&1
$PY -I zoo.py cv 1000 5 100 cat_ref >> logs/q2.out 2>&1
$PY -I zoo.py cv 1000 5 200 cat_ref >> logs/q2.out 2>&1
echo Q2DONE >> logs/q2.out
