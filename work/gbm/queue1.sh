#!/bin/bash
# sequential heavy jobs (one at a time, 2 threads each)
G=./work/gbm
PY=./venv/bin/python
cd $G
echo "queue1 start $(date)" >> queue.log
$PY -I exp_weights2.py > exp_weights2.out 2>&1; echo "weights2 done $(date)" >> queue.log
$PY -I tune.py lgbm 40 > tune_lgbm.out 2>&1; echo "tune lgbm done $(date)" >> queue.log
$PY -I select_params.py lgbm none 5 > select_lgbm.out 2>&1; echo "select lgbm done $(date)" >> queue.log
$PY -I tune.py xgb 40 > tune_xgb.out 2>&1; echo "tune xgb done $(date)" >> queue.log
$PY -I select_params.py xgb none 5 > select_xgb.out 2>&1; echo "select xgb done $(date)" >> queue.log
$PY -I tune.py cat 20 > tune_cat.out 2>&1; echo "tune cat done $(date)" >> queue.log
$PY -I select_params.py cat none 3 > select_cat.out 2>&1; echo "select cat done $(date)" >> queue.log
$PY -I tune.py et 20 > tune_et.out 2>&1; echo "tune et done $(date)" >> queue.log
$PY -I select_params.py et none 4 > select_et.out 2>&1; echo "select et done $(date)" >> queue.log
$PY -I tune.py rf 20 > tune_rf.out 2>&1; echo "tune rf done $(date)" >> queue.log
$PY -I select_params.py rf none 4 > select_rf.out 2>&1; echo "select rf done $(date)" >> queue.log
echo "queue1 end $(date)" >> queue.log
