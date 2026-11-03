#!/bin/zsh
PY=./venv/bin/python
cd ./work/ch
until grep -q QUEUE2_DONE chmodel_etnw.out 2>/dev/null; do sleep 10; done
$PY -I chmodel.py lgbm 100 --save > chmodel_lgbm_off100.out 2>&1
$PY -I chmodel.py lgbm 200 --save > chmodel_lgbm_off200.out 2>&1
echo QUEUE3_DONE >> chmodel_lgbm_off200.out
