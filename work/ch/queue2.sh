#!/bin/zsh
# ch agent queue 2: runs after queue1 (one heavy job at a time).
PY=./venv/bin/python
cd ./work/ch
until grep -q QUEUE1_DONE screen_off0.out 2>/dev/null; do sleep 10; done
$PY -I chmodel.py et 100 --save > chmodel_et_off100.out 2>&1
$PY -I chmodel.py et 200 --save > chmodel_et_off200.out 2>&1
$PY -I chmodel.py etnw 0 --save > chmodel_etnw.out 2>&1
echo QUEUE2_DONE >> chmodel_etnw.out
