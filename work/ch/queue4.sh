#!/bin/zsh
# ch agent: replica queue (one heavy job at a time, 4 threads)
PY=./venv/bin/python
cd ./work/ch
while pgrep -f "chfull.py cv std 0 15" > /dev/null; do sleep 10; done
$PY -I -W ignore chfull.py cv std 15 25 > chfull_std_b.out 2>&1
$PY -I -W ignore chfull.py cv conf 0 15 > chfull_conf.out 2>&1
$PY -I -W ignore chfull.py full 0,1,2 > chfull_full.out 2>&1
echo QUEUE4_DONE >> chfull_full.out
