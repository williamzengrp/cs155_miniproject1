#!/bin/zsh
cd ./work/fe
while pgrep -f "ablate1.py" > /dev/null; do sleep 5; done
PY=./venv/bin/python
$PY -I exp_dup.py > exp_dup.out 2>&1
$PY -I exp_misc.py > exp_misc.out 2>&1
echo CHAIN1_DONE >> exp_misc.out
