#!/bin/zsh
cd ./work/r3_chp
while pgrep -f "chp_seed.py cv 500" > /dev/null; do sleep 5; done
for v in lab alb dup05 teh copy; do
  CHP_THREADS=3 ./venv/bin/python -I -B chp_var.py cv $v 0 10 0 >> var_screen.out 2>&1
done
echo QUEUE1_DONE >> var_screen.out
