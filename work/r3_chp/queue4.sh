#!/bin/zsh
cd ./work/r3_chp
while pgrep -f "chp_var.py cv lab 10 25" > /dev/null; do sleep 3; done
CHP_THREADS=3 ./venv/bin/python -I -B chp_var.py full lab 0,1,2 >> var_screen.out 2>&1
CHP_THREADS=3 ./venv/bin/python -I -B chp_var.py cv lab-alb 0 10 0 >> var_screen.out 2>&1
for v in dup05 teh; do
  CHP_THREADS=3 ./venv/bin/python -I -B chp_var.py cv $v 0 5 0 >> var_screen.out 2>&1
done
CHP_THREADS=3 ./venv/bin/python -I -B chp_seed.py cv 500 3000 10 25 1 >> seed_o500.out 2>&1
echo QUEUE4_DONE >> var_screen.out
