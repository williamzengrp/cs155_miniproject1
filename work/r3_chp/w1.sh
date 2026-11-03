#!/bin/zsh
cd ./work/r3_chp
export CHP_THREADS=1 OMP_NUM_THREADS=1
./venv/bin/python -I -B chp_var.py cv lab 11 18 0 >> w1.out 2>&1
./venv/bin/python -I -B chp_var.py cv alb 10 15 0 >> w1.out 2>&1
./venv/bin/python -I -B chp_var.py full alb 0 >> w1.out 2>&1
./venv/bin/python -I -B chp_var.py cv lab-alb 0 4 0 >> w1.out 2>&1
echo W1_DONE >> w1.out
