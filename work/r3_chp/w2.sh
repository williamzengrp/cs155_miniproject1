#!/bin/zsh
cd ./work/r3_chp
export CHP_THREADS=1 OMP_NUM_THREADS=1
./venv/bin/python -I -B chp_var.py cv lab 18 25 0 >> w2.out 2>&1
./venv/bin/python -I -B chp_var.py cv alb 15 20 0 >> w2.out 2>&1
./venv/bin/python -I -B chp_var.py full alb 1 >> w2.out 2>&1
./venv/bin/python -I -B chp_var.py cv lab-alb 4 8 0 >> w2.out 2>&1
echo W2_DONE >> w2.out
