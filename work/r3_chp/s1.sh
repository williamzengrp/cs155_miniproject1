#!/bin/zsh
cd ./work/r3_chp
export CHP_THREADS=1 OMP_NUM_THREADS=1
./venv/bin/python -I -B chp_var.py cv lab-alb-tev 0 9 500 >> w1.out 2>&1
echo S1_DONE >> w1.out
