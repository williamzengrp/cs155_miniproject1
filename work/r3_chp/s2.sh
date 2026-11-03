#!/bin/zsh
cd ./work/r3_chp
export CHP_THREADS=1 OMP_NUM_THREADS=1
./venv/bin/python -I -B chp_var.py cv lab-alb-tev 9 17 500 >> w2.out 2>&1
./venv/bin/python -I -B chp_var.py full lab-alb-tev 3 >> w2.out 2>&1
echo S2_DONE >> w2.out
