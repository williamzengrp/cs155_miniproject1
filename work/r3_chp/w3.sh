#!/bin/zsh
cd ./work/r3_chp
export CHP_THREADS=1 OMP_NUM_THREADS=1
./venv/bin/python -I -B chp_var.py full lab 0,1,2 >> w3.out 2>&1
./venv/bin/python -I -B chp_var.py cv alb 20 25 0 >> w3.out 2>&1
./venv/bin/python -I -B chp_var.py full alb 2 >> w3.out 2>&1
./venv/bin/python -I -B chp_var.py cv lab-alb 8 10 0 >> w3.out 2>&1
echo W3_DONE >> w3.out
