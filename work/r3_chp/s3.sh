#!/bin/zsh
cd ./work/r3_chp
export CHP_THREADS=1 OMP_NUM_THREADS=1
./venv/bin/python -I -B chp_var.py cv lab-alb-tev 17 25 500 >> w3.out 2>&1
./venv/bin/python -I -B chp_var.py full lab-alb-tev 4 >> w3.out 2>&1
echo S3_DONE >> w3.out
