#!/bin/zsh
cd ./work/r3_chp
while pgrep -f "chp_var.py cv lab 0 10" > /dev/null; do sleep 3; done
CHP_THREADS=3 ./venv/bin/python -I -B chp_var.py cv alb 0 10 0 >> var_screen.out 2>&1
for v in dup05 teh; do
  CHP_THREADS=3 ./venv/bin/python -I -B chp_var.py cv $v 0 5 0 >> var_screen.out 2>&1
done
CHP_THREADS=3 ./venv/bin/python -I -B et_var.py 0 10 base mf3 dN l4 n2500 dp0 mf3l4dN >> et_screen.out 2>&1
echo QUEUE2_DONE >> var_screen.out
