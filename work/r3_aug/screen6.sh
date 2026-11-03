#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=2 OMP_NUM_THREADS=2
for V in "pl=chg:val:sharp0.5:1;noise=0.1:2:0.5;mix=1:0.5:0.5" "pl=chga:val:sharp0.5:1;noise=0.1:2:0.5;mix=1:0.5:0.5"; do
  $S/venv/bin/python -I aug2.py lgbm "$V" 2 2>&1 | grep -E "^\[|Error|error|Trace"
done
