#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=1 OMP_NUM_THREADS=1
for V in "pl=selfch:val:sharp0.5:1" "pl=ch:val:sharp0.5:1;noise=0.1:2:0.5" "pl=selfch:val:soft:1;noise=0.1:2:0.5" "pl=selfch:val:soft:2"; do
  $S/venv/bin/python -I aug.py lgbm "$V" 2 2>&1 | grep -E "^\[|Error|error"
done
for V in "" "pl=ch:val:soft:1" "pl=selfch:val:soft:1"; do
  $S/venv/bin/python -I aug.py et "$V" 2 2>&1 | grep -E "^\[|Error|error"
done
