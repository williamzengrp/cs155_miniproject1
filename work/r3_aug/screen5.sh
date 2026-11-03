#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=1 OMP_NUM_THREADS=1
cp -n cache/runs/lgbm____s0.npz cache/runs2/lgbm____s0.npz
for V in "noise=0.1:2:0.5;mix=1:0.5:0.5" "pl=self:val:soft:1;noise=0.1:2:0.5;mix=1:0.5:0.5" "pl=selfch:val:sharp0.5:1;noise=0.1:2:0.5;mix=1:0.5:0.5"; do
  $S/venv/bin/python -I aug2.py lgbm "$V" 2 2>&1 | grep -E "^\[|Error|error"
done
