#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=3
for M in lgbm; do
 rm -f cache/runs/${M}____s0.npz
 for V in "" "pl=ch:val:soft:1" "pl=ch:val:soft:0.5" "pl=ch:val:soft:2" "pl=self:val:soft:1" "pl=selfch:val:soft:1" "pl=ch:val:sharp0.5:1" "pl=ch:val:hard0.3:1" "noise=0.05:2:0.5" "noise=0.1:2:0.5" "mix=1:0.5:0.5" "tta=0.03:4" "tta=0.1:4"; do
  $S/venv/bin/python -I aug.py $M "$V" 2 2>&1 | grep -E "^\[|Error|error"
 done
done
