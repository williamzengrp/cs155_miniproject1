#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=2 OMP_NUM_THREADS=2
for V in "noise=0.1:2:0.5" "mix=1:0.5:0.5" "noise=0.1:2:0.5;mix=1:0.5:0.5"; do
  $S/venv/bin/python -I chaug.py cv et "$V" 0 5 2>&1 | grep -v -i warn
  $S/venv/bin/python -I cq.py et $(echo "$V" | sed 's/;/_/g; s/:/-/g; s/=/-/g') 1
done
$S/venv/bin/python -I chaug.py cv cat "noise=0.1:1:0.5" 0 5 2>&1 | grep -v -i warn
$S/venv/bin/python -I cq.py cat noise-0.1-1-0.5 1
