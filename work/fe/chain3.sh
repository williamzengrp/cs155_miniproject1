#!/bin/zsh
cd ./work/fe
while pgrep -f "exp_dup.py" > /dev/null; do sleep 5; done
PY=./venv/bin/python
echo "\n## 6. Confirmation runs (3 seed sets 0/100/200; per-repeat AUC averaged over seed sets; diff vs full67 3-seed avg)\n| tag | nfeat | AUC | diff | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|" >> log.md
$PY -I confirm.py full+detrended L1D L2D > confirmB1.out 2>&1
echo CHAIN3_DONE >> confirmB1.out
