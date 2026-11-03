#!/bin/zsh
cd ./work/fe
while pgrep -f "chain1.sh" > /dev/null; do sleep 5; done
PY=./venv/bin/python
echo "\n## 6. Confirmation of lean sets (3 seed sets: 0/100/200; per-repeat AUC averaged over seed sets)\n| tag | nfeat | AUC | diff | pos | per-rep diff | note |\n|---|---|---|---|---|---|---|" >> log.md
$PY -I confirm.py full-albumnn L_noA1_noG L_noA1_noG_noT > confirmB1.out 2>&1
echo CHAIN2_DONE >> confirmB1.out
