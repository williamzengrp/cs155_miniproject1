#!/bin/zsh
W=./work
PY=$W/../venv/bin/python
cd $W/r2_shift
for ws in adv advs advT; do
  NREP=5 DO_FULL=0 NSEED=1 WS=$ws $PY -I shift_cv.py 2 shift_tmp_w${ws} $W/main/final_params.json 1000 lgbm
done
echo CHAIN_DONE
