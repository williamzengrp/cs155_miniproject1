#!/bin/zsh
D=./work/r2_fsel
PY=$D/../../venv/bin/python
until grep -qE "mean group PI|Traceback" $D/pi_lgbm_std.out; do sleep 3; done
for R in f0 g0; do $PY -I $D/nested_sel.py lgbm $R 1000 5 fsel; done
$PY -I $D/run_fsel.py rs_lgbm_k8f60 1000 5 fsel 0
echo CHAIN1_DONE
