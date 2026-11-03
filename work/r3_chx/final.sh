#!/bin/sh
# assemble r3_chx blocks + public-LB checks
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PYB="./venv/bin/python -I"
D=./work/r3_chx
$PYB $D/chx.py save c 2>&1 | grep saved
$PYB $D/et_block.py a 2>&1 | grep r3_chx
$PYB $D/mix.py c a 2>&1 | grep r3_chx
$PYB ./work/main/pubcheck.py r3_chx_c r3_chx_mix_ca 2>&1 | grep -v Warn
echo ---- extended
$PYB $D/pubx.py r3_chx_c r3_chx_mix_ca 2>&1 | grep -v Warn | head -16
