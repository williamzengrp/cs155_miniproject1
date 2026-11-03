#!/bin/zsh
# after screen B: save near-miss members with test_full on std (reuses raw 3-seed OOF), then c9 (OOF only)
Z=./work/r2_zoo
PY=./venv/bin/python
cd $Z
until grep -q QBDONE logs/screenB.out 2>/dev/null; do sleep 3; done
export OMP_NUM_THREADS=2 NJ=2
DO_FULL=1 $PY -I zoo.py run 1000 5 zoo 3 lgb_linear cat_alb > logs/q5.out 2>&1
DO_FULL=0 $PY -I zoo.py run 9000 3 zoo_c9 3 lgb_linear cat_alb >> logs/q5.out 2>&1
echo Q5DONE >> logs/q5.out
