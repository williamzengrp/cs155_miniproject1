#!/bin/bash
S=.
cd $S/work/r3_aug
export NJ=2 OMP_NUM_THREADS=2
$S/venv/bin/python -I chaug.py cv et "pl=ch:soft:1" 0 5
$S/venv/bin/python -I chaug.py cv cat "pl=ch:soft:1" 0 5
