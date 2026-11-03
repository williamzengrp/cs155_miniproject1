#!/bin/zsh
# c11 verification members for the final8v2 candidate + base_et test_full on std
( cd ./work/main && DO_FULL=1 OMP_NUM_THREADS=2 ./venv/bin/python -I base_et_run.py 2 1000 5 b2 > ./work/v2logs/b2_et_std.out 2>&1 ;   DO_FULL=0 OMP_NUM_THREADS=2 ./venv/bin/python -I base_et_run.py 2 11000 3 v11_base > ./work/v2logs/v11_base_et.out 2>&1 ) &
( cd ./work/r2_graph && NREP=3 DO_FULL=0 OMP_NUM_THREADS=3 ./venv/bin/python -I graph_cv.py 3 v11_ds ./work/main/final_params.json 11000 dupm_spec8 lgbm xgb > ./work/v2logs/v11_ds.out 2>&1 ) &
( cd ./work/r2_zoo && NJ=3 DO_FULL=0 OMP_NUM_THREADS=3 ./venv/bin/python -I zoo.py run 11000 3 v11 3 cat_alb > ./work/v2logs/v11_cat_alb.out 2>&1 ) &
wait
echo ALLDONE
