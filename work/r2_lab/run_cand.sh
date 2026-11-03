#!/bin/zsh
# usage: run_cand.sh NAME SPEC DEDUP NINNER   -> results lab_<NAME>_{lgbm,xgb,cat,et,rf} (std, test_full 3 seeds)
#                                               and lab_c9_<NAME>_{...} (c9, OOF only)
W=./work
PY=./venv/bin/python
NAME=$1; export SPEC=$2; export DEDUP=$3; export NINNER=$4
cd $W/r2_lab
NREP=5 DO_FULL=1 $PY -I final_lab.py 2 lab_${NAME} $W/main/final_params.json 1000 lgbm xgb cat et rf > out/std_${NAME}.log 2>&1
cd $W/main && $PY -I eval_candidate.py std --replace fkd_lgbm:lab_${NAME}_lgbm,fkt_xgb:lab_${NAME}_xgb,fkt_cat:lab_${NAME}_cat,fkd_et:lab_${NAME}_et,fkd_rf:lab_${NAME}_rf >> $W/r2_lab/out/std_${NAME}.log 2>&1
cd $W/r2_lab
NREP=3 DO_FULL=0 $PY -I final_lab.py 2 lab_c9_${NAME} $W/main/final_params.json 9000 lgbm xgb cat et rf > out/c9_${NAME}.log 2>&1
cd $W/main && $PY -I eval_candidate.py c9 --replace conf_fkd_lgbm:lab_c9_${NAME}_lgbm,conf_fkt_xgb:lab_c9_${NAME}_xgb,conf_fkt_cat:lab_c9_${NAME}_cat,conf_fkd_et:lab_c9_${NAME}_et,conf_fkd_rf:lab_c9_${NAME}_rf >> $W/r2_lab/out/c9_${NAME}.log 2>&1
echo CHAIN_DONE >> $W/r2_lab/out/c9_${NAME}.log
