# executed by queue3.sh after queue2 finishes (editable until then)
G=./work/gbm
PY=./venv/bin/python
cd $G
GBM_FEATS=v2 $PY -I exp_finalfit.py lgbm rec none 5 5 3 > ff_lgbm.out 2>&1; echo "ff lgbm done $(date)" >> queue.log
GBM_FEATS=v2 $PY -I exp_finalfit.py xgb rec none 5 5 1 > ff_xgb.out 2>&1; echo "ff xgb done $(date)" >> queue.log
GBM_FEATS=v2 $PY -I exp_finalfit.py cat rec none 5 3 1 > ff_cat.out 2>&1; echo "ff cat done $(date)" >> queue.log
GBM_FEATS=v2 $PY -I exp_finalfit.py et rec none 5 3 1 > ff_et.out 2>&1; echo "ff et done $(date)" >> queue.log
