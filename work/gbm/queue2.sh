#!/bin/bash
# sequential heavy jobs (one at a time, 2 threads each); tuning on v1 (67 feats), confirmations on v1 + v2
G=./work/gbm
PY=./venv/bin/python
cd $G
echo "queue2 start $(date)" >> queue.log
GBM_FEATS=v2 $PY -I confirm_v2.py lgbm > confirm_v2_lgbm.out 2>&1; echo "confirm_v2 lgbm done $(date)" >> queue.log
for m in xgb:40:5 cat:16:3 et:20:4 rf:20:4; do
  IFS=: read name nt ntop <<< "$m"
  $PY -I tune.py $name $nt > tune_$name.out 2>&1; echo "tune $name done $(date)" >> queue.log
  $PY -I select_params.py $name none $ntop > select_$name.out 2>&1; echo "select $name done $(date)" >> queue.log
  GBM_FEATS=v2 $PY -I confirm_v2.py $name > confirm_v2_$name.out 2>&1; echo "confirm_v2 $name done $(date)" >> queue.log
done
echo "queue2 end $(date)" >> queue.log
