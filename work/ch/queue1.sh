#!/bin/zsh
# ch agent queue: wait for residual stage-1 runs, then one heavy job at a time.
PY=./venv/bin/python
cd ./work/ch
while [ ! -f cache/resid_lgbm_std_off200.pkl ]; do sleep 10; done
$PY -I chmodel.py et 0 --save > chmodel_et.out 2>&1
$PY -I chmodel.py lgbm 0 --save > chmodel_lgbm.out 2>&1
$PY -I screen.py 0 crossdate cyckey+tempo+nn13+ydp subset_mean+cluster+position onehot interactions > screen_off0.out 2>&1
echo QUEUE1_DONE >> screen_off0.out
