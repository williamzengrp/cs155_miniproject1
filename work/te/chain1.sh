#!/bin/zsh
# sequential heavy jobs (one at a time): wait for confirm.py, then collective, final, catboost
cd ./work/te
while pgrep -f "confirm.py S3_KD" > /dev/null; do sleep 10; done
./venv/bin/python -I collective_cv.py KD1 1 knnw_yr_40 teh_date_m10 cnt_date > out/coll_KD1.log 2>&1 < /dev/null
./venv/bin/python -I collective_eval.py KD1 > out/colleval_KD1.log 2>&1 < /dev/null
./venv/bin/python -I collective_cv.py BASE1 1 > out/coll_BASE1.log 2>&1 < /dev/null
./venv/bin/python -I collective_eval.py BASE1 > out/colleval_BASE1.log 2>&1 < /dev/null
./venv/bin/python -I final.py S3_KD knnw_yr_40,teh_date_m10,cnt_date > out/final.log 2>&1 < /dev/null
./venv/bin/python -I cat_native.py base date date_dedup > out/cat1.log 2>&1 < /dev/null
echo CHAIN1 DONE >> out/final.log
