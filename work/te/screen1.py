"""Screen 1 (1 seed): target encodings of date / ym / year and neighbour-label features on top of 67 feats."""
import sys; sys.path.insert(0, './work/te')
from common import run
V = [
    ('A1_te_date_m3+cnt', ['te_date_m3', 'cnt_date']),
    ('A2_te_date_m1+cnt', ['te_date_m1', 'cnt_date']),
    ('A3_te_date_m10+cnt', ['te_date_m10', 'cnt_date']),
    ('A4_teh_date_m3+cnt', ['teh_date_m3', 'cnt_date']),
    ('A5_teh_date_m10+cnt', ['teh_date_m10', 'cnt_date']),
    ('B1_te_ym_m10+cnt', ['te_ym_m10', 'cnt_ym']),
    ('B2_te_year_m10', ['te_year_m10']),
    ('C1_nbd_s1_m1', ['nbd_s1_m1']),
    ('C2_knn_glob_20', ['knn_glob_20']),
    ('C3_knn_yr_20', ['knn_yr_20']),
    ('C4_knnw_yr_20', ['knnw_yr_20']),
    ('C5_knnw_yr_10', ['knnw_yr_10']),
    ('C6_knnw_yr_40', ['knnw_yr_40']),
]
only = sys.argv[1:]
for tag, spec in V:
    if not only or tag.split('_')[0] in only:
        run(tag, spec, offs=(0,))
