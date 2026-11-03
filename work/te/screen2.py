"""Screen 2 (1 seed): k for year-window kNN labels, album-level TE (label-free album clusters), combos."""
import sys; sys.path.insert(0, './work/te')
from common import run
V = [
    ('E1_knnw_yr_40_k200', ['knnw_yr_40']),
    ('E2_knnw_yr_80', ['knnw_yr_80']),
    ('E3_teh_alb_m10+cnt', ['teh_alb_m10', 'cnt_alb']),
    ('E4_teh_alb_m10+cnt+size', ['teh_alb_m10', 'cnt_alb', 'albsize']),
    ('E5_knnw40+teh_date', ['knnw_yr_40', 'teh_date_m10', 'cnt_date']),
    ('E6_knnw40+teh_alb', ['knnw_yr_40', 'teh_alb_m10', 'cnt_alb']),
]
only = sys.argv[1:]
for tag, spec in V:
    if not only or tag.split('_')[0] in only:
        run(tag, spec, offs=(0,))
