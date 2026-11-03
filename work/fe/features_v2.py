"""fe agent -- recommended label-free feature set v2.

    from features_v2 import build_features_v2
    X, Xt = build_features_v2(tr, te)      # tr, te as returned by harness.load()

All features are label-free and computed on the train+test pool (so CV and test are treated
identically). See work/fe/log.md for the experiments behind every choice.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from time_features import build_features          # noqa: E402
import fe_lib as L                                 # noqa: E402

# L2D: teammates' groups minus 'albumnn' (redundant with albumnn2) and 'groupagg' (neutral under this CV),
# plus 'detrended' (era-relative audio). 3-seed-set CV (5x5 eligible-row splits, LightGBM teammate params):
# 0.83612 vs 0.83354 for the 67 baseline features, +0.00258, positive in 5/5 repeats (see log.md).
# (L1D = same but keeping groupagg, 75 features: 0.83568, +0.00214, 5/5.)
TEAM_GROUPS = ('trend', 'seasonality', 'density', 'albumnn2', 'detrended')
DROP_COLS = ()
EXTRA_BLOCKS = ()


def build_features_v2(tr, te, team_groups=TEAM_GROUPS, drop_cols=DROP_COLS, extra_blocks=EXTRA_BLOCKS):
    X, Xt = build_features(tr, te, team_groups)
    n = len(tr)
    XP = pd.concat([X, Xt], ignore_index=True)
    XP = XP[[c for c in XP.columns if c not in set(drop_cols)]]
    al, raw, d, akey = L.make_pool(tr, te)
    for b in extra_blocks:
        if b == 'dupcnt':
            B = L.dup_feats(al, raw, d, akey)[['dup_uri_cnt']]
        elif b == 'typ':
            B = L.typicality_feats(al, raw, d, akey)
        elif b == 'resid':
            B = L.resid_feats(al, d)
        else:
            raise ValueError(b)
        XP = pd.concat([XP, B.reset_index(drop=True)], axis=1)
    return XP.iloc[:n].reset_index(drop=True), XP.iloc[n:].reset_index(drop=True)
