"""Final label-free feature set: fe agent's L2D (55) + release-date density ratios (3) = 58 features.
All features are computed on the train+test pool without labels (identical treatment in CV and test)."""
import sys
import pandas as pd
sys.path.insert(0, './work')
sys.path.insert(0, './work/fe')
from features_v2 import build_features_v2
import fe_lib as L


def build_features_final(tr, te):
    X, Xt = build_features_v2(tr, te)
    n = len(tr)
    al, raw, d, akey = L.make_pool(tr, te)
    DR = L.density_ratio_feats(al, raw, d, akey).reset_index(drop=True)
    XP = pd.concat([pd.concat([X, Xt], ignore_index=True), DR], axis=1)
    return XP.iloc[:n].reset_index(drop=True), XP.iloc[n:].reset_index(drop=True)
