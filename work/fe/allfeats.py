"""Build (and cache) one pool-level table with all teammate groups + all fe candidate blocks."""
import os, sys
sys.path.insert(0, './work/fe')
from probe import *
from fe_lib import *

FULL = ['trend', 'seasonality', 'density', 'albumnn', 'albumnn2', 'groupagg']
PKL = CACHE + '/allfeats.pkl'


def build_all():
    if os.path.exists(PKL):
        return pd.read_pickle(PKL)
    cols, Xa, Xta = groups_columns()
    XP = pd.concat([Xa, Xta], ignore_index=True)
    al, raw, d, akey = make_pool(tr, te)
    blocks = [dup_feats(al, raw, d, akey), dedup_group_feats(al, raw, d, akey),
              cluster_feats(al, 8), cluster_feats(al, 16)[['km16_lab', 'km16_dmin']], pca_feats(al, 5),
              gmm_feats(al, 8), resid_feats(al, d), release_day_feats(d, raw), album_feats(al, raw, akey, t=2.5)[0],
              typicality_feats(al, raw, d, akey), density_ratio_feats(al, raw, d, akey),
              era_relative_feats(al, raw, d, akey, w=2.0, kinds=('rank',))]
    XP = pd.concat([XP] + [b.reset_index(drop=True) for b in blocks], axis=1)
    XP.attrs['groups'] = cols
    pd.to_pickle((XP, cols), PKL)
    return XP, cols


def group_cols():
    XP, cols = build_all() if not os.path.exists(PKL) else pd.read_pickle(PKL)
    return XP, cols


def sel(cols, groups):
    return cols['base'] + [c for g in groups for c in cols[g]]
