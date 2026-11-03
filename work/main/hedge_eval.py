"""Charlie-replica OOF (0.6 Cat + 0.4 ET, from work/ch/cache) saved as results, then CV of hedge blends:
v2 blend (8 members) mixed with the replica at weight w (rank average of the two).
python -I hedge_eval.py"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, './work')
import harness as H
CACHE = H.HERE + '/ch/cache'
CH_FILE = './reference/submission_inverted.csv'   # Charlie's submitted file (in this branch)
tr, te, y = H.load()

# Charlie's own test predictions: his file is 1 - P(High)
ch = pd.read_csv(CH_FILE)
assert (ch['ID'].to_numpy() == te['ID'].to_numpy()).all()
ch_test = 1 - ch['Popularity_Type'].to_numpy()

for kind, name, nrep, seed0 in (('std', 'chrep_bl', 5, 1000), ('conf', 'conf_chrep_bl', 3, 9000)):
    splits = H.make_splits(tr, y, n_rep=nrep, seed0=seed0)
    import os
    nk = sum(os.path.exists(f'{CACHE}/chfull_{kind}_k{k}.npz') for k in range(len(splits)))
    nrep = nk // 5   # the c9 replica run was stopped early: keep only complete repeats
    splits = [s for s in splits if s[0] < nrep]
    C = [dict(np.load(f'{CACHE}/chfull_{kind}_k{k}.npz')) for k in range(len(splits))]
    oof = np.full((nrep, len(y)), np.nan)
    tst = 0
    for k, (r, fit_idx, val_idx, elig) in enumerate(splits):
        oof[r, val_idx[elig]] = (0.6 * C[k]['cat_v'] + 0.4 * C[k]['et_v'])[elig]
        if 'cat_t' in C[k]:   # the c9 cache holds validation predictions only
            tst = tst + 0.6 * C[k]['cat_t'] + 0.4 * C[k]['et_t']
    aucs = np.array([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof])
    H.save(name, dict(oof=oof, aucs=aucs, test_foldavg=tst / len(splits)), ch_test)
    print(f'{name}: AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)}')

V2 = {'std': ['graph_ds_lgbm', 'graph_ds_xgb', 'zoo_cat_alb', 'fkd_et', 'b2_et', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj'],
      'c9': ['graph_c9_ds_lgbm', 'graph_c9_ds_xgb', 'zoo_c9_cat_alb', 'conf_fkd_et', 'conf_base_et', 'conf_mlp_ple',
             'conf_ch_et', 'conf_ch_lgbm']}
F8 = {'std': ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj'],
      'c9': ['conf_fkd_lgbm', 'conf_fkt_xgb', 'conf_fkt_cat', 'conf_fkd_et', 'conf_fkd_rf', 'conf_mlp_ple',
             'conf_ch_et', 'conf_ch_lgbm']}
REP = {'std': 'chrep_bl', 'c9': 'conf_chrep_bl'}


def rk(o):
    out = np.full_like(o, np.nan)
    m = ~np.isnan(o)
    out[m] = rankdata(o[m]) / m.sum()
    return out


for S in ('std', 'c9'):
    O = {n: H.load_res(n)['oof'] for n in V2[S] + F8[S] + [REP[S]]}
    R = O[REP[S]].shape[0]   # c9: only the replica's complete repeats
    res = {}
    for r in range(R):
        m = ~np.isnan(O[REP[S]][r])
        f8 = np.mean([rk(O[n][r]) for n in F8[S]], 0)
        v2 = np.mean([rk(O[n][r]) for n in V2[S]], 0)
        c = rk(O[REP[S]][r])
        cand = {'final8': f8, 'v2': v2, 'v2+ch(1/9)': (8 * v2 + c) / 9}
        for w in (0.25, 0.4, 0.5):
            cand[f'v2*(1-{w})+ch*{w}'] = (1 - w) * rk(v2) + w * c
        cand['ch alone'] = c
        for k, p in cand.items():
            res.setdefault(k, []).append(roc_auc_score(y[m], p[m]))
    base = np.array(res['v2'])
    print(f'--- {S}')
    for k, a in res.items():
        a = np.array(a)
        print(f'  {k:22s} {a.mean():.5f}  vs v2 {np.mean(a - base):+.5f} pos {(a > base).sum()}/{R}')
