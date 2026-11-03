"""Add-one-member check under the plain CV and the IPW test-like CV (agent `shift`).
python -I diag3.py SET cand1,cand2,...   (candidates = existing result names on that split set)"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main')
import harness as H
from eval_candidate import MEMBERS
S = sys.argv[1]; cands = sys.argv[2].split(',')
tr, te, y = H.load(); n = len(y)
splits = H.make_splits(tr, y, n_rep=5 if S == 'std' else 3, seed0=1000 if S == 'std' else 9000)
date = pd.factorize(tr['track_album_release_date'].astype(str))[0]; uri = pd.factorize(tr['uri'])[0]
dd = pd.DataFrame({'d': date, 'u': uri})
k_oth = dd.groupby('d')['u'].transform('size').to_numpy() - dd.groupby(['d', 'u'])['u'].transform('size').to_numpy()
ipw = 0.8 ** -np.minimum(k_oth, 10)
R = 5 if S == 'std' else 3
tl = np.zeros((R, n), bool)
for r, fit_idx, val_idx, elig in splits:
    nv = pd.Series(uri[val_idx]).groupby(date[val_idx]).nunique()
    tl[r, val_idx] = nv.reindex(date[val_idx]).to_numpy() == 1
base = [H.load_res(x) for x in MEMBERS[S]]
def ev(res):
    a, b = [], []
    for r in range(R):
        m = ~np.isnan(res[0]['oof'][r])
        p = np.column_stack([rankdata(x['oof'][r, m]) / m.sum() for x in res]).mean(1)
        s = tl[r, m]; yr = y[m]
        a.append(roc_auc_score(yr, p)); b.append(roc_auc_score(yr[s], p[s], sample_weight=ipw[m][s]))
    return np.array(a), np.array(b)
A0, B0 = ev(base)
for c in cands:
    rc = H.load_res(c)
    A, B = ev(base + [rc])
    da, db = A - A0, B - B0
    print(f'[{S}] +{c:22s} standalone {rc["aucs"].mean():.5f}  d_all {da.mean():+.5f} ({(da>0).sum()}/{R})  d_testlike {db.mean():+.5f} ({(db>0).sum()}/{R})')
