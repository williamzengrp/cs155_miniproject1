"""Test-like subset of the CV (agent `shift`): validation rows whose train date-mates are ALL in the fit part
(no other validation row shares the date) -> like test rows, whose train date-mates are all labelled.
python -I diag2.py SET (std|c9). Prints member / blend-variant AUCs on that subset vs all eligible rows."""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main')
import harness as H
from eval_candidate import MEMBERS
S = sys.argv[1]
tr, te, y = H.load()
n = len(y)
splits = H.make_splits(tr, y, n_rep=5 if S == 'std' else 3, seed0=1000 if S == 'std' else 9000)
date = pd.factorize(tr['track_album_release_date'].astype(str))[0]
uri = pd.factorize(tr['uri'])[0]
names = MEMBERS[S]
res = [H.load_res(nm) for nm in names]
R = res[0]['oof'].shape[0]
tl = np.zeros((R, n), bool)          # test-like: no other val row (different uri) on the same date
KC = int(sys.argv[2]) if len(sys.argv) > 2 else 10
# k = other train rows (different uri) on the same date; P(all in fit) ~ 0.8^k -> IPW weight 0.8^-min(k,KC)
dd = pd.DataFrame({'d': date, 'u': uri})
k_oth = dd.groupby('d')['u'].transform('size').to_numpy() - dd.groupby(['d', 'u'])['u'].transform('size').to_numpy()
ipw = 0.8 ** -np.minimum(k_oth, KC)
for r, fit_idx, val_idx, elig in splits:
    dv = pd.Series(date[val_idx])
    # number of distinct-uri val rows per date
    nv = pd.Series(uri[val_idx]).groupby(date[val_idx]).nunique()
    tl[r, val_idx] = nv.reindex(date[val_idx]).to_numpy() == 1
lab = np.array([1, 1, 1, 1, 1, 0, 0, 0], float)
variants = {'final8': np.ones(8), 'label x2': 1 + lab}
for j, nm in enumerate(names):
    w = np.ones(8); w[j] = 0; variants['-' + nm] = w
acc = {k: [] for k in list(names) + list(variants)}
shares = []
for r in range(R):
    m = ~np.isnan(res[0]['oof'][r])
    O = np.column_stack([rankdata(x['oof'][r, m]) / m.sum() for x in res])
    s = tl[r, m]
    shares.append(s.mean())
    yr = y[m]
    wi = ipw[m][s]
    for j, nm in enumerate(names):
        acc[nm].append((roc_auc_score(yr, O[:, j]), roc_auc_score(yr[s], O[s, j], sample_weight=wi)))
    for k, w in variants.items():
        p = O @ w
        acc[k].append((roc_auc_score(yr, p), roc_auc_score(yr[s], p[s], sample_weight=wi)))
print(f'[{S}] IPW (k clip {KC}) test-like share of eligible rows: {np.mean(shares):.3f}')
b = np.array(acc['final8'])
for k, v in acc.items():
    v = np.array(v)
    if k in names:
        print(f'  {k:22s} all {v[:,0].mean():.5f}  test-like {v[:,1].mean():.5f}')
    else:
        d = v - b
        print(f'  {k:22s} all {v[:,0].mean():.5f}  test-like {v[:,1].mean():.5f}   d_all {d[:,0].mean():+.5f} ({(d[:,0]>0).sum()}/{R})'
              f'  d_testlike {d[:,1].mean():+.5f} ({(d[:,1]>0).sum()}/{R})')
