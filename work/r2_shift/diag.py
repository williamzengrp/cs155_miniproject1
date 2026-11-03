"""Diagnostics on saved OOFs (agent `shift`): does a test-like weighting of the CV change member/blend conclusions?
python -I diag.py SET   (SET = std | c9)
Weights for eligible validation rows (label-free w.r.t. validation labels):
  w_cnt : p_test(cnt_date bucket) / p_val(cnt_date bucket)  -- test rows see all train labels, CV val rows only 80%
  w_adv : adversarial density ratio p(test|x)/(1-p(test|x)) from adv.py (label-free features, grouped CV)
Prints weighted / unweighted AUC of each member, final8, and leave-one-member-out / GBM-heavy variants."""
import sys, pickle
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main')
import harness as H
from eval_candidate import MEMBERS
S = sys.argv[1]
assert S in ('std', 'c9')
C = W + '/r2_shift/cache/'
tr, te, y = H.load()
n = len(y)
lf = pickle.load(open(C + ('lf_1000_5.pkl' if S == 'std' else 'lf_9000_3.pkl'), 'rb'))
full = pickle.load(open(C + 'full.pkl', 'rb'))
adv = pickle.load(open(C + 'adv.pkl', 'rb'))
p_adv = np.clip(adv['oof_all'][:n], 0.02, 0.98)
w_adv_row = p_adv / (1 - p_adv)
names = MEMBERS[S]
res = [H.load_res(nm) for nm in names]
R = res[0]['oof'].shape[0]
cnt_val = np.full((R, n), np.nan)
for f in lf:
    cnt_val[f['r'], f['val_idx'][f['elig']]] = f['F_val']['cnt_date'].to_numpy()[f['elig']]
cnt_te = full[0][1]['cnt_date'].to_numpy()
bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 12.5, 1e9]
pt = np.histogram(cnt_te, bins)[0] / len(cnt_te)
pv = np.histogram(cnt_val[~np.isnan(cnt_val)], bins)[0] / (~np.isnan(cnt_val)).sum()
print('cnt_date bucket shares  test:', np.round(pt, 3), '\n                         val :', np.round(pv, 3))
ratio = pt / pv
print('ratio:', np.round(ratio, 2))


def blend_ranks(r, m, idx_w=None):
    O = np.column_stack([rankdata(x['oof'][r, m]) / m.sum() for x in res])
    return O


rows = []
for r in range(R):
    m = ~np.isnan(res[0]['oof'][r])
    yr = y[m]
    wc = ratio[np.digitize(cnt_val[r, m], bins) - 1]
    wa = w_adv_row[m]
    O = blend_ranks(r, m)
    out = {}
    for j, nm in enumerate(names):
        out[nm] = (roc_auc_score(yr, O[:, j]), roc_auc_score(yr, O[:, j], sample_weight=wc),
                   roc_auc_score(yr, O[:, j], sample_weight=wa))
    variants = {'final8': np.ones(len(names))}
    for j, nm in enumerate(names):
        w = np.ones(len(names)); w[j] = 0; variants[f'-{nm}'] = w
    g = np.array([1 if nm in names[:5] else 0 for nm in names], float)
    variants['label-members x2'] = 1 + g
    variants['label-members only'] = g
    for k, wv in variants.items():
        p = O @ wv
        out[k] = (roc_auc_score(yr, p), roc_auc_score(yr, p, sample_weight=wc), roc_auc_score(yr, p, sample_weight=wa))
    rows.append(out)
keys = list(rows[0].keys())
A = {k: np.array([rw[k] for rw in rows]) for k in keys}   # (R, 3)
base = A['final8']
print(f'\n[{S}] {"":28s} {"unweighted":>10s} {"w_cnt":>8s} {"w_adv":>8s}   (blend variants: diff vs final8, pos)')
for k in keys:
    a = A[k].mean(0)
    if k in names:
        print(f'  {k:28s} {a[0]:10.5f} {a[1]:8.5f} {a[2]:8.5f}')
    else:
        d = A[k] - base
        print(f'  {k:28s} {a[0]:10.5f} {a[1]:8.5f} {a[2]:8.5f}   d_unw {d[:,0].mean():+.5f} ({(d[:,0]>0).sum()}/{R})'
              f'  d_cnt {d[:,1].mean():+.5f} ({(d[:,1]>0).sum()}/{R})  d_adv {d[:,2].mean():+.5f} ({(d[:,2]>0).sum()}/{R})')
