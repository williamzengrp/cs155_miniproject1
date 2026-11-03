"""ch agent: screen charlie-branch label-free feature blocks on top of the 58 final features.
LightGBM teammate params, n_jobs=2, random_state = fold k + OFF; standard 5x5 splits; paired vs the 58-feature
model with the same seeds (OFF=0 reference = results/v3_lgbm; other offsets computed + cached here).

usage: python -I screen.py OFF blk1 [blk2 ...]      (a block may be 'a+b' to add several blocks together)
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/main')
sys.path.insert(0, WORK + '/ch')
import harness as H
from features_final import build_features_final
from ch_feats import charlie_pool
warnings.filterwarnings('ignore')

OFF = int(sys.argv[1])
CACHE = WORK + '/ch/cache'
tr, te, y = H.load()
n = len(tr)
X, Xt = build_features_final(tr, te)
F, BL = charlie_pool(tr, te)
P = json.load(open(WORK + '/best_params_v3.json'))['lgbm']
splits = H.make_splits(tr, y, n_rep=5)


def run(tag, A):
    path = f'{CACHE}/scr_{tag}_off{OFF}.npz'
    if os.path.exists(path):
        d = np.load(path)
        return dict(oof=d['oof'], aucs=d['aucs'], auc=float(d['aucs'].mean()))
    t = time.time()

    def fp(fit_idx, val_idx, k):
        m = LGBMClassifier(**P, random_state=k + OFF, n_jobs=2, verbose=-1).fit(A.iloc[fit_idx], y[fit_idx])
        return m.predict_proba(A.iloc[val_idx])[:, 1], None
    res = H.run_cv(fp, splits, y, tag=f'{tag} off{OFF}')
    np.savez(path, oof=res['oof'], aucs=res['aucs'])
    print(f'   ({time.time() - t:.0f}s)', flush=True)
    return res


if OFF == 0:
    ref = H.load_res('v3_lgbm')
    ref = dict(oof=ref['oof'], aucs=ref['aucs'], auc=float(ref['aucs'].mean()))
else:
    ref = run('base58', X)
print(f'REF 58 off{OFF}: {ref["auc"]:.5f} per-rep {np.round(ref["aucs"], 4)}', flush=True)
for blk in sys.argv[2:]:
    cols = []
    for b in blk.split('+'):
        cols += [c for c in BL[b] if c not in cols]
    A = pd.concat([X, F[cols].iloc[:n].reset_index(drop=True)], axis=1)
    res = run(blk, A)
    d = res['aucs'] - ref['aucs']
    print(f'| 58+{blk} | {A.shape[1]} | {res["auc"]:.5f} | {d.mean():+.5f} | {(d > 0).sum()}/5 | '
          f'{" ".join(f"{x:+.4f}" for x in d)} |', flush=True)
