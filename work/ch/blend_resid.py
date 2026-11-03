"""ch agent: proxy for applying the residual correction to every blend member: add alpha * c_i (c from the
seed-0 LightGBM inner-OOF residuals, same-date kernel bw / shrink) to each v3_* model's OOF probability, then
equal-weight rank blend per repeat. Paired vs the plain v3 5-model blend. (Approximation: each model would
normally use its own inner residuals.)"""
import pickle, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import ResidualCorrector
tr, te, y = H.load()
splits = H.make_splits(tr, y, n_rep=5)
C = pickle.load(open(f'{WORK}/ch/cache/resid_lgbm_std_off0.pkl', 'rb'))
items = []
for k, (r, fi, vi, el) in enumerate(splits):
    rc = ResidualCorrector(tr.iloc[fi], y[fi] - C['folds'][k]['inner'])
    items.append(rc.prepare(tr.iloc[vi]))
M = ['lgbm', 'xgb', 'cat', 'et', 'rf']
O = {m: H.load_res('v3_' + m)['oof'] for m in M}
def blend(alpha, bw, s, members=M):
    aucs = []
    for r in range(5):
        cols = []
        for m in M:
            o = O[m][r].copy()
            if alpha and m in members:
                for k, (rr, fi, vi, el) in enumerate(splits):
                    if rr != r: continue
                    c = ResidualCorrector.correction(items[k], bw, s)
                    o[vi[el]] = np.clip(o[vi[el]] + alpha * c[el], 0, 1)
            mk = ~np.isnan(o)
            cols.append(rankdata(o[mk]) / mk.sum())
        aucs.append(roc_auc_score(y[mk], np.mean(cols, axis=0)))
    return np.array(aucs)
b = blend(0, 1, 1)
print(f'v3 5-model equal rank blend {b.mean():.5f}')
for a, bw, s in [(0.15, 3, 1), (0.15, 3, 3), (0.3, 3, 1), (0.5, 3, 10)]:
    d = blend(a, bw, s) - b
    print(f'| all 5 corrected a={a} bw={bw} s={s} | {b.mean()+d.mean():.5f} | {d.mean():+.5f} | {(d>0).sum()}/5 | {" ".join(f"{x:+.4f}" for x in d)} |')
