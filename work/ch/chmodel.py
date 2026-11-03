"""ch agent: charlie's final label-free feature view (PROJECT = basic + density + secondary, 115 cols, own
re-implementation in ch_feats.py) as a separate model for the rank blend. Its date target encodings are NOT
included (label features are the te agent's area).

  et   : charlie's ExtraTrees config (900 trees, depth 15, min_leaf 2, max_features 0.1827) with charlie's
         1/count-per-uri sample weights, fold-local median imputation.
  etnw : same without weights.
  lgbm : teammate LightGBM params on the same 115 columns.
usage: python -I chmodel.py MODEL [OFF] [--save]
"""
import json, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_feats import charlie_pool
warnings.filterwarnings('ignore')

MODEL = sys.argv[1]
OFF = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else 0
tr, te, y = H.load()
n = len(tr)
F, BL = charlie_pool(tr, te)
XP = F[BL['PROJECT']].astype(np.float32)
X, Xt = XP.iloc[:n].reset_index(drop=True), XP.iloc[n:].reset_index(drop=True)
P = json.load(open(WORK + '/best_params_v3.json'))
uri = tr['uri'].to_numpy()
splits = H.make_splits(tr, y, n_rep=5)
print('features', X.shape, flush=True)


def weights(idx):
    if MODEL != 'et':
        return None
    u = pd.Series(uri[idx])
    w = 1.0 / u.map(u.value_counts()).to_numpy(float)
    return w / w.mean()


def fit_predict_rows(fit_idx, apply_frames, seed):
    A = X.iloc[fit_idx]
    if MODEL in ('et', 'etnw'):
        med = A.median()
        m = ExtraTreesClassifier(n_estimators=900, max_depth=15, min_samples_leaf=2, max_features=0.1827077108658931,
                                 n_jobs=2, random_state=seed)
        m.fit(A.fillna(med), y[fit_idx], sample_weight=weights(fit_idx))
        return [m.predict_proba(B.fillna(med))[:, 1] for B in apply_frames]
    m = LGBMClassifier(**P['lgbm'], random_state=seed, n_jobs=2, verbose=-1).fit(A, y[fit_idx])
    return [m.predict_proba(B)[:, 1] for B in apply_frames]


t = time.time()
res = H.run_cv(lambda fi, vi, k: tuple(fit_predict_rows(fi, [X.iloc[vi], Xt], k + OFF)), splits, y,
               tag=f'ch {MODEL} PROJECT off{OFF}')
print(f'   ({time.time() - t:.0f}s)', flush=True)
if '--save' in sys.argv and OFF != 0:          # seed-robustness runs: OOF only
    H.save(f'ch_{MODEL}_proj_off{OFF}', res)
    print('saved', f'ch_{MODEL}_proj_off{OFF}')
elif '--save' in sys.argv:
    full = np.mean([fit_predict_rows(np.arange(n), [Xt], s)[0] for s in (0, 1, 2)], axis=0)
    name = f'ch_{MODEL}_proj'
    H.save(name, res, full)
    print('saved', name, ' corr(test_full, test_foldavg) =', np.corrcoef(full, res['test_foldavg'])[0, 1].round(4))
