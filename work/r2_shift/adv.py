"""Adversarial validation (agent `shift`): can label-free features tell train rows from test rows?
python -I adv.py  -> prints adversarial AUCs, top features, univariate shifts; saves cache/adv.pkl
  (OOF p(test|x) for every train row, from grouped-by-uri CV, 3 seeds; used for density-ratio weights).
Uses only label-free features (58 final features) + uri multiplicity; never Popularity labels."""
import sys, pickle, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
W = './work'
sys.path.insert(0, W)
import harness as H
warnings.filterwarnings('ignore')
C = W + '/r2_shift/cache/'
tr, te, y = H.load()
b = pickle.load(open(C + 'base.pkl', 'rb'))
X, Xt = b['X'], b['Xt']
n, m = len(X), len(Xt)
XA = pd.concat([X, Xt], ignore_index=True)
z = np.r_[np.zeros(n), np.ones(m)].astype(int)
uri_all = pd.concat([tr['uri'], te['uri']], ignore_index=True)
g = pd.factorize(uri_all)[0]
cnt_tr = tr['uri'].map(tr['uri'].value_counts()).to_numpy()
cnt_te = te['uri'].map(te['uri'].value_counts()).to_numpy()
mult = np.r_[cnt_tr, cnt_te]
print(f'train rows with a within-train copy: {(cnt_tr > 1).mean():.3f}; test rows with a within-test copy: {(cnt_te > 1).mean():.3f}')


def adv_oof(cols, rows, seeds=(0, 1, 2), params=None):
    p = params or dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0)
    rows = np.asarray(rows)
    oof = np.zeros(len(rows))
    imp = np.zeros(len(cols))
    for s in seeds:
        for a, bb in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(rows, z[rows], g[rows]):
            mdl = LGBMClassifier(**p, random_state=s, n_jobs=2, verbose=-1, importance_type='gain')
            mdl.fit(XA.iloc[rows[a]][cols], z[rows[a]])
            oof[bb] += mdl.predict_proba(XA.iloc[rows[bb]][cols])[:, 1] / len(seeds)
            imp += mdl.feature_importances_
    return oof, pd.Series(imp / imp.sum(), index=cols).sort_values(ascending=False)


cols = list(X.columns)
allrows = np.arange(n + m)
oof_all, imp_all = adv_oof(cols, allrows)
print(f'\n[ADV all rows, 58 feats] AUC {roc_auc_score(z, oof_all):.4f}')
print(imp_all.head(12).round(3).to_string())

# singletons only: train rows with no train copy vs test rows with no test copy
sing = np.r_[cnt_tr == 1, cnt_te == 1]
rows_s = np.nonzero(sing)[0]
oof_s, imp_s = adv_oof(cols, rows_s)
print(f'\n[ADV singletons only, 58 feats] AUC {roc_auc_score(z[rows_s], oof_s):.4f}  (n={len(rows_s)})')
print(imp_s.head(12).round(3).to_string())

# without the duplicate-sensitive features
dupcols = ['a_global_nn_dist', 'a_global_nn5_dist', 'a_n_close10', 'a_n_close20', 'a_n_close30', 'a_nn2_dist',
           'n_same_date_audio']
cols_nd = [c for c in cols if c not in dupcols]
oof_nd, imp_nd = adv_oof(cols_nd, allrows)
print(f'\n[ADV all rows, no dup-sensitive feats] AUC {roc_auc_score(z, oof_nd):.4f}')
print(imp_nd.head(10).round(3).to_string())
oof_snd, imp_snd = adv_oof(cols_nd, rows_s)
print(f'[ADV singletons, no dup-sensitive feats] AUC {roc_auc_score(z[rows_s], oof_snd):.4f}')
print(imp_snd.head(10).round(3).to_string())

# univariate shift: AUC of each feature for train-singletons vs test-singletons, and all rows
print('\nunivariate |AUC-0.5| (all rows / singletons), top 15:')
uni = []
for c in cols:
    v = XA[c].fillna(XA[c].median()).to_numpy()
    uni.append((c, roc_auc_score(z, v) - 0.5, roc_auc_score(z[rows_s], v[rows_s]) - 0.5))
uni = pd.DataFrame(uni, columns=['feat', 'all', 'sing']).set_index('feat')
print(uni.reindex(uni['all'].abs().sort_values(ascending=False).index).head(15).round(4).to_string())

# relation between train p(test|x) and labels / multiplicity (descriptive only)
p_tr = oof_all[:n]
print('\ntrain p(test|x) by multiplicity:', {k: round(float(p_tr[cnt_tr == k].mean()), 4) for k in (1, 2, 3)})
q = pd.qcut(p_tr, 5, labels=False, duplicates='drop')
print('High rate by p(test|x) quintile (train rows):', np.round([y[q == i].mean() for i in range(q.max() + 1)], 3))
pickle.dump(dict(oof_all=oof_all, oof_s=oof_s, rows_s=rows_s, oof_nd=oof_nd, oof_snd=oof_snd, uni=uni,
                 imp_all=imp_all, imp_s=imp_s), open(C + 'adv.pkl', 'wb'))
print('saved', C + 'adv.pkl')
