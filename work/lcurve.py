import sys, json, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
sys.path.insert(0, './work')
import harness as H
from time_features import build_features, FEATURE_GROUPS
warnings.filterwarnings('ignore')
tr, te, y = H.load()
X, Xt = build_features(tr, te, FEATURE_GROUPS)
P = json.load(open(H.HERE + '/best_params_v3.json'))['lgbm']
splits = H.make_splits(tr, y, n_rep=2)
rng = np.random.default_rng(0)
fracs = [0.4, 0.6, 0.8, 1.0]
res = {f: [] for f in fracs}
for r in range(2):
    oof = {f: np.full(len(y), np.nan) for f in fracs}
    for (rr, fit_idx, val_idx, elig) in [s for s in splits if s[0] == r]:
        for f in fracs:
            sub = rng.choice(fit_idx, int(len(fit_idx) * f), replace=False) if f < 1 else fit_idx
            m = LGBMClassifier(**P, random_state=0, n_jobs=1, verbose=-1).fit(X.iloc[sub], y[sub])
            oof[f][val_idx[elig]] = m.predict_proba(X.iloc[val_idx[elig]])[:, 1]
    for f in fracs:
        mk = ~np.isnan(oof[f]); res[f].append(roc_auc_score(y[mk], oof[f][mk]))
for f in fracs:
    print(f'train frac of fit part {f:.1f} (= {f*0.8:.2f} of train): AUC {np.mean(res[f]):.4f}  {np.round(res[f],4)}')
