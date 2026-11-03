"""Save CatBoost + BEST_SPEC label features: CV OOF (1 seed, from cat_kd.py) + test_full (3 seeds, all train)."""
import sys; sys.path.insert(0, './work/te')
import numpy as np
from common import H, full_fit_test, cat, OUT, SEED_OFFS
import label_features as LF
d = np.load(f'{OUT}/res1_CAT_KD.npz')
res = dict(oof=d['oof'], aucs=d['aucs'], auc=float(d['aucs'].mean()), test_foldavg=d['test_foldavg'])
tf = full_fit_test(list(LF.BEST_SPEC), offs=SEED_OFFS, model=cat)
H.save('te_cat_kd', res, tf)
print('saved te_cat_kd CV', round(res['auc'], 5), 'corr(test_full, test_foldavg)', np.corrcoef(tf, res['test_foldavg'])[0, 1].round(4))
