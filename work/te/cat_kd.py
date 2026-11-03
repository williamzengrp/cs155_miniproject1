"""CatBoost (teammates' params, thread_count=2, seed k) on 67 feats + BEST_SPEC label features vs plain CatBoost."""
import sys; sys.path.insert(0, './work/te')
import numpy as np
from common import run, cat, OUT, auc_of_oof
import label_features as LF
ref = auc_of_oof(np.load(f'{OUT}/oof_cat_base_s1.npy'))
res = run('CAT_KD_s1', list(LF.BEST_SPEC), offs=(0,), with_test=True, model=cat, ref=ref)
np.savez(f'{OUT}/res1_CAT_KD.npz', oof=res['oof'], aucs=res['aucs'], test_foldavg=res['test_foldavg'])
for v in ('date', 'date_dedup'):
    a = auc_of_oof(np.load(f'{OUT}/oof_cat_{v}_s1.npy')); d = a - ref
    print(f'cat_{v} vs cat_base: AUC {a.mean():.5f} diff {d.mean():+.5f} per-rep {np.round(d, 4)} pos {(d > 0).sum()}/5')
