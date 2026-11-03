"""Final runs / saves (agent te).

usage: python final.py <tag67> <feat1,feat2,...> [--skip67]
 1. 67 features + label features: CV result from confirm.py (out/res3_<tag67>.npz, 3 seeds averaged per
    fold) + test_full = models trained on ALL train rows (label features of train rows = inner OOF over
    train grouped by uri; test rows = all train labels), mean of 3 seeds -> H.save('te_lgbm_best').
 2. v2 (fe agent's 55 label-free features): baseline vs + label features, SAME 3 seeds and splits ->
    paired diff; H.save('te_lgbm_best_v2') with test_full as above.
"""
import sys

sys.path.insert(0, './work/te')
import numpy as np  # noqa: E402
from common import H, run, full_fit_test, OUT, SEED_OFFS, base_frame  # noqa: E402

tag67, feats = sys.argv[1], sys.argv[2]
spec = [f for f in feats.split(',') if f]
if '--skip67' not in sys.argv:
    d = np.load(f'{OUT}/res3_{tag67}.npz')
    res = dict(oof=d['oof'], aucs=d['aucs'], auc=float(d['aucs'].mean()), test_foldavg=d['test_foldavg'])
    tf = full_fit_test(spec, offs=SEED_OFFS, base='67')
    H.save('te_lgbm_best', res, tf)
    print('saved te_lgbm_best', spec, 'CV', round(res['auc'], 5), 'corr(test_full, test_foldavg)',
          np.corrcoef(tf, res['test_foldavg'])[0, 1].round(4), flush=True)

print('v2 base shape', base_frame('v2').shape, flush=True)
rb = run('V2_base_s3', [], offs=SEED_OFFS, with_test=True, base='v2')
np.savez(f'{OUT}/res3_V2_base_s3.npz', oof=rb['oof'], aucs=rb['aucs'], test_foldavg=rb['test_foldavg'])
rv = run('V2_' + tag67, spec, offs=SEED_OFFS, with_test=True, base='v2', ref=rb['aucs'])
np.savez(f'{OUT}/res3_V2_{tag67}.npz', oof=rv['oof'], aucs=rv['aucs'], test_foldavg=rv['test_foldavg'])
tf2 = full_fit_test(spec, offs=SEED_OFFS, base='v2')
H.save('te_lgbm_best_v2', rv, tf2)
print('saved te_lgbm_best_v2; corr(test_full, test_foldavg)', np.corrcoef(tf2, rv['test_foldavg'])[0, 1].round(4))
