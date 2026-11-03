"""Check that results/r3_chg_hc2_w6.npz == assembly of r3_chg/cache/hc2_k*.npz (w=0.6) and test_full == mean of the
3 full-refit seeds (hc2_full_s*.npz), i.e. no other ingredient sneaked in."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
tr, te, y = H.load()
n = len(tr)
splits = H.make_splits(tr, y, n_rep=5)
R = H.load_res('r3_chg_hc2_w6')
w = 0.6
oof = np.full((5, n), np.nan)
tfa, nt = 0, 0
cover = np.zeros((5, n), int)
for k in range(25):
    c = np.load(f'{WORK}/r3_chg/cache/hc2_k{k}.npz')
    r, fit, val, elig = splits[k]
    assert len(set(fit) & set(val)) == 0
    cover[r, val] += 1
    oof[r, val[elig]] = ((1 - w) * c['cat_v'] + w * c['et_v'])[elig]
    tfa = tfa + (1 - w) * c['cat_t'] + w * c['et_t']; nt += 1
print('each row in exactly one val fold per repeat:', (cover == 1).all())
print('oof identical to saved:', np.array_equal(np.isnan(oof), np.isnan(R['oof'])), np.nanmax(np.abs(oof - R['oof'])))
print('saved aucs:', np.round(R['aucs'], 5), ' recomputed:', np.round([roc_auc_score(y[~np.isnan(o)], o[~np.isnan(o)]) for o in oof], 5))
F = [np.load(f'{WORK}/r3_chg/cache/hc2_full_s{s}.npz') for s in range(3)]
full = np.mean([(1 - w) * f['cat_t'] + w * f['et_t'] for f in F], axis=0)
print('test_full == mean of 3 full-refit seeds:', np.max(np.abs(full - R['test_full'])), ' len', len(R['test_full']), len(te))
print('test_foldavg max diff:', np.max(np.abs(tfa / nt - R['test_foldavg'])))
print('full-refit files keys:', list(F[0].keys()), ' ip lengths (must be n=%d):' % n, [len(f['cat_ip']) for f in F])
print('corr(test_full, test_foldavg):', np.corrcoef(R['test_full'], R['test_foldavg'])[0, 1])
