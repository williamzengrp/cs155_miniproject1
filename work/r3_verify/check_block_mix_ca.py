import sys, os
sys.dont_write_bytecode = True
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
B = dict(np.load(WORK + '/results/r3_chx_mix_ca.npz', allow_pickle=True))
R = dict(np.load(WORK + '/results/chrep_bl.npz', allow_pickle=True))
print('keys block', list(B), 'keys ref', list(R))
o, orf = B['oof'], R['oof']
print('oof shapes', o.shape, orf.shape)
for r in range(5):
    print(f'rep {r}: nonNaN block {np.isfinite(o[r]).sum()} ref {np.isfinite(orf[r]).sum()} '
          f'pattern equal {bool((np.isfinite(o[r]) == np.isfinite(orf[r])).all())} '
          f'AUC block {roc_auc_score(y[np.isfinite(o[r])], o[r][np.isfinite(o[r])]):.5f} '
          f'ref {roc_auc_score(y[np.isfinite(orf[r])], orf[r][np.isfinite(orf[r])]):.5f}')
print('stored aucs', B.get('aucs'), 'mean', np.nanmean(B['aucs']), 'ref', R['aucs'], np.mean(R['aucs']))
# recompute from cache
C = WORK + '/r3_chx/cache'
mx = 0.0; srcs = set()
for k in range(25):
    r, fit, val, elig = splits[k]
    assert not set(fit) & set(val)
    c = np.load(f'{C}/c_k{k}.npz')
    ep = f'{C}/et_a_k{k}.npz' if os.path.exists(f'{C}/et_a_k{k}.npz') else f'{C}/a_k{k}.npz'
    srcs.add(os.path.basename(ep).split('_k')[0])
    e = np.load(ep)
    mx = max(mx, np.max(np.abs(o[r, val[elig]] - (0.6 * c['cat_v'] + 0.4 * e['et_v'])[elig])))
    # elig = val rows whose uri is not in fit rows
    u = tr['uri'].astype(str).to_numpy()
    assert (elig == ~np.isin(u[val], u[fit])).all()
print('max |oof - recomputed from cache|', mx, 'ET sources', srcs)
fl = []
for s in range(10):
    p1, p2 = f'{C}/c_full_s{s}.npz', f'{C}/et_a_full_s{s}.npz'
    if os.path.exists(p1) and os.path.exists(p2):
        fl.append(0.6 * np.load(p1)['cat_t'] + 0.4 * np.load(p2)['et_t'])
full = np.mean(fl, axis=0)
tf = B['test_full']
print('n full seeds', len(fl), 'max |test_full - recomputed|', np.max(np.abs(tf - full)), 'len', len(tf))
print('corr(test_full, test_foldavg)', np.corrcoef(tf, B['test_foldavg'])[0, 1], 'identical?', np.allclose(tf, B['test_foldavg']))
print('spearman(test_full, charlie file)', spearmanr(tf, R['test_full'])[0])
# per-seed spread of full refits (should differ across seeds -> genuine refits)
print('full-seed pairwise corr min', min(np.corrcoef(np.array(fl))[np.triu_indices(len(fl), 1)]))
# full refit cache: fit-row inner preds cover all n train rows
for s in range(6):
    d1, d2 = np.load(f'{C}/c_full_s{s}.npz'), np.load(f'{C}/et_a_full_s{s}.npz')
    print(s, 'c_full keys', sorted(d1.files), 'len ip', len(d1['cat_ip']), '| et_a_full keys', sorted(d2.files), len(d2['et_ip']))
