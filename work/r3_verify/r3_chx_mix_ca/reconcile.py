"""Check that results/r3_chx_mix_ca.npz equals the mix.py assembly of the audited cache files (c_k*, a_k*/et_a_k*,
c_full_s*, et_a_full_s*) and report OOF coverage / AUCs.  Read-only."""
import sys, os
sys.dont_write_bytecode = True
import numpy as np
WORK = './work'
C_ = WORK + '/r3_chx/cache'
sys.path.insert(0, WORK)
import harness as H
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
R = H.load_res('r3_chx_mix_ca')
print({k: v.shape for k, v in R.items()})


def ep(k):
    p = f'{C_}/et_a_k{k}.npz'
    return p if os.path.exists(p) else f'{C_}/a_k{k}.npz'


oof = np.full((5, n), np.nan); tst = np.zeros(m)
for k in range(25):
    r, fit, val, elig = splits[k]
    c, e = np.load(f'{C_}/c_k{k}.npz'), np.load(ep(k))
    assert len(c['cat_v']) == len(val) == len(e['et_v'])
    oof[r, val[elig]] = (0.6 * c['cat_v'] + 0.4 * e['et_v'])[elig]
    tst += (0.6 * c['cat_t'] + 0.4 * e['et_t']) / 25
fl = [0.6 * np.load(f'{C_}/c_full_s{s}.npz')['cat_t'] + 0.4 * np.load(f'{C_}/et_a_full_s{s}.npz')['et_t'] for s in range(6)]
full = np.mean(fl, axis=0)
same = lambda a, b: np.allclose(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9), atol=1e-12)
print('oof equal', same(oof, R['oof']), ' test_foldavg equal', same(tst, R['test_foldavg']),
      ' test_full equal', same(full, R['test_full']))
print('non-NaN per repeat', (~np.isnan(R['oof'])).sum(1), ' per-rep AUC', np.round(R['aucs'], 5),
      ' mean', round(float(np.nanmean(R['aucs'])), 5))
print('corr(test_full, test_foldavg)', round(float(np.corrcoef(R['test_full'], R['test_foldavg'])[0, 1]), 4))
# each eligible OOF row: its uri never in that fold's fit rows (sanity of the eligibility mask)
uri = tr['uri'].to_numpy()
bad = 0
for k in range(25):
    r, fit, val, elig = splits[k]
    bad += len(set(uri[val[elig]]) & set(uri[fit]))
print('eligible-val uris present in fit rows (should be 0):', bad)
