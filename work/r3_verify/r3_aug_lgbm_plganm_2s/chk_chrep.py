import sys, numpy as np
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
tr, te, y = H.load()
S = H.make_splits(tr, y, n_rep=5, seed0=1000)
o = H.load_res('chrep_bl')['oof']
mx = 0
for k, (r, fi, vi, el) in enumerate(S):
    c = dict(np.load(f'{WORK}/ch/cache/chfull_std_k{k}.npz'))
    assert len(c['cat_v']) == len(vi)
    mx = max(mx, np.abs(o[r, vi[el]] - (0.6 * c['cat_v'] + 0.4 * c['et_v'])[el]).max())
print('chrep_bl OOF == 0.6*cat_v+0.4*et_v from chfull_std cache on seed0=1000 splits, all 25 folds: maxabs', mx)
