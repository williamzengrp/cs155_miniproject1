"""Auditor: re-run the full refit (all train rows) of variant VARIANT seed index s and compare with the builder's
varfull cache. usage: python -I -B full_check.py VARIANT S_IDX"""
import sys, time
sys.dont_write_bytecode = True
W = './work'
variant, s = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, W + '/r3_chp')
sys.argv = ['chp_var.py', 'noop', variant]
t0 = time.time()
import chp_var as V
import numpy as np
fit = np.arange(V.n)
S = V.SEED0 + 1000 * (s + 1)
mats, al = V.matrices(fit, [], 42 + (1000 * (s + 1) if 'tev' in V.toks else 0), 1000 * (s + 1))
print('full fit rows', len(fit), 'X fit shape', mats[0].shape, 'test shape', mats[2].shape, flush=True)
out = V.outer(fit, mats, al, S)
c = dict(np.load(f'{W}/r3_chp/cache/varfull_{variant}_s{s}.npz'))
for key in ('cat_t_2000', 'et_t'):
    print('full', variant, s, key, 'max|diff| vs builder cache', float(np.abs(out[key] - c[key]).max()), flush=True)
print('done', round(time.time() - t0, 1), flush=True)
