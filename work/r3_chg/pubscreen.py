"""pubsim conditional means for RAW (uncorrected) first-repeat screen blends, in memory (no result files):
python -I pubscreen.py TAG [TAG ...]   (TAG = r3_chg cache tag with *_raw_k0..4; 'rep' = plain replica raw)"""
import sys
import numpy as np
from scipy.stats import norm
WORK = './work'
sys.path.insert(0, WORK + '/main')
import pubsim as PS
H = PS.H
PS.TARGET = 0.873
tr, te, y = H.load(); n = len(y)
sp = H.make_splits(tr, y, n_rep=1)
W = float(sys.argv[1]); tags = sys.argv[2:]
orig = H.load_res
mem = {}
for tag in tags:
    o = np.full((5, n), np.nan)
    for k in range(5):
        _, fi, vi, el = sp[k]
        p = f'{WORK}/ch/cache/chfull_std_k{k}.npz' if tag == 'rep' else f'{WORK}/r3_chg/cache/{tag}_raw_k{k}.npz'
        c = np.load(p)
        o[0, vi[el]] = ((1 - W) * c['cat_raw_v'] + W * c['et_raw_v'])[el]
    mem['scr_' + tag] = o
H.load_res = lambda name: {'oof': mem[name]} if name in mem else orig(name)
cands = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
for tag in tags:
    cands[f'{tag}@1'] = [('scr_' + tag, 1.0)]
    cands[f'ch.5+{tag}@.5'] = [('ch', 0.5), ('scr_' + tag, 0.5)]
res = PS.run(cands, 431, verbose=False, extra_obs={'G40': 0.86977})
g = 431 / 3521; fac = np.sqrt((1 + g) / (1 - g))
for name, cvg, cm, sd, p in res:
    pc = 1 - norm.cdf((PS.TARGET - PS.CH_PUB - cm) / (sd * fac))
    print(f'w_et={W} {name:24s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  P {pc:.3f}')
