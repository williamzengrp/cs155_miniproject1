"""Small blend grid (pubsim conditional model, same conditioning as pubcheck.py) over Charlie's file, an r3_chx block X,
zoo_cat_alb and graph_hc2_lgbm.  python -I pubgrid.py X [X ...]"""
import sys, itertools
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS
PS.TARGET = 0.873
cands = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
for X in sys.argv[1:]:
    for wx, wc, wa, wg in itertools.product((0.3, 0.45, 0.6, 0.75), (0.0, 0.1, 0.2, 0.3), (0.0, 0.1, 0.15), (0.1, 0.15, 0.2, 0.3)):
        if abs(wx + wc + wa + wg - 1) < 1e-9:
            sp = [(t, w) for t, w in (('ch', wc), (X, wx), ('zoo_cat_alb', wa), ('graph_hc2_lgbm', wg)) if w > 0]
            cands[f'{X} {wx}/ch {wc}/catalb {wa}/hc2lgbm {wg}'] = sp
res = PS.run(cands, 431, verbose=False, extra_obs={'G40': 0.86977})
g = 431 / 3521; fac = np.sqrt((1 + g) / (1 - g))
out = []
for name, cvg, cm, sd, p in res:
    pc = 1 - norm.cdf((PS.TARGET - PS.CH_PUB - cm) / (sd * fac))
    out.append((name, cvg, PS.CH_PUB + cm, sd * fac, pc))
print('top by E[public]:')
for r in sorted(out, key=lambda t: -t[2])[:8]:
    print(f'  {r[0]:60s} CV gain {r[1]:+.5f}  E[public] {r[2]:.5f} (sd {r[3]:.5f})  P(>=0.873) {r[4]:.3f}')
print('top by P(>=0.873):')
for r in sorted(out, key=lambda t: -t[4])[:5]:
    print(f'  {r[0]:60s} CV gain {r[1]:+.5f}  E[public] {r[2]:.5f} (sd {r[3]:.5f})  P(>=0.873) {r[4]:.3f}')
