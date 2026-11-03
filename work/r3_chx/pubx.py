"""Extra public-LB checks (pubsim conditional model; same conditioning as pubcheck.py) for r3_chx blocks:
the block replacing Charlie's file entirely, and multi-way blends.  python -I pubx.py NAME [NAME ...]"""
import os, sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS
PS.TARGET = 0.873
npub = 431
EXTRA = {'G40': 0.86977}
cands = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
for n in sys.argv[1:]:
    assert 'tabpfn' not in n and not n.startswith('r3_tab')
    cands[f'{n}@1'] = [(n, 1.0)]
    for w in (0.6, 0.7, 0.8):
        cands[f'ch{1 - w:.1f}+{n}@{w}'] = [('ch', 1 - w), (n, w)]
    cands[f'{n}.6+hc2lgbm.4'] = [(n, 0.6), ('graph_hc2_lgbm', 0.4)]
    cands[f'{n}.7+hc2lgbm.3'] = [(n, 0.7), ('graph_hc2_lgbm', 0.3)]
    cands[f'{n}.7+catalb.15+hc2lgbm.15'] = [(n, 0.7), ('zoo_cat_alb', 0.15), ('graph_hc2_lgbm', 0.15)]
    cands[f'{n}.55+catalb.1+hc2lgbm.15+ch.2'] = [(n, 0.55), ('zoo_cat_alb', 0.1), ('graph_hc2_lgbm', 0.15), ('ch', 0.2)]
    cands[f'{n}.75+catalb.1+hc2lgbm.15'] = [(n, 0.75), ('zoo_cat_alb', 0.1), ('graph_hc2_lgbm', 0.15)]
    cands[f'ch.55+catalb.1+hc2lgbm.15+{n}.2'] = [('ch', 0.55), ('zoo_cat_alb', 0.1), ('graph_hc2_lgbm', 0.15), (n, 0.2)]
res = PS.run(cands, npub, verbose=False, extra_obs=EXTRA)
g = npub / 3521; fac = np.sqrt((1 + g) / (1 - g))
need = PS.TARGET - PS.CH_PUB
for name, cvg, cm, sd, p in sorted(res, key=lambda t: -t[2]):
    pc = 1 - norm.cdf((need - cm) / (sd * fac))
    print(f'{name[:60]:60s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  P(>=0.873) {pc:.3f}')
