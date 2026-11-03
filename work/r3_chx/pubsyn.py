"""Synthetic (additive, rank space) combos of screened r3_chx variants -> pubsim conditional estimates (screening aid)."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS
PS.TARGET = 0.873
A, B, C = 'r3_chx_scr_a', 'r3_chx_scr_b', 'r3_chx_scr_c'
syn = {'ac': [(A, 1), (C, 1), ('ch', -1)], 'bc': [(B, 1), (C, 1), ('ch', -1)], 'ab': [(A, 1), (B, 1), ('ch', -1)],
       'abc': [(A, 1), (B, 1), (C, 1), ('ch', -2)], 'c': [(C, 1)], 'a': [(A, 1)]}
cands = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
for nm, sp in syn.items():
    cands[f'{nm}@1'] = sp
    cands[f'{nm}.75+catalb.1+hc2lgbm.15'] = [(t, 0.75 * w) for t, w in sp] + [('zoo_cat_alb', 0.1), ('graph_hc2_lgbm', 0.15)]
    cands[f'ch.5+{nm}.5'] = [('ch', 0.5)] + [(t, 0.5 * w) for t, w in sp]
res = PS.run(cands, 431, verbose=False, extra_obs={'G40': 0.86977})
g = 431 / 3521; fac = np.sqrt((1 + g) / (1 - g))
for name, cvg, cm, sd, p in sorted(res, key=lambda t: -t[2]):
    pc = 1 - norm.cdf((PS.TARGET - PS.CH_PUB - cm) / (sd * fac))
    print(f'{name:40s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  P {pc:.3f}')
