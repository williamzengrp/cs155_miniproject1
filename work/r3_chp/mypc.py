"""r3_chp: pubcheck with custom candidate specs (same conditioning / sd inflation as work/main/pubcheck.py).
python -I mypc.py "name=tok:w,tok:w" ...      tok = ch | <result name>"""
import os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS

PS.TARGET = float(os.environ.get('TARGET', '0.873'))
npub = int(os.environ.get('NPUB', '431'))
cands = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
for c in sys.argv[1:]:
    name, spec = c.split('=')
    cands[name] = [(t.split(':')[0], float(t.split(':')[1])) for t in spec.split(',')]
assert not any('tabpfn' in t or t.startswith('r3_tab') for v in cands.values() for t, _ in v)
res = PS.run(cands, npub, verbose=False, extra_obs={'G40': 0.86977})
g = npub / 3521; fac = np.sqrt((1 + g) / (1 - g))
for name, cvg, cm, sd, p in res:
    pc = 1 - norm.cdf((PS.TARGET - PS.CH_PUB - cm) / (sd * fac))
    print(f'{name[:60]:60s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  P(>={PS.TARGET}) {pc:.3f}')
