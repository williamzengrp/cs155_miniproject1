"""Public-LB simulator check (work/main/pubsim.py, same conditioning as work/main/pubcheck.py) of r3_chg results
used ALONE (as a replacement of Charlie's file) and blended with ch at a few weights.
python -I pubalone.py NAME [NAME ...]"""
import os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS
PS.TARGET = float(os.environ.get('TARGET', '0.873'))
npub = int(os.environ.get('NPUB', '431'))
OBS_DEF = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
EXTRA = {'G40': 0.86977}
names = sys.argv[1:]
assert not any('tabpfn' in n for n in names)
cands = dict(OBS_DEF)
for n in names:
    cands[f'{n}@1.0'] = [(n, 1.0)]
    for w in (0.5, 0.7, 0.85):
        cands[f'ch{1 - w:.2f}+{n}@{w}'] = [('ch', 1 - w), (n, w)]
res = PS.run(cands, npub, verbose=False, extra_obs=EXTRA)
g = npub / 3521; fac = np.sqrt((1 + g) / (1 - g))
need = PS.TARGET - PS.CH_PUB
for name, cvg, cm, sd, p in res:
    pc = 1 - norm.cdf((need - cm) / (sd * fac))
    print(f'{name[:60]:60s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  P(>={PS.TARGET}) {pc:.3f}')
