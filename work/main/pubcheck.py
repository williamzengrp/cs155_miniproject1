"""Quick public-potential check of new result(s) (std OOF + test_full) as blend partners of Charlie's file.
Conditions on the 5 public scores of files WITHOUT TabPFN (f8, f7, v2, h25, G40 0.86977). Foundation models (TabPFN)
are not allowed: no TabPFN block may be used. P includes the finite-pool sd inflation.
python -I pubcheck.py NAME [NAME ...]    (NPUB env, default 431; TARGET env, default 0.873)"""
import os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS

PS.TARGET = float(os.environ.get('TARGET', '0.873'))
npub = int(os.environ.get('NPUB', '431'))
OBS_DEF = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}
EXTRA = {'G40': 0.86977}


def check(names, verbose=True):
    assert not any('tabpfn' in n or n.startswith('r3_tab') for n in names), 'TabPFN is not allowed'
    cands = dict(OBS_DEF)
    for n in names:
        for w in (0.2, 0.3, 0.4, 0.5):
            cands[f'ch{1 - w:.1f}+{n}@{w}'] = [('ch', 1 - w), (n, w)]
        cands[f'ch.7+catalb.15+{n}.15'] = [('ch', 0.7), ('zoo_cat_alb', 0.15), (n, 0.15)]
        cands[f'ch.55+catalb.1+hc2lgbm.15+{n}.2'] = [('ch', 0.55), ('zoo_cat_alb', 0.1), ('graph_hc2_lgbm', 0.15), (n, 0.2)]
    res = PS.run(cands, npub, verbose=False, extra_obs=EXTRA)
    g = npub / 3521; fac = np.sqrt((1 + g) / (1 - g))
    need = PS.TARGET - PS.CH_PUB
    out = []
    for name, cvg, cm, sd, p in res:
        pc = 1 - norm.cdf((need - cm) / (sd * fac))
        out.append((name, cvg, cm, sd * fac, pc))
        if verbose:
            print(f'{name[:60]:60s} CV gain vs ch {cvg:+.5f}  E[public] {PS.CH_PUB + cm:.5f} (sd {sd * fac:.5f})  '
                  f'P(>={PS.TARGET}) {pc:.3f}')
    return out


if __name__ == '__main__':
    check(sys.argv[1:])
