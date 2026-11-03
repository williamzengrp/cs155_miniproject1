"""Blends around file 02 (graph-Charlie .6 + Charlie .2 + CatBoost-album .2, public 0.87079), conditioned on every
allowed public score (f8 f7 v2 h25 ch + G40 01 02 05 10). No TabPFN anywhere.
python -I near02.py [NPUB]"""
import sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work/main')
import pubsim as PS

npub = int(sys.argv[1]) if len(sys.argv) > 1 else 431
G, K, L = 'r3_chg_hc2_w6', 'zoo_cat_alb', 'r3_chp_labalbs2c'
OBS_DEF = {'G40': [('ch', .6), ('graph_hc2_lgbm', .4)],
           'O01': [(L, .6), ('ch', .2), (K, .2)],
           'O02': [(G, .6), ('ch', .2), (K, .2)],
           'O05': [('ch', .5), (K, .5)],
           'O10': [(G, .7), ('ch', .3)]}
EXTRA = {'G40': 0.86977, 'O01': 0.86947, 'O02': 0.87079, 'O05': 0.86933, 'O10': 0.86941}
cands = dict(OBS_DEF)
for g in np.arange(0.40, 0.851, 0.05):
    for k in np.arange(0.10, 0.451, 0.05):
        c = 1 - g - k
        if c < -1e-9:
            continue
        cands[f'G{g:.2f}+C{c:.2f}+K{k:.2f}'] = [(t, round(w, 2)) for t, w in ((G, g), ('ch', c), (K, k)) if w > 1e-9]
for x in ['graph_hc2_lgbm', 'graph_hc2_cat', 'r3_aug_lgbm_plganm_2s', 'r3_chg_hc2', L, 'zoo_lgb_linear',
          'te_lgbm_best_v2', 'r3_chx_mix_ca', 'graph_ds_lgbm']:
    for w in (0.1, 0.2):
        cands[f'02*{1 - w:.1f}+{x}@{w}'] = [(G, .6 * (1 - w)), ('ch', .2 * (1 - w)), (K, .2 * (1 - w)), (x, w)]
names, mu, cm, C = PS.run(cands, npub, verbose=False, extra_obs=EXTRA, full=True)
gg = npub / 3521
sd = np.sqrt(np.maximum(np.diag(C), 0) * (1 + gg) / (1 - gg))
E = PS.CH_PUB + cm
p02 = 1 - norm.cdf((EXTRA['O02'] - E) / np.maximum(sd, 1e-9))
p873 = 1 - norm.cdf((0.873 - E) / np.maximum(sd, 1e-9))
print(f'npub {npub}: {len(names)} candidates. Top 25 by expected public:')
for i in np.argsort(-E)[:25]:
    print(f'  {names[i]:40s} CVgain {mu[i]:+.5f}  E[pub] {E[i]:.5f}  sd {sd[i]:.5f}  P>02 {p02[i]:.3f}  P>=.873 {p873[i]:.3f}')
    print(f'      spec: {",".join(f"{t}:{w:.3f}" for t, w in cands[names[i]])}')
i = names.index('G0.60+C0.20+K0.20')
print(f'sanity (copy of 02): E {E[i]:.5f} sd {sd[i]:.5f}')
