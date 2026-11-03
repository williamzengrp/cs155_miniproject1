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
           'O10': [(G, .7), ('ch', .3)],
           'O11': [(G, .4), ('ch', .3), (K, .15), ('zoo_lgb_linear', .15)]}
EXTRA = {'G40': 0.86977, 'O01': 0.86947, 'O02': 0.87079, 'O05': 0.86933, 'O10': 0.86941, 'O11': 0.87183}
cands = dict(OBS_DEF)
LL = 'zoo_lgb_linear'
for g in (0.25, 0.30, 0.35, 0.40, 0.45):
    for k in (0.05, 0.10, 0.15, 0.20):
        for l in (0.10, 0.15, 0.20, 0.25, 0.30):
            c = round(1 - g - k - l, 2)
            if 0.15 <= c <= 0.50:
                cands[f'G{g:.2f}+C{c:.2f}+K{k:.2f}+LL{l:.2f}'] = [(G, g), ('ch', c), (K, k), (LL, l)]
for x in ['te_lgbm_best_v2', 'graph_ds_lgbm', 'graph_ds_xgb', 'graph_hc2_lgbm', 'graph_hc2_cat', 'r3_aug_lgbm_plganm_2s',
          'r3_chx_mix_ca', 'eda_chl_lgbm', 'mdl_mlp_ple', 'ch_lgbm_proj', 'fkd_et', 'b2_et', 'v2']:
    cands[f'11*0.9+{x}@0.1'] = [(G, .36), ('ch', .27), (K, .135), (LL, .135), (x, .1)]
cands['G0.60+C0.20+K0.20'] = [(G, .6), ('ch', .2), (K, .2)]
names, mu, cm, C = PS.run(cands, npub, verbose=False, extra_obs=EXTRA, full=True)
gg = npub / 3521
sd = np.sqrt(np.maximum(np.diag(C), 0) * (1 + gg) / (1 - gg))
E = PS.CH_PUB + cm
p02 = 1 - norm.cdf((EXTRA['O11'] - E) / np.maximum(sd, 1e-9))
p873 = 1 - norm.cdf((0.873 - E) / np.maximum(sd, 1e-9))
print(f'npub {npub}: {len(names)} candidates. Top 25 by expected public:')
for i in np.argsort(-E)[:25]:
    print(f'  {names[i]:40s} CVgain {mu[i]:+.5f}  E[pub] {E[i]:.5f}  sd {sd[i]:.5f}  P>11 {p02[i]:.3f}  P>=.873 {p873[i]:.3f}')
    print(f'      spec: {",".join(f"{t}:{w:.3f}" for t, w in cands[names[i]])}')
i = names.index('G0.60+C0.20+K0.20')
print(f'sanity (copy of 02): E {E[i]:.5f} sd {sd[i]:.5f}')
