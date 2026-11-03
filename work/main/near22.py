"""Blends around file 12 (public 0.87202), conditioned on every allowed public score (f8 f7 v2 h25 ch + G40 01 02 05 10
11 12). No TabPFN. Grid over graph-Charlie G / Charlie C / CatBoost-album K / linear LightGBM LL, G swapped for its
siblings, and every compatible std block as a 5th member.   python -I near12.py [NPUB]"""
import sys, glob, os
import numpy as np
from scipy.stats import norm
sys.path.insert(0, './work')
sys.path.insert(0, './work/main')
import harness as H
import pubsim as PS

npub = int(sys.argv[1]); TRIES = int(sys.argv[2])
NEW = dict((a.split('=')[0], float(a.split('=')[1])) for a in sys.argv[3:])
G, K, L, LL = 'r3_chg_hc2_w6', 'zoo_cat_alb', 'r3_chp_labalbs2c', 'zoo_lgb_linear'
OBS_DEF = {'G40': [('ch', .6), ('graph_hc2_lgbm', .4)],
           'O01': [(L, .6), ('ch', .2), (K, .2)],
           'O02': [(G, .6), ('ch', .2), (K, .2)],
           'O05': [('ch', .5), (K, .5)],
           'O10': [(G, .7), ('ch', .3)],
           'O11': [(G, .4), ('ch', .3), (K, .15), (LL, .15)],
           'O12': [(G, .35), ('ch', .3), (K, .05), (LL, .3)]}
EXTRA = {'G40': 0.86977, 'O01': 0.86947, 'O02': 0.87079, 'O05': 0.86933, 'O10': 0.86941, 'O11': 0.87183,
         'O12': 0.87202}
FILES = {'13': [(G, .28), ('ch', .28), (LL, .24), ('r3_chx_c_cat', .20)],
         '14': [('r3_chg_hc2_et', .40), ('ch', .30), (LL, .30)],
         '15': [(G, .28), ('ch', .28), (LL, .24), ('ch_lgbm_proj', .20)],
         '16': [(G, .28), ('ch', .28), (LL, .24), ('r3_aug_lgbm_nm', .20)],
         '17': [('r3_chg_hc2_cat', .40), ('ch', .35), (LL, .25)],
         '18': [('r3_chg_hc2_et', .32), ('ch', .24), (LL, .24), ('r3_chx_c_cat', .20)],
         '19': [('r3_chg_hc2_et', .28), ('ch', .08), (LL, .24), ('r3_chx_c_cat', .40)],
         '20': [('r3_chg_hc2_et', .45), ('ch', .12), (LL, .28), ('r3_chx_c_cat', .15)],
         '21': [('r3_chg_hc2_et', .20), (LL, .20), ('r3_chx_c_cat', .60)],
         '22': [('r3_chg_hc2_et', .16), (LL, .16), ('r3_chx_c_cat', .48), ('ch_lgbm_proj', .20)]}
for k, v in NEW.items():
    OBS_DEF['O' + k] = FILES[k]; EXTRA['O' + k] = v
cands = dict(OBS_DEF)
def add(n, spec):
    cands[n] = [(t, w) for t, w in spec if w > 0]
ET, X = 'r3_chg_hc2_et', 'r3_chx_c_cat'
PJ = 'ch_lgbm_proj'
for p in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
    for e in (0.08, 0.12, 0.16, 0.20):
        for l in (0.08, 0.12, 0.16, 0.20):
            x = round(1 - p - e - l, 2)
            if 0.30 <= x <= 0.65:
                add(f'P{p:.2f}+ET{e:.2f}+LL{l:.2f}+X{x:.2f}', [(ET, e), (LL, l), (X, x), (PJ, p)])
for y in ('ch_lgbm_proj', 'r3_aug_lgbm_nm', K, G, 'r3_chg_hc2_cat', 'r3_chx_c_et', 'r3_chp_albs2c', 'graph_hc2_lgbm',
          'ch_etnw_proj'):
    for w in (0.1, 0.2):
        add(f'22*{1 - w:.1f}+{y}@{w}', [(ET, .16 * (1 - w)), (LL, .16 * (1 - w)), (X, .48 * (1 - w)), (PJ, .20 * (1 - w)), (y, w)])
for y in ('r3_chx_c', 'r3_chx_mix_ca', 'r3_chp_albs2c', 'r3_chp_alb', 'r3_chx_scr_mix_ca'):
    add(f'22 X->{y}', [(ET, .16), (LL, .16), (y, .48), (PJ, .20)])
    add(f'22 X.32+{y}.16', [(ET, .16), (LL, .16), (X, .32), (y, .16), (PJ, .20)])
add('22copy', [(ET, .16), (LL, .16), (X, .48), (PJ, .20)])
names, mu, cm, C = PS.run(cands, npub, verbose=False, extra_obs=EXTRA, full=True)
gg = npub / 3521
C = C * (1 + gg) / (1 - gg)
sd = np.sqrt(np.maximum(np.diag(C), 0))
E = PS.CH_PUB + cm
best0 = max(EXTRA.values())
rng = np.random.default_rng(0)
w, V = np.linalg.eigh(C)
Z = E[None, :] + (rng.standard_normal((60000, len(E))) * np.sqrt(np.maximum(w, 0))) @ V.T
print(f'npub {npub}: {len(names)} candidates; best public so far {best0}; tries left {TRIES}; observed new {NEW}')
cur = np.full(len(Z), best0); picks = []
for t in range(TRIES):
    gain = np.maximum(cur[:, None], Z).mean(0)
    j = int(np.argmax(gain)); picks.append(j); cur = np.maximum(cur, Z[:, j])
    print(f'  {t + 1}. {names[j]:34s} E[pub] {E[j]:.5f} sd {sd[j]:.5f} CVgain {mu[j]:+.5f} | E[best] {cur.mean():.5f} '
          f'P(best>=.873) {(cur >= .873).mean():.3f} P(best>=.875) {(cur >= .875).mean():.3f}')
    print(f'      spec: {",".join(f"{tk}:{x:.3f}" for tk, x in cands[names[j]])}')
g1 = np.maximum(best0, Z).mean(0)
print('single-shot ranking E[max(best, X)]:')
for i in np.argsort(-g1)[:8]:
    print(f'     {names[i]:34s} E[pub] {E[i]:.5f} sd {sd[i]:.5f} E[best] {g1[i]:.5f} P>=.873 {(Z[:, i] >= .873).mean():.3f}')
