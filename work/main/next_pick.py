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
         '18': [('r3_chg_hc2_et', .32), ('ch', .24), (LL, .24), ('r3_chx_c_cat', .20)]}
for k, v in NEW.items():
    OBS_DEF['O' + k] = FILES[k]; EXTRA['O' + k] = v
cands = dict(OBS_DEF)
def add(n, spec):
    cands[n] = [(t, w) for t, w in spec if w > 0]
add('P1 G.35+C.35+LL.30', [(G, .35), ('ch', .35), (LL, .30)])
add('P2 G.30+C.40+LL.30', [(G, .30), ('ch', .40), (LL, .30)])
add('P3 12*.9+r3_chx_c_cat.1', [(G, .315), ('ch', .27), (K, .045), (LL, .27), ('r3_chx_c_cat', .1)])
add('P4 12*.9+r3_chp_albs2c.1', [(G, .315), ('ch', .27), (K, .045), (LL, .27), ('r3_chp_albs2c', .1)])
SW = ['r3_chg_hc2_et', 'r3_chg_hc2_cat', 'r3_chg_hc2', 'r3_chg_hc2_r1_w6', 'r3_chg_hc2_r1_et', 'r3_chg_hc2_r1_cat',
      'r3_chg_hc2_w7']
for x in SW:
    add(f'S 12 G->{x}', [(x, .35), ('ch', .30), (K, .05), (LL, .30)])
    add(f'S P1 G->{x}', [(x, .35), ('ch', .35), (LL, .30)])
    add(f'H G.2+{x}.15+C.35+LL.3', [(G, .2), (x, .15), ('ch', .35), (LL, .30)])
for x in ('r3_chg_hc2_et', 'r3_chg_hc2_cat'):
    for w in (0.25, 0.30, 0.40):
        for l in (0.25, 0.30, 0.35):
            add(f'W {x}{w:.2f}+C{1 - w - l:.2f}+LL{l:.2f}', [(x, w), ('ch', round(1 - w - l, 2)), (LL, l)])
for x in ('ch_etnw_proj', 'ch_lgbm_proj', 'r3_aug_lgbm_nm', 'ch_et_proj', 'r3_aug_lgbm_plnm_s100', 'r3_chx_c_et',
          'r3_chx_c_cat'):
    for w in (0.1, 0.2):
        add(f'F P1*{1 - w:.1f}+{x}@{w}', [(G, .35 * (1 - w)), ('ch', .35 * (1 - w)), (LL, .30 * (1 - w)), (x, w)])
for k, v in FILES.items():
    if k not in NEW:
        add('file' + k, v)
for x, tag in (('r3_chg_hc2_et', 'ET'), ('r3_chg_hc2_cat', 'CAT')):
    for w in (0.45, 0.50, 0.55):
        for l in (0.20, 0.25, 0.30):
            add(f'{tag}{w:.2f}+C{1 - w - l:.2f}+LL{l:.2f}', [(x, w), ('ch', round(1 - w - l, 2)), (LL, l)])
    for y in ('r3_chx_c_cat', 'ch_lgbm_proj', 'r3_aug_lgbm_nm'):
        add(f'{tag}.36+C.27+LL.27+{y}.10', [(x, .36), ('ch', .27), (LL, .27), (y, .10)])
        add(f'{tag}.32+C.24+LL.24+{y}.20', [(x, .32), ('ch', .24), (LL, .24), (y, .20)])
    add(f'{tag}.30+G.10+C.30+LL.30', [(x, .30), (G, .10), ('ch', .30), (LL, .30)])
for y, z in (('r3_chx_c_cat', 'ch_lgbm_proj'), ('r3_chx_c_cat', 'r3_aug_lgbm_nm'), ('ch_lgbm_proj', 'r3_aug_lgbm_nm')):
    add(f'P1*.7+{y}.15+{z}.15', [(G, .245), ('ch', .245), (LL, .21), (y, .15), (z, .15)])
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
