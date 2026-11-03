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

npub = int(sys.argv[1]) if len(sys.argv) > 1 else 431
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
ref = H.load_res('chrep_bl')['oof']
ok = []
for f in sorted(glob.glob(H.HERE + '/results/*.npz')):
    n = os.path.basename(f)[:-4]
    if 'tabpfn' in n or n.startswith(('r3_tab', 'c11_', 'conf_', 'zoo_c9', 'graph_c9', 'chrep')) or n in (G, K, LL):
        continue
    try:
        r = H.load_res(n)
    except Exception:
        continue
    tf = r.get('test_full') if isinstance(r, dict) else r['test_full']
    o = r['oof']
    if tf is None or np.size(tf) != 863 or not np.isfinite(tf).all() or o.shape != ref.shape:
        continue
    if not all(((~np.isnan(o[k])) == (~np.isnan(ref[k]))).all() for k in range(ref.shape[0])):
        continue
    ok.append(n)
print('compatible 5th-member blocks:', len(ok), flush=True)
cands = dict(OBS_DEF)
for g in (0.20, 0.25, 0.30, 0.35, 0.40):
    for k in (0.0, 0.05, 0.10):
        for l in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            c = round(1 - g - k - l, 2)
            if 0.10 <= c <= 0.50:
                cands[f'G{g:.2f}+C{c:.2f}+K{k:.2f}+LL{l:.2f}'] = [(t, w) for t, w in ((G, g), ('ch', c), (K, k), (LL, l)) if w > 0]
for x in ok:
    cands[f'12*0.9+{x}@0.1'] = [(G, .315), ('ch', .27), (K, .045), (LL, .27), (x, .1)]
    if x.startswith('r3_chg'):
        cands[f'12 with G->{x}'] = [(x, .35), ('ch', .3), (K, .05), (LL, .3)]
names, mu, cm, C = PS.run(cands, npub, verbose=False, extra_obs=EXTRA, full=True)
gg = npub / 3521
sd = np.sqrt(np.maximum(np.diag(C), 0) * (1 + gg) / (1 - gg))
E = PS.CH_PUB + cm
best = EXTRA['O12']
z = (E - best) / np.maximum(sd, 1e-9)
eimp = (E - best) * norm.cdf(z) + sd * norm.pdf(z)   # expected gain of the best public score
p12 = norm.cdf(z)
p873 = 1 - norm.cdf((0.873 - E) / np.maximum(sd, 1e-9))
print(f'npub {npub}: {len(names)} candidates. Top 30 by expected public:')
for i in np.argsort(-E)[:30]:
    print(f'  {names[i]:44s} CVgain {mu[i]:+.5f}  E[pub] {E[i]:.5f}  sd {sd[i]:.5f}  P>12 {p12[i]:.3f}  '
          f'P>=.873 {p873[i]:.3f}  E[gain of best] {eimp[i]:.5f}')
print('Top 10 by expected gain of the best public score:')
for i in np.argsort(-eimp)[:10]:
    print(f'  {names[i]:44s} CVgain {mu[i]:+.5f}  E[pub] {E[i]:.5f}  sd {sd[i]:.5f}  P>12 {p12[i]:.3f}  '
          f'P>=.873 {p873[i]:.3f}  E[gain of best] {eimp[i]:.5f}')
i = names.index('G0.35+C0.30+K0.05+LL0.30')
print(f'sanity (copy of 12): E {E[i]:.5f} sd {sd[i]:.5f}')
