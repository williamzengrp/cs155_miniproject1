"""Assess all round-3 blocks (results/r3_*) with the public simulator (7 observations) and choose a submission batch
maximising P(at least one public >= TARGET). Only blends that beat Charlie's model offline are eligible.
python -I assess_r3.py [BATCH] [NPUB] [extra observed: name=score ...]   (TARGET env, default 0.873)
Extra observed names must be candidate names printed by a previous run (their definitions are rebuilt here)."""
import os, sys, glob, itertools
import numpy as np
sys.path.insert(0, './work')
sys.path.insert(0, './work/main')
import harness as H
import pubsim as PS

PS.TARGET = float(os.environ.get('TARGET', '0.873'))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
npub = int(sys.argv[2]) if len(sys.argv) > 2 else 431
extra_cli = dict((a.split('=')[0], float(a.split('=')[1])) for a in sys.argv[3:])

ref = H.load_res('chrep_bl')['oof']
new = []
_only = [x for x in os.environ.get('R3_BLOCKS', '').split(',') if x]
for f in sorted(glob.glob(H.HERE + '/results/r3_*.npz')):
    n = os.path.basename(f)[:-4]
    if _only and n not in _only:
        continue
    if n.startswith('r3_tab') or 'tabpfn' in n:
        continue   # foundation models are not allowed
    r = H.load_res(n)
    tf = r['test_full']
    if r['oof'].shape != ref.shape or tf is None or np.size(tf) != 863 or not np.isfinite(tf).all():
        print('skip (bad shape/test_full):', n); continue
    rows_ok = [k for k in range(ref.shape[0]) if not np.isnan(r['oof'][k]).all()]
    if not rows_ok or not all(((~np.isnan(r['oof'][k])) == (~np.isnan(ref[k]))).all() for k in rows_ok):
        print('skip (OOF pattern differs from chrep_bl):', n); continue
    new.append(n)
print('new blocks:', new)

OBS_DEF = {'G40': [('ch', 0.6), ('graph_hc2_lgbm', 0.4)]}   # TabPFN files (final9, T2) are not allowed / not used
EXTRA = {'G40': 0.86977}
OLD = ['zoo_cat_alb', 'graph_hc2_lgbm', 'zoo_lgb_linear', 'graph_hc2_cat',
       'eda_chl_lgbm', 'eda_chl_et', 'te_lgbm_best_v2', 'fkd_lgbm']
cands = dict(OBS_DEF)
for x in new + OLD:
    for w in (0.2, 0.3, 0.4, 0.5):
        cands[f'ch{1 - w:.1f}+{x}@{w}'] = [('ch', 1 - w), (x, w)]
    cands[f'ch.7+catalb.15+{x}.15'] = [('ch', 0.7), ('zoo_cat_alb', 0.15), (x, 0.15)]
for x in new:   # a new Charlie-like block may also REPLACE part of Charlie's file
    for w in (0.5, 0.7, 1.0):
        cands[f'{x}@{w}+ch{1 - w:.1f}'] = [(x, w)] + ([('ch', 1 - w)] if w < 1 else [])
    cands[f'{x}.6+ch.2+catalb.2'] = [(x, 0.6), ('ch', 0.2), ('zoo_cat_alb', 0.2)]
for x, z in itertools.combinations(new, 2):
    cands[f'ch.5+{x}.25+{z}.25'] = [('ch', 0.5), (x, 0.25), (z, 0.25)]
for k in extra_cli:
    assert k in cands, f'unknown observed candidate {k}'
EXTRA.update(extra_cli)

names, mu, cm, C = PS.run(cands, npub, verbose=False, extra_obs=EXTRA, full=True)
g = npub / 3521; C = C * (1 + g) / (1 - g)
need = PS.TARGET - PS.CH_PUB
ok = [i for i in range(len(names)) if mu[i] > 0]
names = [names[i] for i in ok]; mu = mu[ok]; cm = cm[ok]; C = C[np.ix_(ok, ok)]
sd = np.sqrt(np.diag(C))
from scipy.stats import norm
P = 1 - norm.cdf((need - cm) / sd)
order = np.argsort(-cm)
print(f'\nTop 15 by expected public (target {PS.TARGET}, npub {npub}, {len(names)} CV-positive candidates):')
for i in order[:15]:
    print(f'  {names[i][:66]:66s} CVgain {mu[i]:+.5f}  E[pub] {PS.CH_PUB + cm[i]:.5f} sd {sd[i]:.5f}  P {P[i]:.3f}')
rng = np.random.default_rng(0)
L = np.linalg.cholesky(C + np.eye(len(cm)) * 1e-12)
Z = cm[None, :] + rng.standard_normal((40000, len(cm))) @ L.T
hit = Z >= need
cur = np.zeros(len(Z), bool); picks = []
print(f'\nGreedy batch of {B} maximising P(any >= {PS.TARGET}):')
for b in range(B):
    j = int(np.argmax((cur[:, None] | hit).mean(0)))
    picks.append(j); cur |= hit[:, j]
    print(f'  {b + 1}. {names[j][:66]:66s} CVgain {mu[j]:+.5f} E[pub] {PS.CH_PUB + cm[j]:.5f} sd {sd[j]:.5f} '
          f'P {hit[:, j].mean():.3f}  P(any) {cur.mean():.3f}')
    print(f'     spec: {",".join(f"{t}:{w}" for t, w in cands[names[j]])}')
