"""Family-mix blocks from cached r3_chx folds: 0.6 * CatBoost(variant X) + 0.4 * ExtraTrees(variant Y), each family
residual-corrected exactly as in Charlie's pipeline. Each family is trained independently inside the same fold and the same
inner split (seed S+100), so the mix IS the pipeline where the CatBoost family gets X's extra inputs and ExtraTrees gets Y's.
X/Y = variant letters (a | b | c | e ...) or 'rep' (replica cache work/ch/cache, CV only).
ET folds/full refits are read from cache/et_<Y>_* (ET-only runs) if present, else from cache/<Y>_*.
python -I mix.py X Y [screen]  -> results/r3_chx_mix_<X><Y>  (screen: results/r3_chx_scr_mix_<X><Y>, no test_full)"""
import sys, os
sys.dont_write_bytecode = True
import numpy as np
WORK = './work'
C_ = WORK + '/r3_chx/cache'
sys.path.insert(0, WORK)
import harness as H
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
X, Y = sys.argv[1], sys.argv[2]
screen = len(sys.argv) > 3 and sys.argv[3] == 'screen'


def cpath(v, k):
    return f'{WORK}/ch/cache/chfull_std_k{k}.npz' if v == 'rep' else f'{C_}/{v}_k{k}.npz'


def epath(v, k):
    if v == 'rep':
        return f'{WORK}/ch/cache/chfull_std_k{k}.npz'
    p = f'{C_}/et_{v}_k{k}.npz'
    return p if os.path.exists(p) else f'{C_}/{v}_k{k}.npz'


reps = [r for r in range(5) if all(os.path.exists(cpath(X, k)) and os.path.exists(epath(Y, k))
                                   for k in range(5 * r, 5 * r + 5))]
oof = np.full((5, n), np.nan)
tst = np.zeros(m)
for r in reps:
    for k in range(5 * r, 5 * r + 5):
        _, fit, val, elig = splits[k]
        c, e = np.load(cpath(X, k)), np.load(epath(Y, k))
        oof[r, val[elig]] = (0.6 * c['cat_v'] + 0.4 * e['et_v'])[elig]
        tst += (0.6 * c['cat_t'] + 0.4 * e['et_t']) / (5 * len(reps))
aucs = np.array([roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) if r in reps else np.nan
                 for r in range(5)])
ref = H.load_res('chrep_bl')['aucs']
full = None
if not screen:
    seeds = [s for s in range(10) if os.path.exists(f'{C_}/{X}_full_s{s}.npz') and
             (os.path.exists(f'{C_}/et_{Y}_full_s{s}.npz') or os.path.exists(f'{C_}/{Y}_full_s{s}.npz'))]
    assert seeds, 'no full refits'
    fl = []
    for s in seeds:
        ep = f'{C_}/et_{Y}_full_s{s}.npz'
        ep = ep if os.path.exists(ep) else f'{C_}/{Y}_full_s{s}.npz'
        fl.append(0.6 * np.load(f'{C_}/{X}_full_s{s}.npz')['cat_t'] + 0.4 * np.load(ep)['et_t'])
    full = np.mean(fl, axis=0)
name = f'r3_chx_scr_mix_{X}{Y}' if screen else f'r3_chx_mix_{X}{Y}'
H.save(name, dict(oof=oof, aucs=aucs, test_foldavg=tst), full)
chf = H.load_res('chrep_bl')['test_full']
print(f'{name}: reps {reps} AUC {np.nanmean(aucs):.5f} vs replica {ref[reps].mean():.5f} diff per rep '
      f'{np.round(aucs[reps] - ref[reps], 5)}' +
      ('' if full is None else f'; full seeds {seeds}; spearman(test_full, charlie file) {spearmanr(full, chf)[0]:.4f}; '
                               f'corr(foldavg, full) {np.corrcoef(tst, full)[0, 1]:.4f}'))
