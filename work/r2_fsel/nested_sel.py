"""NESTED feature selection (honest estimate of the selection procedure): in outer fold k the kept feature set
is decided ONLY from that fold's in-fold permutation importance (pi.py, inner folds of the fit part).
python -I nested_sel.py MODEL RULE SEED0 NREP PREFIX  -> results/<PREFIX>_nsel_<MODEL>_<RULE>.npz (OOF only)
RULE: f0   drop single features with PI <= 0         (label features always kept)
      g0   drop whole groups with group PI <= 0      (label group always kept)
      q25  drop the 25% least important single features
Prints kept-set sizes and the paired difference vs the same model on all features."""
import sys, pickle, time
import numpy as np
sys.path.insert(0, './work/r2_fsel')
import fsel_lib as L
MODEL, RULE, SEED0, NREP, PREFIX = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
PIM = sys.argv[6] if len(sys.argv) > 6 else 'lgbm'
pi = pickle.load(open(L.C + f'pi_{PIM}_{SEED0}_{NREP}.pkl', 'rb'))
folds = L.load_folds(SEED0, NREP)


def kept(k):
    pf, pg = pi['feat'][k], pi['group'][k]
    if RULE == 'f0':
        return [c for j, c in enumerate(L.ALL) if pf[j] > 0 or c in L.LABEL]
    if RULE == 'g0':
        bad = {g for g, v in zip(pi['group_names'], pg) if v <= 0 and g != 'label'}
        return [c for c in L.ALL if not any(c in L.GROUPS[g] for g in bad)]
    if RULE == 'q25':
        thr = np.quantile(pf, 0.25)
        return [c for j, c in enumerate(L.ALL) if pf[j] > thr or c in L.LABEL]
    raise ValueError(RULE)


R = max(f['r'] for f in folds) + 1
oof = np.full((R, L.n), np.nan)
tst = np.zeros(L.m)
sizes = []
t0 = time.time()
for k, f in enumerate(folds):
    cols = kept(k)
    sizes.append(len(cols))
    mdl = L.mk(MODEL, k).fit(L.mat(MODEL, cols, L.X.iloc[f['fit_idx']], f['F_fit']), L.y[f['fit_idx']])
    pv = mdl.predict_proba(L.mat(MODEL, cols, L.X.iloc[f['val_idx']], f['F_val']))[:, 1]
    oof[f['r'], f['val_idx'][f['elig']]] = pv[f['elig']]
    tst += mdl.predict_proba(L.mat(MODEL, cols, L.Xt, f['F_te']))[:, 1] / len(folds)
a = L.save(f'{PREFIX}_nsel_{MODEL}_{RULE}', oof, tst)
print(f'[{PREFIX}_nsel_{MODEL}_{RULE}] kept {np.mean(sizes):.1f} (min {min(sizes)} max {max(sizes)}) AUC {a.mean():.5f} '
      f'per-rep {np.round(a, 4)} ({time.time()-t0:.0f}s)', flush=True)
