"""r3_chp: extra-seed OUTER fits of Charlie's replica (CatBoost with staged predictions + ExtraTrees), no inner fits
(the residual correction of the averaged raw predictions re-uses the seed-0 inner OOF of the same fold / full fit).

  python -I -B chp_seed.py cv SOFF ITERS K0 K1 [TEVARY]   std folds; S = SEED0 + k + SOFF; date-TE cross-fit seed
                                                          42 (or 42 + SOFF if TEVARY=1) -> cache/sd_o<SOFF>_i<ITERS>_te<TEVARY>_k<k>.npz
  python -I -B chp_seed.py full S_IDX ITERS [TEVARY]      full refit seeds S = SEED0 + 1000*(s+1), s in S_IDX (e.g. 3,4)
                                                          -> cache/sdfull_s<s>_i<ITERS>_te<TEVARY>.npz
Staged CatBoost predictions are stored at STAGES (<= ITERS) as cat_v_<it> / cat_t_<it>."""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, GroupedDateTE, CFG, make_model, dup_weights
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
THREADS = int(os.environ.get('CHP_THREADS', '3'))
CACHE = WORK + '/r3_chp/cache'
STAGES = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]
tr, te, y = H.load()
n = len(tr)
F, BL = charlie_pool(tr, te)
R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)


def matrices(fit, applies, with_test, te_seed):
    enc = GroupedDateTE(seed=te_seed)
    Efit = enc.fit_transform([R.ktr[i] for i in fit], y[fit], R.groups[fit])
    Xf = np.column_stack([R.Ptr[fit], Efit])
    Xa = [np.column_stack([R.Ptr[b], enc.transform([R.ktr[i] for i in b])]) for b in applies]
    Xt = np.column_stack([R.Pte, enc.transform(R.kte)]) if with_test else None
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xf)
    return imp.transform(Xf), [imp.transform(x) for x in Xa], (imp.transform(Xt) if with_test else None)


def cat_model(seed, iters):
    c = CFG['cat']
    return CatBoostClassifier(iterations=iters, learning_rate=c['lr'], depth=c['depth'], l2_leaf_reg=c['l2'],
                              rsm=c['colsample'], random_strength=c['random_strength'], border_count=c['borders'],
                              loss_function='Logloss', verbose=False, allow_writing_files=False,
                              thread_count=THREADS, random_seed=seed)


def outer(fit, Xf, Xa, Xt, S, iters):
    out = {}
    m = cat_model(S, iters).fit(Xf, y[fit], sample_weight=dup_weights(R.uri[fit], CFG['cat']['duplicate_power']))
    for it in [s for s in STAGES if s <= iters]:
        if Xa is not None:
            out[f'cat_v_{it}'] = m.predict_proba(Xa, ntree_end=it)[:, 1]
        out[f'cat_t_{it}'] = m.predict_proba(Xt, ntree_end=it)[:, 1]
    e = make_model('extra', S, THREADS).fit(Xf, y[fit], sample_weight=dup_weights(R.uri[fit], CFG['extra']['duplicate_power']))
    if Xa is not None:
        out['et_v'] = e.predict_proba(Xa)[:, 1]
    out['et_t'] = e.predict_proba(Xt)[:, 1]
    return out


mode = sys.argv[1]
if mode == 'cv':
    soff, iters, k0, k1 = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    tev = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    splits = H.make_splits(tr, y, n_rep=5)
    for k in range(k0, min(k1, len(splits))):
        path = f'{CACHE}/sd_o{soff}_i{iters}_te{tev}_k{k}.npz'
        if os.path.exists(path):
            continue
        r, fit, val, elig = splits[k]
        t0 = time.time()
        Xf, (Xv,), Xt = matrices(fit, [val], True, 42 + (soff if tev else 0))
        out = outer(fit, Xf, Xv, Xt, SEED0 + k + soff, iters)
        np.savez(path, **out)
        print(f'[cv soff={soff} it={iters} tev={tev} k={k}] {time.time() - t0:.1f}s', flush=True)
elif mode == 'full':
    sidx = [int(s) for s in sys.argv[2].split(',')]
    iters = int(sys.argv[3])
    tev = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    fit = np.arange(n)
    for s in sidx:
        path = f'{CACHE}/sdfull_s{s}_i{iters}_te{tev}.npz'
        if os.path.exists(path):
            continue
        t0 = time.time()
        Xf, _, Xt = matrices(fit, [], True, 42 + (1000 * (s + 1) if tev else 0))
        out = outer(fit, Xf, None, Xt, SEED0 + 1000 * (s + 1), iters)
        np.savez(path, **out)
        print(f'[full s={s} it={iters} tev={tev}] {time.time() - t0:.1f}s', flush=True)
