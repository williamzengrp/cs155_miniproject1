"""Audit of r3_chp_labalbs2c (lab-alb / lab-alb-tev outer models of r3_chp/chp_var.py), dynamic part.
Re-implements chp_var.py's matrices()/outer() verbatim but with the label vector as an explicit argument.
  python -I -B dyn.py K SOFF MODE     MODE: flip  -> run(y) vs cache, then run(y with VAL labels flipped) vs run(y)
                                      MODE: shuf  -> run(y) vs cache, then run(y with FIT labels permuted): val AUC
"""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
import pandas as pd
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/ch'); sys.path.insert(0, WORK + '/te'); sys.path.insert(0, WORK + '/main')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, GroupedDateTE, CFG, dup_weights
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from label_features import label_features, BEST_SPEC, get_ctx
warnings.filterwarnings('ignore')
THREADS = 1
CACHE = WORK + '/r3_chp/cache'
k, soff, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
variant = 'lab-alb-tev' if soff else 'lab-alb'
toks = variant.split('-')
ccfg, ecfg = dict(CFG['cat']), dict(CFG['extra'])
ITERS = ccfg['rounds']
tr, te, y = H.load()
n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
cols = list(BL['PROJECT'])
print('PROJECT cols', len(cols), 'suspicious:', [c for c in cols if any(s in c.lower() for s in ('id', 'uri', 'index', 'row', 'popul'))], flush=True)
R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)
CTX = get_ctx(tr, te)
XA_DUMMY = pd.DataFrame(index=range(n + m))
ALB = CTX.key['alb']


def matrices(yy, fit, applies, te_seed, lab_seed):
    enc = GroupedDateTE(seed=te_seed)
    Efit = enc.fit_transform([R.ktr[i] for i in fit], yy[fit], R.groups[fit])
    Xf = [R.Ptr[fit], Efit]
    Xa = [[R.Ptr[b], enc.transform([R.ktr[i] for i in b])] for b in applies]
    Xt = [R.Pte, enc.transform(R.kte)]
    Lf, La = label_features(XA_DUMMY, yy, fit, list(applies) + [np.arange(n, n + m)], tr, te, seed=lab_seed,
                            spec=BEST_SPEC)
    Xf.append(Lf.to_numpy(float)); [x.append(l.to_numpy(float)) for x, l in zip(Xa, La[:-1])]
    Xt.append(La[-1].to_numpy(float))
    Xf = np.column_stack(Xf); Xa = [np.column_stack(x) for x in Xa]; Xt = np.column_stack(Xt)
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xf)
    Xf, Xa, Xt = imp.transform(Xf), [imp.transform(x) for x in Xa], imp.transform(Xt)
    al = (ALB[fit], [ALB[b] for b in applies], ALB[n:])
    return (Xf, Xa, Xt), al


def to_pool(X, a, lab=None, w=None):
    df = pd.DataFrame(X, columns=[f'f{j}' for j in range(X.shape[1])])
    df['alb'] = np.asarray(a).astype(str)
    return Pool(df, label=lab, weight=w, cat_features=['alb'])


def outer(yy, fit, mats, al, S):
    (Xf, Xa, Xt), (af, aa, at) = mats, al
    c = ccfg
    cb = CatBoostClassifier(iterations=ITERS, learning_rate=c['lr'], depth=c['depth'], l2_leaf_reg=c['l2'],
                            rsm=c['colsample'], random_strength=c['random_strength'], border_count=c['borders'],
                            loss_function='Logloss', verbose=False, allow_writing_files=False,
                            thread_count=THREADS, random_seed=S)
    cb.fit(to_pool(Xf, af, yy[fit], None))   # duplicate_power 0 for cat (CFG) -> no weights
    out = {'cat_v': cb.predict_proba(to_pool(Xa[0], aa[0]))[:, 1], 'cat_t': cb.predict_proba(to_pool(Xt, at))[:, 1]}
    e = ecfg
    et = ExtraTreesClassifier(n_estimators=e['rounds'], max_depth=e['depth'], min_samples_leaf=e['min_leaf'],
                              max_features=e['colsample'], n_jobs=THREADS, random_state=S)
    et.fit(Xf, yy[fit], sample_weight=dup_weights(R.uri[fit], e['duplicate_power']))
    out['et_v'] = et.predict_proba(Xa[0])[:, 1]
    out['et_t'] = et.predict_proba(Xt)[:, 1]
    return out, (Xf, Xa[0], Xt)


assert ccfg['duplicate_power'] == 0
splits = H.make_splits(tr, y, n_rep=5)
r, fit, val, elig = splits[k]
te_seed, lab_seed, S = 42 + (soff if 'tev' in toks else 0), k + soff, SEED0 + k + soff


def run(yy, tag):
    t0 = time.time()
    mats, al = matrices(yy, fit, [val], te_seed, lab_seed)
    out, X = outer(yy, fit, mats, al, S)
    print(f'  [{tag}] {time.time() - t0:.1f}s', flush=True)
    return out, X


def auc(p):
    return roc_auc_score(y[val][elig], p[elig])


o0, X0 = run(y, 'true y')
cache = dict(np.load(f'{CACHE}/var_{variant}_o{soff}_k{k}.npz'))
for a, b in (('cat_v', 'cat_v_2000'), ('cat_t', 'cat_t_2000'), ('et_v', 'et_v'), ('et_t', 'et_t')):
    print(f'k={k} soff={soff} reproduce cache {b}: max|diff| {np.abs(o0[a] - cache[b]).max():.3g}', flush=True)
print(f'k={k} true-y AUC (eligible val) cat {auc(o0["cat_v"]):.4f} et {auc(o0["et_v"]):.4f} '
      f'blend {auc(0.6 * o0["cat_v"] + 0.4 * o0["et_v"]):.4f}', flush=True)
if mode == 'flip':
    yf = y.copy(); yf[val] = 1 - yf[val]
    o1, X1 = run(yf, 'VAL labels flipped')
    print(f'k={k} flip: identical inputs (Xf, Xv, Xt): {[np.array_equal(a, b) for a, b in zip(X0, X1)]}', flush=True)
    for a in ('cat_v', 'cat_t', 'et_v', 'et_t'):
        print(f'k={k} flip: {a} max|diff| vs true-y run {np.abs(o0[a] - o1[a]).max():.3g}', flush=True)
else:
    rng = np.random.default_rng(12345)
    ys = y.copy(); ys[fit] = rng.permutation(y[fit])
    o1, X1 = run(ys, 'FIT labels permuted')
    print(f'k={k} shuffled-fit AUC (eligible val, true labels) cat {auc(o1["cat_v"]):.4f} et {auc(o1["et_v"]):.4f} '
          f'blend {auc(0.6 * o1["cat_v"] + 0.4 * o1["et_v"]):.4f}  (n elig {elig.sum()})', flush=True)
print('DONE', flush=True)
