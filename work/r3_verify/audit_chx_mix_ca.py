"""Leakage audit of r3_chx_mix_ca (cat variant c + ET variant a) on std fold k (default 0).
Stage 1: val-label FLIP invariance (y[val] -> 1-y[val]) of matrices / inner sets (variant a: date TE + label feats)
         and of the ET family predictions (bitwise expected), + reproduction vs cached a_k{k}.
Stage 2: FIT-label shuffle (val labels kept true): exact cat(c)+et(a) pipeline -> val AUC must be ~0.5."""
import sys, os, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
WORK = './work'
for p in (WORK + '/r3_chx', WORK, WORK + '/ch', WORK + '/main', WORK + '/te'):
    sys.path.insert(0, p)
import harness as H
from sklearn.metrics import roc_auc_score
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
from ch_common import connected_groups
from ch_feats import charlie_pool
from chx_lib import ReplicaX, CD_COLS, SEED0
from features_final import build_features_final
from label_features import get_ctx
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stage = sys.argv[2] if len(sys.argv) > 2 else '12'
T = 2
F, BL = charlie_pool(tr, te)
X, Xt = build_features_final(tr, te)
XA = pd.concat([X, Xt], ignore_index=True)
CD = XA[CD_COLS].to_numpy(float)
alb = get_ctx(tr, te).key['alb']
G = connected_groups(tr)
r, fit, val, elig = splits[k]
S = SEED0 + k
C = WORK + '/r3_chx/cache'
print(f'fold {k}: fit {len(fit)} val {len(val)} elig {elig.sum()}', flush=True)


def mk(yv, v):
    return ReplicaX(tr, te, yv, F[BL['PROJECT']], G, v, threads=T, XA=XA if 'a' in v else None,
                    CD=CD if 'a' in v else None, alb=alb if 'c' in v else None)


def md(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


if '1' in stage:
    t0 = time.time()
    yf = y.copy(); yf[val] = 1 - y[val]
    Ra, Rf = mk(y, 'a'), mk(yf, 'a')
    Xf1, (Xv1,), Xt1 = Ra.matrices(fit, [val], True, lf_seed=k)
    Xf2, (Xv2,), Xt2 = Rf.matrices(fit, [val], True, lf_seed=k)
    print(f'[flip] matrices maxdiff fit {md(Xf1, Xf2):.3g} val {md(Xv1, Xv2):.3g} test {md(Xt1, Xt2):.3g}', flush=True)
    i1, i2 = Ra.inner_sets(fit, S), Rf.inner_sets(fit, S)
    print('[flip] inner sets maxdiff', [(md(a[2], b[2]), md(a[3], b[3]), bool((a[0] == b[0]).all())) for a, b in zip(i1, i2)],
          flush=True)
    (rv1,), rt1, ip1 = Ra.run_family('extra', fit, Xf1, [Xv1], [val], Xt1, i1, S)
    (rv2,), rt2, ip2 = Rf.run_family('extra', fit, Xf2, [Xv2], [val], Xt2, i2, S)
    v1 = Ra.correct('extra', fit, ip1, rv1, tr.iloc[val]); v2 = Rf.correct('extra', fit, ip2, rv2, tr.iloc[val])
    t1 = Ra.correct('extra', fit, ip1, rt1, te); t2 = Rf.correct('extra', fit, ip2, rt2, te)
    print(f'[flip] ET a: raw_v {md(rv1, rv2):.3g} ip {md(ip1, ip2):.3g} v {md(v1, v2):.3g} t {md(t1, t2):.3g}', flush=True)
    cp = f'{C}/et_a_k{k}.npz' if os.path.exists(f'{C}/et_a_k{k}.npz') else f'{C}/a_k{k}.npz'
    c = np.load(cp)
    print(f'[repro] ET a vs cache {os.path.basename(cp)}: v {md(v1, c["et_v"]):.3g} t {md(t1, c["et_t"]):.3g} ip {md(ip1, c["et_ip"]):.3g}'
          f'  AUC true-run val {roc_auc_score(y[val[elig]], v1[elig]):.5f}', flush=True)
    # date-TE-only matrices of variant c: flip invariance
    Rc, Rcf = mk(y, 'c'), mk(yf, 'c')
    A1 = Rc.matrices(fit, [val], True, lf_seed=k); A2 = Rcf.matrices(fit, [val], True, lf_seed=k)
    print(f'[flip] variant c matrices maxdiff fit {md(A1[0], A2[0]):.3g} val {md(A1[1][0], A2[1][0]):.3g} test {md(A1[2], A2[2]):.3g}'
          f'  ({time.time() - t0:.0f}s)', flush=True)

if '2' in stage:
    t0 = time.time()
    ys = y.copy()
    ys[fit] = np.random.default_rng(12345).permutation(y[fit])
    out = {}
    for v, fam, tag in (('a', 'extra', 'et'), ('c', 'cat', 'cat')):
        R = mk(ys, v)
        Xf, (Xv,), Xt_ = R.matrices(fit, [val], True, lf_seed=k)
        inner = R.inner_sets(fit, S)
        (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], [val], Xt_, inner, S)
        cv_ = R.correct(fam, fit, ip, rv, tr.iloc[val])
        out[tag] = cv_
        print(f'[shuffle] {tag}({v}) val AUC (elig, true labels): raw {roc_auc_score(y[val[elig]], rv[elig]):.4f} '
              f'corrected {roc_auc_score(y[val[elig]], cv_[elig]):.4f}  ({time.time() - t0:.0f}s)', flush=True)
    mix = 0.6 * out['cat'] + 0.4 * out['et']
    print(f'[shuffle] MIX 0.6cat+0.4et val AUC {roc_auc_score(y[val[elig]], mix[elig]):.4f}; '
          f'null SE ~ {np.sqrt((elig.sum() + 1) / (12 * y[val[elig]].sum() * (1 - y[val[elig]]).sum())):.4f}', flush=True)
    cc, ce = np.load(f'{C}/c_k{k}.npz'), np.load(f'{C}/a_k{k}.npz' if not os.path.exists(f'{C}/et_a_k{k}.npz') else f'{C}/et_a_k{k}.npz')
    tm = 0.6 * cc['cat_v'] + 0.4 * ce['et_v']
    print(f'[true-run cache] mix val AUC {roc_auc_score(y[val[elig]], tm[elig]):.4f}', flush=True)

if '3' in stage:
    # CatBoost variant c with VAL labels flipped: must reproduce the cached TRUE-label run c_k{k} (val-label invariance)
    t0 = time.time()
    yf = y.copy(); yf[val] = 1 - y[val]
    R = mk(yf, 'c')
    Xf, (Xv,), Xt_ = R.matrices(fit, [val], True, lf_seed=k)
    inner = R.inner_sets(fit, S)
    (rv,), rt, ip = R.run_family('cat', fit, Xf, [Xv], [val], Xt_, inner, S)
    v = R.correct('cat', fit, ip, rv, tr.iloc[val]); t = R.correct('cat', fit, ip, rt, te)
    c = np.load(f'{C}/c_k{k}.npz')
    print(f'[flip-cat] c(flipped val labels) vs cached true-label c_k{k}: raw_v {md(rv, c["cat_raw_v"]):.3g} '
          f'ip {md(ip, c["cat_ip"]):.3g} v {md(v, c["cat_v"]):.3g} t {md(t, c["cat_t"]):.3g}; '
          f'AUC val {roc_auc_score(y[val[elig]], v[elig]):.5f} vs cached {roc_auc_score(y[val[elig]], c["cat_v"][elig]):.5f} '
          f'({time.time() - t0:.0f}s)', flush=True)

if '4' in stage:
    cols = list(F[BL['PROJECT']].columns)
    sus = [c for c in cols if any(t in c.lower() for t in ('id', 'index', 'uri', 'row', 'unnamed', 'popul', 'name'))]
    P = F[BL['PROJECT']].iloc[:n].astype(float)
    au = {c: roc_auc_score(y, P[c].fillna(P[c].median())) for c in cols}
    top = sorted(au.items(), key=lambda t: -abs(t[1] - 0.5))[:5]
    print(f'[cols] PROJECT ncol {len(cols)}; suspicious-name cols {sus}; max single-col AUC dev {top}', flush=True)
    print('[cols] train.csv columns:', list(tr.columns), flush=True)
    a_ = np.asarray(alb)
    print(f'[alb] n albums {len(np.unique(a_))}, train-only {len(np.setdiff1d(a_[:n], a_[n:]))}; '
          f'corr(alb code, row index) {np.corrcoef(a_, np.arange(len(a_)))[0, 1]:.3f}', flush=True)
