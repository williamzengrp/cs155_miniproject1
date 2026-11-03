"""r3_chp: OUTER-only variants of Charlie's pipeline (no inner fits; the residual correction is applied later in
chp_eval.py with the replica's cached inner-OOF residuals of the same fold / full fit).

  python -I -B chp_var.py cv VARIANT K0 K1 [SOFF]   std folds K0..K1-1, S = SEED0 + k + SOFF
                                                    -> cache/var_<VARIANT>_o<SOFF>_k<k>.npz
  python -I -B chp_var.py full VARIANT S_IDX        full refit (all train rows), S = SEED0 + 1000*(s+1) (+ TE seed
                                                    42 + 1000*(s+1) if the variant has 'tev')
                                                    -> cache/varfull_<VARIANT>_s<s>.npz
VARIANT = tokens joined by '-', e.g. 'dup05', 'lab-alb', 'it3000-dup05'. Tokens:
  CatBoost params: dup05 / dup1 (duplicate_power), d7 / d8 / d10 (depth), l2lo / l2hi, rsm25 / rsm7, rs2, lr2,
                   it<N> (iterations; staged predictions at STAGES <= N are stored as cat_v_<it> / cat_t_<it>)
  ET params:       etdp0 (no dup weights), etmf3 (max_features .3), etl4 (min_samples_leaf 4), etdN (no max depth)
  features:        lab  = te agent's BEST_SPEC label features (knnw_yr_40, teh_date_m10, cnt_date), fit rows
                          out-of-fold (inner 5-fold grouped by uri, seed k), val/test rows from all fit labels
                   teh  = 2 hierarchical date TEs (month -> year, date -> month), same cross-fit as Charlie's TE
                   copy = label-free copy detectors of features_final (nn distances, close counts)
                   alb  = label-free album-cluster id as a native CatBoost categorical (ET unchanged)
                   tev  = vary the date-TE cross-fit seed with SOFF (bagging over the TE noise)
ET is refit only when a feature or ET token is present (otherwise the replica's cached ET is reused)."""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
import pandas as pd
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
sys.path.insert(0, WORK + '/te')
sys.path.insert(0, WORK + '/main')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, GroupedDateTE, CFG, dup_weights
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings('ignore')
THREADS = int(os.environ.get('CHP_THREADS', '3'))
CACHE = WORK + '/r3_chp/cache'
STAGES = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]
CAT_TOK = {'dup05': dict(duplicate_power=0.5), 'dup1': dict(duplicate_power=1.0), 'd7': dict(depth=7),
           'd8': dict(depth=8), 'd10': dict(depth=10), 'l2lo': dict(l2=3.0), 'l2hi': dict(l2=30.0),
           'rsm25': dict(colsample=0.25), 'rsm7': dict(colsample=0.7), 'rs2': dict(random_strength=2.0),
           'lr2': dict(lr=2 * CFG['cat']['lr'])}
ET_TOK = {'etdp0': dict(duplicate_power=0.0), 'etmf3': dict(colsample=0.3), 'etl4': dict(min_leaf=4),
          'etdN': dict(depth=None)}
FEAT_TOK = {'lab', 'teh', 'copy', 'alb', 'tev'}
COPY_COLS = ['n_same_date_audio', 'a_n_close10', 'a_n_close20', 'a_n_close30', 'a_nn2_dist', 'a_global_nn_dist',
             'a_global_nn5_dist']

mode, variant = sys.argv[1], sys.argv[2]
toks = variant.split('-')
ccfg, ecfg = dict(CFG['cat']), dict(CFG['extra'])
ITERS = ccfg['rounds']
for t in toks:
    if t in CAT_TOK:
        ccfg.update(CAT_TOK[t])
    elif t in ET_TOK:
        ecfg.update(ET_TOK[t])
    elif t.startswith('it'):
        ITERS = int(t[2:])
    else:
        assert t in FEAT_TOK, t
REFIT_ET = any(t in ET_TOK or t in ('lab', 'teh', 'copy') for t in toks)

tr, te, y = H.load()
n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)
if 'copy' in toks:
    from features_final import build_features_final
    Xa_, Xt_ = build_features_final(tr, te)
    COPY_tr = Xa_[COPY_COLS].to_numpy(float)
    COPY_te = Xt_[COPY_COLS].to_numpy(float)
if 'lab' in toks or 'alb' in toks:
    from label_features import label_features, BEST_SPEC, get_ctx
    CTX = get_ctx(tr, te)
    XA_DUMMY = pd.DataFrame(index=range(n + m))
    ALB = CTX.key['alb']


class HierTE(GroupedDateTE):
    """Charlie's grouped date TE + 2 hierarchical columns (month TE shrunk toward the year TE, date TE shrunk toward
    that month TE; strength 10). Same cross-fitting of the fit rows (fold labels only)."""
    M = 10.0

    def _encode(self, keys, state):
        base = super()._encode(keys, state)
        prior, maps = state
        out = np.empty((len(keys), 2))
        for i, k in enumerate(keys):
            ty, cy = maps[0].get(k[0], (0., 0)) if k[0] is not None else (0., 0)
            yte = (ty + self.s * prior) / (cy + self.s)
            tm, cm = maps[1].get(k[1], (0., 0)) if k[1] is not None else (0., 0)
            mte = (tm + self.M * yte) / (cm + self.M)
            td, cd = maps[2].get(k[2], (0., 0)) if k[2] is not None else (0., 0)
            out[i] = (mte, (td + self.M * mte) / (cd + self.M))
        return np.column_stack([base, out])

    def fit_transform(self, keys, y, groups):   # = GroupedDateTE.fit_transform with 8 output columns
        from sklearn.model_selection import StratifiedGroupKFold
        y = np.asarray(y)
        out = np.full((len(keys), 8), np.nan)
        for a, b in StratifiedGroupKFold(self.folds, shuffle=True, random_state=self.seed).split(
                np.arange(len(keys)), y, groups):
            assert not set(groups[a]) & set(groups[b])
            out[b] = self._encode([keys[i] for i in b], self._build([keys[i] for i in a], y[a]))
        assert np.isfinite(out).all()
        self.state = self._build(keys, y)
        return out


def matrices(fit, applies, te_seed, lab_seed):
    """applies: list of TRAIN index arrays; test matrix always built. Returns float matrices (imputed) and album
    ids (or None) for fit / applies / test."""
    enc = (HierTE if 'teh' in toks else GroupedDateTE)(seed=te_seed)
    Efit = enc.fit_transform([R.ktr[i] for i in fit], y[fit], R.groups[fit])
    Xf = [R.Ptr[fit], Efit]
    Xa = [[R.Ptr[b], enc.transform([R.ktr[i] for i in b])] for b in applies]
    Xt = [R.Pte, enc.transform(R.kte)]
    if 'copy' in toks:
        Xf.append(COPY_tr[fit]); [x.append(COPY_tr[b]) for x, b in zip(Xa, applies)]; Xt.append(COPY_te)
    if 'lab' in toks:
        Lf, La = label_features(XA_DUMMY, y, fit, list(applies) + [np.arange(n, n + m)], tr, te, seed=lab_seed,
                                spec=BEST_SPEC)
        Xf.append(Lf.to_numpy(float)); [x.append(l.to_numpy(float)) for x, l in zip(Xa, La[:-1])]
        Xt.append(La[-1].to_numpy(float))
    Xf = np.column_stack(Xf); Xa = [np.column_stack(x) for x in Xa]; Xt = np.column_stack(Xt)
    imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xf)
    Xf, Xa, Xt = imp.transform(Xf), [imp.transform(x) for x in Xa], imp.transform(Xt)
    if 'alb' in toks:
        al = (ALB[fit], [ALB[b] for b in applies], ALB[n:])
    else:
        al = (None, [None] * len(applies), None)
    return (Xf, Xa, Xt), al


def to_pool(X, a, lab=None, w=None):
    if a is None:
        return Pool(X, label=lab, weight=w)
    df = pd.DataFrame(X, columns=[f'f{j}' for j in range(X.shape[1])])
    df['alb'] = np.asarray(a).astype(str)
    return Pool(df, label=lab, weight=w, cat_features=['alb'])


def outer(fit, mats, al, S):
    (Xf, Xa, Xt), (af, aa, at) = mats, al
    c = ccfg
    cb = CatBoostClassifier(iterations=ITERS, learning_rate=c['lr'], depth=c['depth'], l2_leaf_reg=c['l2'],
                            rsm=c['colsample'], random_strength=c['random_strength'], border_count=c['borders'],
                            loss_function='Logloss', verbose=False, allow_writing_files=False,
                            thread_count=THREADS, random_seed=S)
    cb.fit(to_pool(Xf, af, y[fit], None if c['duplicate_power'] == 0 else
                   dup_weights(R.uri[fit], c['duplicate_power'])))
    out = {}
    Pa = [to_pool(x, a) for x, a in zip(Xa, aa)]
    Pt = to_pool(Xt, at)
    for it in [s for s in STAGES if s <= ITERS] + ([ITERS] if ITERS not in STAGES else []):
        for j, P in enumerate(Pa):
            out[f'cat_v_{it}'] = cb.predict_proba(P, ntree_end=it)[:, 1]
        out[f'cat_t_{it}'] = cb.predict_proba(Pt, ntree_end=it)[:, 1]
    if REFIT_ET:
        e = ecfg
        et = ExtraTreesClassifier(n_estimators=e['rounds'], max_depth=e['depth'], min_samples_leaf=e['min_leaf'],
                                  max_features=e['colsample'], n_jobs=THREADS, random_state=S)
        et.fit(Xf, y[fit], sample_weight=dup_weights(R.uri[fit], e['duplicate_power']))
        for x in Xa:
            out['et_v'] = et.predict_proba(x)[:, 1]
        out['et_t'] = et.predict_proba(Xt)[:, 1]
    return out


if mode == 'cv':
    k0, k1 = int(sys.argv[3]), int(sys.argv[4])
    soff = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    splits = H.make_splits(tr, y, n_rep=5)
    for k in range(k0, min(k1, len(splits))):
        path = f'{CACHE}/var_{variant}_o{soff}_k{k}.npz'
        if os.path.exists(path):
            continue
        r, fit, val, elig = splits[k]
        t0 = time.time()
        mats, al = matrices(fit, [val], 42 + (soff if 'tev' in toks else 0), k + soff)
        out = outer(fit, mats, al, SEED0 + k + soff)
        np.savez(path, **out)
        print(f'[cv {variant} soff={soff} k={k}] {time.time() - t0:.1f}s', flush=True)
elif mode == 'full':
    fit = np.arange(n)
    for s in [int(x) for x in sys.argv[3].split(',')]:
        path = f'{CACHE}/varfull_{variant}_s{s}.npz'
        if os.path.exists(path):
            continue
        t0 = time.time()
        S = SEED0 + 1000 * (s + 1)
        mats, al = matrices(fit, [], 42 + (1000 * (s + 1) if 'tev' in toks else 0), 1000 * (s + 1))
        out = outer(fit, mats, al, S)
        np.savez(path, **out)
        print(f'[full {variant} s={s}] {time.time() - t0:.1f}s', flush=True)
