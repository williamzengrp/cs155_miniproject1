"""Zoo of learners / objectives on the SAME inputs as the final tree members (work/main/final_cv.py):
58 label-free features + 3 out-of-fold label features (te/label_features.py BEST_SPEC), optional label-free
categorical keys (release-date string, year-month, album-cluster id) for CatBoost.

Usage
  python -I zoo.py run SEED0 NREP PREFIX NSEEDS model [model ...]
      CV on H.make_splits(n_rep=NREP, seed0=SEED0) with NSEEDS model seeds per fold (seed = k + 100*j, label
      features fixed with seed=k exactly as final_cv.py), OOF = mean over seeds; then (env DO_FULL=1, default)
      full refit on all training rows with seeds 0,1,2 (label features seed s, model seed s, like final_cv.py).
      Saves results/<PREFIX>_<model>.npz via H.save (oof, aucs, test_foldavg, test_full).
  python -I zoo.py cv SEED0 NREP OFFSET model ...   raw single-seed OOF -> r2_zoo/raw/ (screening)
  env NJ (threads, default 2).
Leakage: label features exactly as final_cv.py (fit_idx labels only, inner folds grouped by uri). CatBoost's
ordered target statistics for the categorical keys are computed by CatBoost from the fit rows' labels only.
"""
import os, sys, json, time, pickle, warnings
import numpy as np, pandas as pd
WORK = './work'
ZOO = WORK + '/r2_zoo'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/main'); sys.path.insert(0, WORK + '/te')
import harness as H
warnings.filterwarnings('ignore')
NJ = int(os.environ.get('NJ', '2'))
P = json.load(open(WORK + '/main/final_params.json'))
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------------------------------------ inputs
def get_cache(seed0, nrep):
    f = f'{ZOO}/cache/lf_{seed0}_{nrep}.pkl'
    if not os.path.exists(f):
        from features_final import build_features_final
        from label_features import label_features, BEST_SPEC, get_ctx
        tr, te, y = H.load()
        X, Xt = build_features_final(tr, te)
        XA = pd.concat([X, Xt], ignore_index=True)
        n, m = len(tr), len(te)
        test_idx = np.arange(n, n + m)
        folds = []
        for k, (r, fit_idx, val_idx, elig) in enumerate(H.make_splits(tr, y, n_rep=nrep, seed0=seed0)):
            F_fit, (F_val, F_te) = label_features(XA, y, fit_idx, [val_idx, test_idx], tr, te, seed=k, spec=BEST_SPEC)
            folds.append(dict(r=r, fit_idx=fit_idx, val_idx=val_idx, elig=elig, F_fit=F_fit, F_val=F_val, F_te=F_te))
        full = []
        for s in range(3):
            F_fit, (F_te,) = label_features(XA, y, np.arange(n), [test_idx], tr, te, seed=s, spec=BEST_SPEC)
            full.append(dict(F_fit=F_fit, F_te=F_te))
        ctx = get_ctx(tr, te)
        raw = pd.concat([tr, te], ignore_index=True)['track_album_release_date'].astype(str).to_numpy()
        cats = dict(date=ctx.key['date'], ym=ctx.key['ym'], year=ctx.key['year'], alb=ctx.key['alb'], raw=raw,
                    t=ctx.t, uri=ctx.uri)
        pickle.dump(dict(X=X, Xt=Xt, y=y, folds=folds, full=full, cats=cats, seed0=seed0, nrep=nrep), open(f, 'wb'))
    return pickle.load(open(f, 'rb'))


class Inputs:
    def __init__(self, D):
        self.D = D
        self.X, self.Xt, self.y = D['X'], D['Xt'], D['y']
        self.n, self.m = len(self.X), len(self.Xt)
        self.fill = self.X.median()
        C = D['cats']
        self.C = pd.DataFrame({'c_date': C['date'].astype(str), 'c_ym': C['ym'].astype(str),
                               'c_alb': C['alb'].astype(str), 'c_year': C['year'].astype(str)})

    def mat(self, rows, F, view):
        """rows: indices into train (0..n-1) or test (n..n+m-1) pool; F: label-feature frame aligned to rows."""
        rows = np.asarray(rows)
        A = pd.concat([self.X, self.Xt], ignore_index=True).iloc[rows].reset_index(drop=True)
        M = pd.concat([A, F.reset_index(drop=True)], axis=1)
        if view == 'fill':
            M = M.fillna(pd.concat([self.fill, F.median()]))
        elif view.startswith('cat'):
            cols = {'cat3': ['c_date', 'c_ym', 'c_alb'], 'cat_alb': ['c_alb'], 'cat_date': ['c_date'],
                    'cat_ym': ['c_ym'], 'cat_da': ['c_date', 'c_alb']}[view]
            M = pd.concat([M, self.C.iloc[rows][cols].reset_index(drop=True)], axis=1)
        return M


# ------------------------------------------------------------------------------------------------ models
MONO_COLS = {'release_year': -1, 't_frac_year': -1, 't_log_age': 1, 't_decade': -1, 'days_since_release': 1}


class Model:
    view = 'num'

    def __init__(self, name, seed):
        self.name, self.seed = name, seed

    def fit(self, M, y):
        self.est = self.make(M).fit(M, y)
        return self

    def predict(self, M):
        return self.est.predict_proba(M)[:, 1]


def lgbm_base(**kw):
    p = dict(P['lgbm']); p.update(kw)
    return p


class LGB(Model):
    def __init__(self, name, seed, **kw):
        super().__init__(name, seed); self.kw = kw

    def make(self, M):
        from lightgbm import LGBMClassifier
        kw = dict(self.kw)
        if 'monotone' in kw:
            kw.pop('monotone')
            kw['monotone_constraints'] = [MONO_COLS.get(c, 0) for c in M.columns]
        return LGBMClassifier(**kw, random_state=self.seed, n_jobs=NJ, verbose=-1)


class XGB(Model):
    def __init__(self, name, seed, **kw):
        super().__init__(name, seed); self.kw = kw

    def make(self, M):
        from xgboost import XGBClassifier
        kw = dict(self.kw)
        if 'monotone' in kw:
            kw.pop('monotone')
            kw['monotone_constraints'] = '(' + ','.join(str(MONO_COLS.get(c, 0)) for c in M.columns) + ')'
        return XGBClassifier(**kw, tree_method='hist', random_state=self.seed, n_jobs=NJ)


class XGBRank(Model):
    """XGBoost pairwise ranking objective with the whole fit set as ONE query (smooth AUC surrogate)."""
    def __init__(self, name, seed, **kw):
        super().__init__(name, seed); self.kw = kw

    def fit(self, M, y):
        from xgboost import XGBRanker
        self.est = XGBRanker(**self.kw, tree_method='hist', random_state=self.seed, n_jobs=NJ)
        self.est.fit(M, y, qid=np.zeros(len(y), int))
        return self

    def predict(self, M):
        return self.est.predict(M)


class LGBRankNet(Model):
    """LightGBM with a custom all-pairs RankNet objective: loss = mean over (pos,neg) pairs of
    log(1+exp(-(s_pos - s_neg))) -> exact smooth AUC surrogate (fit set = one query)."""
    def __init__(self, name, seed, **kw):
        super().__init__(name, seed); self.kw = kw

    def fit(self, M, y):
        from lightgbm import LGBMRegressor
        pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
        npos, nneg = len(pos), len(neg)

        def obj(y_true, s):
            D = s[neg][None, :] - s[pos][:, None]          # (npos, nneg): s_neg - s_pos
            sg = 1.0 / (1.0 + np.exp(-D))
            h = sg * (1 - sg)
            g = np.zeros(len(s)); hh = np.zeros(len(s))
            g[pos] = -sg.sum(1) / nneg
            g[neg] = sg.sum(0) / npos
            hh[pos] = h.sum(1) / nneg
            hh[neg] = h.sum(0) / npos
            return g, np.maximum(hh, 1e-6)
        self.est = LGBMRegressor(objective=obj, **self.kw, random_state=self.seed, n_jobs=NJ, verbose=-1)
        self.est.fit(M, y.astype(float))
        return self

    def predict(self, M):
        return self.est.predict(M, raw_score=True)


class Cat(Model):
    def __init__(self, name, seed, view='num', **kw):
        super().__init__(name, seed); self.kw = kw; self.view = view

    def make(self, M):
        from catboost import CatBoostClassifier
        cats = [c for c in M.columns if c.startswith('c_')]
        return CatBoostClassifier(**self.kw, cat_features=cats or None, random_seed=self.seed, verbose=0,
                                  thread_count=NJ, allow_writing_files=False)


class HGB(Model):
    def __init__(self, name, seed, **kw):
        super().__init__(name, seed); self.kw = kw

    def make(self, M):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**self.kw, early_stopping=False, random_state=self.seed)


class SK(Model):
    view = 'fill'

    def __init__(self, name, seed, ctor, **kw):
        super().__init__(name, seed); self.ctor = ctor; self.kw = kw

    def make(self, M):
        return self.ctor(**self.kw, random_state=self.seed, n_jobs=NJ)


def bagged_dt(**kw):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    ne = kw.pop('n_estimators'); ms = kw.pop('max_samples', 1.0)
    rs = kw.pop('random_state'); nj = kw.pop('n_jobs')
    return BaggingClassifier(DecisionTreeClassifier(**kw), n_estimators=ne, max_samples=ms, bootstrap=True,
                             random_state=rs, n_jobs=nj)


def model(name, seed):
    L, X_, C_ = P['lgbm'], P['xgb'], P['cat']
    if name == 'lgbm_ref':
        return LGB(name, seed, **lgbm_base())
    if name == 'cat_ref':                      # == fkt_cat (final_params cat, no categorical keys)
        return Cat(name, seed, view='num', **C_)
    if name == 'xgb_ref':
        return XGB(name, seed, **X_)
    if name == 'lgb_dart':
        return LGB(name, seed, **lgbm_base(boosting_type='dart', n_estimators=500, learning_rate=0.05, drop_rate=0.1,
                                           skip_drop=0.5, max_drop=50, drop_seed=seed))
    if name == 'lgb_goss':
        p = lgbm_base(data_sample_strategy='goss', top_rate=0.3, other_rate=0.2)
        p.pop('subsample'); p.pop('subsample_freq')
        return LGB(name, seed, **p)
    if name == 'lgb_d2':
        return LGB(name, seed, n_estimators=2500, learning_rate=0.01, max_depth=2, num_leaves=4, min_child_samples=10,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0)
    if name == 'lgb_d1':
        return LGB(name, seed, n_estimators=3000, learning_rate=0.02, max_depth=1, num_leaves=2, min_child_samples=10,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0)
    if name == 'xgb_d3':
        return XGB(name, seed, n_estimators=3000, learning_rate=0.005, max_depth=3, min_child_weight=2,
                   subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0)
    if name == 'lgb_deep':
        return LGB(name, seed, n_estimators=700, learning_rate=0.01, num_leaves=256, max_depth=-1, min_child_samples=3,
                   subsample=0.7, subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0)
    if name == 'xgb_deep':
        return XGB(name, seed, n_estimators=1500, learning_rate=0.005, max_depth=14, min_child_weight=1,
                   subsample=0.75, colsample_bytree=0.3, reg_lambda=1.0)
    if name == 'xgb_rank':
        return XGBRank(name, seed, objective='rank:pairwise', lambdarank_pair_method='mean',
                       lambdarank_num_pair_per_sample=8, **X_)
    if name == 'lgb_ranknet':
        return LGBRankNet(name, seed, n_estimators=662, learning_rate=0.05, num_leaves=30, max_depth=9,
                          min_child_samples=7, subsample=0.75, subsample_freq=1, colsample_bytree=0.7156,
                          reg_lambda=0.104)
    if name == 'lgb_mono':
        return LGB(name, seed, monotone=True, **lgbm_base())
    if name == 'xgb_mono':
        return XGB(name, seed, monotone=True, **X_)
    if name == 'lgb_linear':
        return LGB(name, seed, **lgbm_base(linear_tree=True, linear_lambda=1.0))
    if name == 'lgb_bal':
        return LGB(name, seed, **lgbm_base(is_unbalance=True))
    if name == 'hgb':
        return HGB(name, seed, learning_rate=0.03, max_iter=600, max_leaf_nodes=24, min_samples_leaf=8,
                   l2_regularization=1.0, max_features=0.5)
    if name == 'bag_dt':
        return SK(name, seed, bagged_dt, n_estimators=150, max_samples=0.7, max_depth=16, min_samples_leaf=2, criterion='entropy')
    if name.startswith('cat_') and name[4:] in ('cat3', 'alb', 'date', 'ym', 'da'):
        v = 'cat3' if name == 'cat_cat3' else 'cat_' + name[4:]
        return Cat(name, seed, view=v, **C_)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ runners
def cv(I, mname, offsets, verbose=True):
    folds = I.D['folds']
    R = max(f['r'] for f in folds) + 1
    oof = np.full((R, I.n), np.nan)
    tst = np.zeros(I.m)
    test_rows = np.arange(I.n, I.n + I.m)
    t0 = time.time()
    view = model(mname, 0).view
    for k, f in enumerate(folds):
        Mf = I.mat(f['fit_idx'], f['F_fit'], view)
        Mv = I.mat(f['val_idx'], f['F_val'], view)
        Mt = I.mat(test_rows, f['F_te'], view)
        pv, pt = 0, 0
        for off in offsets:
            mdl = model(mname, k + off).fit(Mf, I.y[f['fit_idx']])
            pv = pv + mdl.predict(Mv) / len(offsets)
            pt = pt + mdl.predict(Mt) / len(offsets)
        oof[f['r'], f['val_idx'][f['elig']]] = np.asarray(pv)[f['elig']]
        tst += pt / len(folds)
    aucs = np.array([roc_auc_score(I.y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in range(R)])
    if verbose:
        print(f'  [{mname} offs={offsets}] AUC {aucs.mean():.5f} per-rep {np.round(aucs, 4)} ({time.time()-t0:.0f}s)',
              flush=True)
    return dict(oof=oof, aucs=aucs, auc=float(aucs.mean()), test_foldavg=tst)


def full_refit(I, mname):
    test_rows = np.arange(I.n, I.n + I.m)
    view = model(mname, 0).view
    out = np.zeros(I.m)
    for s, fr in enumerate(I.D['full']):
        mdl = model(mname, s).fit(I.mat(np.arange(I.n), fr['F_fit'], view), I.y)
        out += mdl.predict(I.mat(test_rows, fr['F_te'], view)) / len(I.D['full'])
    return out


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'run':
        seed0, nrep, prefix, nseeds = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
        I = Inputs(get_cache(seed0, nrep))
        os.makedirs(f'{ZOO}/raw', exist_ok=True)
        for mname in sys.argv[6:]:
            parts = []
            for j in range(nseeds):   # one pass per model seed offset; reuse raw single-seed passes if cached
                fr = f'{ZOO}/raw/{seed0}_{nrep}_{mname}_off{100 * j}.npz'
                if os.path.exists(fr):
                    parts.append({k: v for k, v in np.load(fr).items()})
                else:
                    rj = cv(I, mname, [100 * j])
                    np.savez(fr, oof=rj['oof'], aucs=rj['aucs'], test_foldavg=rj['test_foldavg'])
                    parts.append(rj)
            oof = np.mean([p_['oof'] for p_ in parts], 0)
            R = oof.shape[0]
            aucs = np.array([roc_auc_score(I.y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in range(R)])
            res = dict(oof=oof, aucs=aucs, auc=float(aucs.mean()),
                       test_foldavg=np.mean([p_['test_foldavg'] for p_ in parts], 0))
            print(f'  [{mname} {nseeds} seeds] AUC {res["auc"]:.5f} per-rep {np.round(aucs, 4)}', flush=True)
            tf = full_refit(I, mname) if os.environ.get('DO_FULL', '1') == '1' else None
            H.save(f'{prefix}_{mname}', res, tf)
            c = np.nan if tf is None else np.corrcoef(tf, res['test_foldavg'])[0, 1]
            print(f'[{prefix}_{mname}] saved; AUC {res["auc"]:.5f} corr(foldavg,full)={c:.4f}', flush=True)
    elif mode == 'cv':
        seed0, nrep, off = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        I = Inputs(get_cache(seed0, nrep))
        os.makedirs(f'{ZOO}/raw', exist_ok=True)
        for mname in sys.argv[5:]:
            res = cv(I, mname, [off])
            np.savez(f'{ZOO}/raw/{seed0}_{nrep}_{mname}_off{off}.npz', oof=res['oof'], aucs=res['aucs'],
                     test_foldavg=res['test_foldavg'])
