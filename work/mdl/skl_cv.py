"""Non-tree sklearn models (MLP, SVM-RBF, spline logistic regression, kNN, HGB) under the shared harness.

usage:
  skl_cv.py tune <family>                 -> small grid on H.make_splits(n_rep=2, seed0=7000), nothing saved
  skl_cv.py std <family> '<json params>' <save_name>
                                          -> 5-repeat standard splits, blend report, saves results/<save_name>.npz
All preprocessing (median fill, quantile-normal / standard scaling, spline basis) is fit on fit rows only.
"""
import os, sys, json, time, warnings, itertools
sys.path.insert(0, './work/mdl')
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
import common as C
warnings.filterwarnings('ignore')
NJ = int(os.environ.get('NJ', '2'))
H = C.H

COMPACT_DROP = ['release_year', 'days_since_release', 't_log_age', 't_decade', 's_doy', 's_week', 's_quarter',
                's_mon_sin', 's_mon_cos', 's_dow_sin', 's_dow_cos', 's_doy_sin', 's_doy_cos', 's_is_friday',
                's_is_tuesday', 's_holiday', 's_summer']
TOP_TIME = ['t_frac_year', 'n_same_date', 'n_same_year', 'n_same_month']   # for spline x time interactions

tr, te, y, X0, Xt0 = C.data()


def feats(name):
    if name == 'all':
        return X0, Xt0
    drop = list(COMPACT_DROP) + [c for c in X0.columns if c.startswith('g_dev_')]
    if name == 'compact':
        return X0.drop(columns=drop), Xt0.drop(columns=drop)
    raise ValueError(name)


def fit_predict_one(P, Afit, yfit, others, seed):
    fam = P['fam']
    if fam == 'mlp':
        m = MLPClassifier(hidden_layer_sizes=tuple(P['h']), alpha=P['alpha'], learning_rate_init=P.get('lr', 1e-3),
                          batch_size=P.get('bs', 128), max_iter=P.get('max_iter', 300), early_stopping=True,
                          validation_fraction=0.15, n_iter_no_change=P.get('patience', 20), random_state=seed)
        m.fit(Afit, yfit)
        return [m.predict_proba(o)[:, 1] for o in others]
    if fam == 'svm':
        g = P['gamma_mult'] / Afit.shape[1]           # 'scale' == 1/(n_features*var) and var~1 after scaling
        m = SVC(C=P['C'], kernel='rbf', gamma=g, random_state=seed)
        m.fit(Afit, yfit)
        return [m.decision_function(o) for o in others]
    if fam == 'lr':
        m = LogisticRegression(C=P['C'], max_iter=3000)
        m.fit(Afit, yfit)
        return [m.decision_function(o) for o in others]
    if fam == 'knn':
        m = KNeighborsClassifier(n_neighbors=P['k'], weights=P.get('weights', 'uniform'), n_jobs=1)
        m.fit(Afit, yfit)
        return [m.predict_proba(o)[:, 1] for o in others]
    if fam == 'hgb':
        m = HistGradientBoostingClassifier(learning_rate=P['lr'], max_iter=P['iters'], max_leaf_nodes=P['leaves'],
                                           min_samples_leaf=P['msl'], l2_regularization=P.get('l2', 0.0),
                                           max_features=P.get('mf', 1.0), early_stopping=False, random_state=seed)
        m.fit(Afit, yfit)
        return [m.predict_proba(o)[:, 1] for o in others]
    raise ValueError(fam)


def transform(P, Xfit, Xothers):
    """Per-fold preprocessing. Spline LR: quantile-normal -> spline basis (+ optional time interactions)."""
    scaler = P.get('scaler', 'quantile')
    A, others = C.prep(Xfit, Xothers, scaler='none' if P['fam'] == 'hgb' else scaler)
    if P['fam'] == 'lr' and P.get('spline'):
        st = SplineTransformer(n_knots=P['knots'], degree=3, knots='quantile', extrapolation='constant')
        B = st.fit_transform(A)
        Bo = [st.transform(o) for o in others]
        if P.get('inter'):
            # pairwise products between the (scaled) time/density columns and all other scaled columns
            cols = list(Xfit.columns)
            ti = [cols.index(c) for c in TOP_TIME if c in cols]
            def inter(M):
                return np.hstack([M[:, [i]] * np.delete(M, i, axis=1) for i in ti])
            B = np.hstack([B, inter(A)])
            Bo = [np.hstack([b, inter(o)]) for b, o in zip(Bo, others)]
        # standardise basis columns (fit rows only) so one C fits all
        mu, sd = B.mean(0), B.std(0) + 1e-9
        return (B - mu) / sd, [(b - mu) / sd for b in Bo]
    return A, others


def make_fp(P, X, Xt, with_test, seeds):
    times = []

    def fp(fit_idx, val_idx, k):
        t = time.time()
        Afit, others = transform(P, X.iloc[fit_idx], [X.iloc[val_idx]] + ([Xt] if with_test else []))
        outs = [fit_predict_one(P, Afit, y[fit_idx], others, seed=1000 * k + s) for s in seeds]
        pv = np.mean([o[0] for o in outs], axis=0)
        pt = np.mean([o[1] for o in outs], axis=0) if with_test else None
        times.append(time.time() - t)
        return pv, pt
    fp.times = times
    return fp


def full_fit(P, X, Xt, seeds):
    Afit, (At,) = transform(P, X, [Xt])
    return np.mean([fit_predict_one(P, Afit, y, [At], seed=77 + s)[0] for s in seeds], axis=0)


GRIDS = {
    'svm': [dict(fam='svm', feats=f, C=c, gamma_mult=g) for f in ('all', 'compact') for c in (0.3, 1.0, 3.0)
            for g in (0.5, 1.0, 2.0)],
    'lr': ([dict(fam='lr', feats=f, spline=True, knots=kn, C=c) for f in ('all', 'compact') for kn in (5, 8)
            for c in (0.003, 0.01, 0.03)] +
           [dict(fam='lr', feats='compact', spline=True, knots=5, C=c, inter=True) for c in (0.003, 0.01, 0.03)]),
    'knn': [dict(fam='knn', feats=f, k=k, weights=w) for f in ('all', 'compact') for k in (25, 50, 100, 200)
            for w in ('uniform', 'distance')],
    'mlp': [dict(fam='mlp', feats=f, h=h, alpha=a, seeds=3) for f in ('all', 'compact') for h in ([64], [128, 64])
            for a in (1e-3, 1e-2, 1e-1)],
    'hgb': [dict(fam='hgb', feats='all', lr=0.03, iters=it, leaves=lv, msl=msl, l2=1.0, mf=mf)
            for it in (300, 600) for lv in (15, 31) for msl in (10, 30) for mf in (0.5,)],
}


def run(P, which, save_name=None):
    X, Xt = feats(P.get('feats', 'all'))
    seeds = list(range(P.get('seeds', 1)))
    splits = H.make_splits(tr, y, n_rep=2, seed0=7000) if which == 'tune' else H.make_splits(tr, y, n_rep=5)
    with_test = which == 'std'
    fp = make_fp(P, X, Xt, with_test, seeds)
    res = H.run_cv(fp, splits, y, tag=json.dumps(P), verbose=False)
    print(f'{which} AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)}  {np.mean(fp.times):.1f}s/fold  {json.dumps(P)}'
          + ('  ' + C.tune_gain(y, res['oof']) if which == 'tune' else ''), flush=True)
    if which == 'std':
        C.blend_report(y, res['oof'], label=P['fam'])
        if save_name:
            t = time.time()
            tf = full_fit(P, X, Xt, list(range(max(3, len(seeds)))) if P['fam'] in ('mlp', 'hgb') else [0])
            print(f'  full fit {time.time() - t:.1f}s; corr(full, foldavg) '
                  f'{np.corrcoef(tf, res["test_foldavg"])[0, 1]:.4f}', flush=True)
            H.save(save_name, res, tf)
            print('  saved', save_name, flush=True)
    return res


if __name__ == '__main__':
    which, fam = sys.argv[1], sys.argv[2]
    with threadpool_limits(NJ):
        if which == 'tune':
            for P in GRIDS[fam]:
                run(P, 'tune')
        else:
            run(json.loads(sys.argv[3]), 'std', sys.argv[4] if len(sys.argv) > 4 else None)
