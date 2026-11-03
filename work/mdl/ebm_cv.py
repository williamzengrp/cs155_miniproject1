"""Explainable Boosting Machine (interpret 0.7.8, GA2M: per-feature shape functions + pairwise interactions).
Runs in work/mdl/venv.
usage: ebm_cv.py tune            -> small grid on tuning splits (n_rep=2, seed0=7000)
       ebm_cv.py std '<json>' <save_name>
EBM handles NaN natively ('separate' bin) and needs no scaling, so raw features are passed.
"""
import os, sys, json, time, warnings
sys.path.insert(0, './work/mdl')
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
import common as C
warnings.filterwarnings('ignore')
H = C.H
NJ = int(os.environ.get('NJ', '2'))
tr, te, y, X0, Xt0 = C.data()
COMPACT_DROP = ['release_year', 'days_since_release', 't_log_age', 't_decade', 's_doy', 's_week', 's_quarter',
                's_mon_sin', 's_mon_cos', 's_dow_sin', 's_dow_cos', 's_doy_sin', 's_doy_cos', 's_is_friday',
                's_is_tuesday', 's_holiday', 's_summer'] + [c for c in X0.columns if c.startswith('g_dev_')]


def feats(name):
    if name in ('v2', 'compact_v2'):
        X, Xt = C.data_v2()
        return (X, Xt) if name == 'v2' else (X.drop(columns=C.DATE_REDUNDANT), Xt.drop(columns=C.DATE_REDUNDANT))
    return (X0, Xt0) if name == 'all' else (X0.drop(columns=COMPACT_DROP), Xt0.drop(columns=COMPACT_DROP))


def model(P, seed):
    kw = {k: v for k, v in P.items() if k not in ('feats',)}
    return ExplainableBoostingClassifier(n_jobs=NJ, random_state=seed, **kw)


def cv(P, which, save_name=None):
    X, Xt = feats(P['feats'])
    splits = H.make_splits(tr, y, n_rep=2, seed0=7000) if which == 'tune' else H.make_splits(tr, y, n_rep=5)
    with_test = which == 'std'
    times = []

    def fp(fi, vi, k):
        t = time.time()
        m = model(P, k)
        m.fit(X.iloc[fi], y[fi])
        pv = m.predict_proba(X.iloc[vi])[:, 1]
        pt = m.predict_proba(Xt)[:, 1] if with_test else None
        times.append(time.time() - t)
        return pv, pt
    res = H.run_cv(fp, splits, y, verbose=False)
    print(f'{which} AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)}  {np.mean(times):.1f}s/fold  {json.dumps(P)}'
          + ('  ' + C.tune_gain(y, res['oof']) if which == 'tune' else ''), flush=True)
    if which == 'tune':
        tag = '_'.join(f'{k}{v}' for k, v in P.items())
        np.savez(os.path.join(C.MDL, 'cache', f'tune_ebm_{tag}.npz'), oof=res['oof'], aucs=res['aucs'])
    if which == 'std':
        C.blend_report(y, res['oof'], label='ebm')
        if save_name:
            t = time.time()
            tf = np.mean([model(P, 500 + s).fit(X, y).predict_proba(Xt)[:, 1] for s in range(2)], axis=0)
            print(f'  full fit x2 {time.time() - t:.1f}s; corr(full, foldavg) '
                  f'{np.corrcoef(tf, res["test_foldavg"])[0, 1]:.4f}', flush=True)
            H.save(save_name, res, tf)
            print('  saved', save_name, flush=True)
    return res


GRID = [dict(feats='compact', outer_bags=8),
        dict(feats='all', outer_bags=8),
        dict(feats='compact', outer_bags=8, interactions=0)]

if __name__ == '__main__':
    if sys.argv[1] == 'tune':
        for P in GRID:
            cv(P, 'tune')
    else:
        cv(json.loads(sys.argv[2]), 'std', sys.argv[3] if len(sys.argv) > 3 else None)
