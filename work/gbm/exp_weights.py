"""Item 2: training-set handling of duplicate songs, baseline params, standard 5x5 test-mimicking CV."""
import sys
sys.path.insert(0, './work/gbm')
import numpy as np, pandas as pd
import lib as L

models = sys.argv[1].split(',') if len(sys.argv) > 1 else ['lgbm', 'xgb']
schemes = ['none', 'inv', 'dedup', 'dup0.5', 'dup0.25', 'dup0', 'invdup0.5']
splits = L.H.make_splits(L.tr, L.y, n_rep=5)


def fp_ncopies(name, params):
    """n_copies (within the fit part) as a feature; predictions made with n_copies=1 (test-like)."""
    A, _ = L.DATA[name]
    def fp(fit_idx, val_idx, k):
        Af = A.iloc[fit_idx].copy(); Af['n_copies'] = L.M.uri_counts(L.URI[fit_idx])
        Av = A.iloc[val_idx].copy(); Av['n_copies'] = 1
        m = L.M.make_model(name, params, seed=k, n_jobs=L.NJ); m.fit(Af, L.y[fit_idx])
        return m.predict_proba(Av)[:, 1], None
    return fp


for name in models:
    L.log(f'\n### weights exp: {name} (baseline params)')
    R = {}
    for sc in schemes:
        R[sc] = L.run(name, L.BASE[name], splits, scheme=sc, tag=f'{name} {sc}', verbose=False)
        d = '' if sc == 'none' else L.pdiff(R[sc], R['none'])
        L.log(f'- {name} {sc:10s} AUC {R[sc]["auc"]:.5f} per-rep {np.round(R[sc]["aucs"], 4)}  vs none {d}')
    res = L.H.run_cv(fp_ncopies(name, L.BASE[name]), splits, L.y, verbose=False)
    L.log(f'- {name} ncopies-feat AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)}  vs none {L.pdiff(res, R["none"])}')
