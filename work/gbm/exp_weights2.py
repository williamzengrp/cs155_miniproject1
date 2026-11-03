"""Item 2 (cont.): up-weighting multi-copy songs (trend in exp_weights suggested more weight is not worse)."""
import sys
sys.path.insert(0, './work/gbm')
import numpy as np
import lib as L
splits = L.H.make_splits(L.tr, L.y, n_rep=5)
for name in ['lgbm', 'xgb']:
    ref = L.H.load_res('base_' + name)          # identical to scheme 'none' (verified above)
    L.log(f'\n### weights exp (up-weighting): {name} (baseline params)')
    for sc in ['dup1.5', 'dup2']:
        r = L.run(name, L.BASE[name], splits, scheme=sc, verbose=False)
        L.log(f'- {name} {sc:10s} AUC {r["auc"]:.5f} per-rep {np.round(r["aucs"], 4)}  vs none {L.pdiff(r, ref)}')
