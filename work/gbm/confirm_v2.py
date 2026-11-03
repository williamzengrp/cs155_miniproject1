"""Coordinator request: re-tuned params vs teammate params on the 55-feature v2 set (fe/features_v2.py),
same standard 5x5 splits and model seeds (fold index k) for both.

usage: GBM_FEATS=v2 python -I confirm_v2.py <model> [<model> ...]   (params from gbm/sel_<model>_none.json)
"""
import sys
import os
import json
sys.path.insert(0, './work/gbm')
import numpy as np
import lib as L

assert L.M.FEATS == 'v2', 'run with GBM_FEATS=v2'
splits = L.H.make_splits(L.tr, L.y, n_rep=5)
for name in sys.argv[1:]:
    sel = json.load(open(os.path.join(L.M.GBM, f'sel_{name}_none.json')))
    rt = L.run(name, sel['params'], splits, verbose=False)
    rb = L.run(name, L.BASE[name], splits, verbose=False)
    L.log(f'- 5x5 confirm on v2 (55 feats) {name}: tuned {rt["auc"]:.5f} {np.round(rt["aucs"], 4)} | teammate '
          f'params {rb["auc"]:.5f} {np.round(rb["aucs"], 4)} | diff {L.pdiff(rt, rb)}')
    np.savez(os.path.join(L.M.GBM, f'eval_{name}_none_v2.npz'), tuned_oof=rt['oof'], base_oof=rb['oof'],
             tuned_aucs=rt['aucs'], base_aucs=rb['aucs'])
