"""r3_aug: seed-average saved blocks (same variant, different SEEDOFF): mean OOF / fold-avg test / test_full.
python -I combine.py OUT NAME1 NAME2 [...]"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
out, names = sys.argv[1], sys.argv[2:]
tr, te, y = H.load()
R = [H.load_res(n) for n in names]
oof = np.mean([r['oof'] for r in R], axis=0)
aucs = np.array([roc_auc_score(y[~np.isnan(oof[k])], oof[k][~np.isnan(oof[k])]) for k in range(oof.shape[0])])
res = dict(oof=oof, aucs=aucs, auc=float(aucs.mean()), test_foldavg=np.mean([r['test_foldavg'] for r in R], axis=0))
tf = np.mean([r['test_full'] for r in R], axis=0)
H.save(out, res, tf)
print(out, 'AUC', round(aucs.mean(), 5), np.round(aucs, 4), ' members', [round(r['aucs'].mean(), 5) for r in R])
