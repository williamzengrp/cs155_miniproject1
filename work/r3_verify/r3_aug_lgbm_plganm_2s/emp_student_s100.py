"""Empirical val-label invariance test of the aug2.py student pipeline (plganm variant, SEEDOFF=0), fold k.
Teacher OOF blocks are held fixed (precomputed); label features are RECOMPUTED with modified y."""
import sys, time, numpy as np
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/te')
from sklearn.metrics import roc_auc_score
K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
src = open(WORK + '/r3_aug/aug2.py').read()
src = src[:src.index('t0 = time.time()')]
sys.argv = ['aug2.py', 'lgbm', 'pl=chga:val:sharp0.5:1;noise=0.1:2:0.5;mix=1:0.5:0.5', '5']
import os; os.environ['NJ'] = '1'; os.environ['SEEDOFF'] = '100'
ns = {'__name__': 'aug2ns'}
exec(compile(src, 'aug2_head', 'exec'), ns)
from label_features import label_features, BEST_SPEC
H = ns['H']; y0 = ns['y'].copy(); n, m = ns['n'], ns['m']
r, fit_idx, val_idx, elig = ns['splits'][K]
XA = np.zeros((n + m, 1)); test_idx = np.arange(n, n + m)
saved = H.load_res('r3_aug_lgbm_plganm_s100')['oof'][r, val_idx[elig]]
t0 = time.time()
# 1. reproduce saved OOF with cached LF
pv_c, pt_c = ns['run_fold'](K)
print('fold', K, 'rep', r, 'reproduce saved OOF: maxabs', np.abs(pv_c[elig] - saved).max(), 'AUC', roc_auc_score(y0[val_idx[elig]], pv_c[elig]), f'{time.time()-t0:.0f}s', flush=True)
def lf(yy):
    F_fit, (F_val, F_te) = label_features(XA, yy, fit_idx, [val_idx, test_idx], ns['tr'], ns['te'], seed=K, spec=BEST_SPEC)
    return (F_fit.to_numpy(np.float32), F_val.to_numpy(np.float32), F_te.to_numpy(np.float32))
L0 = lf(y0)
print('recomputed LF == cached LF:', [bool(np.array_equal(a, b, equal_nan=True)) for a, b in zip(L0, ns['C']['LF'][('cv', K)])], f'{time.time()-t0:.0f}s', flush=True)
def run(yy):
    ns['y'] = yy
    ns['C']['LF'][('cv', K)] = lf(yy)
    return ns['run_fold'](K)
pv0, pt0 = run(y0)
print('baseline (recomputed LF) == cached-LF run:', bool(np.array_equal(pv0, pv_c)), flush=True)
# 2. flip ALL validation labels
y1 = y0.copy(); y1[val_idx] = 1 - y1[val_idx]
pv1, pt1 = run(y1)
print('FLIP ALL VAL LABELS -> val preds identical:', bool(np.array_equal(pv1, pv0)), 'maxabs', np.abs(pv1 - pv0).max(), ' test preds identical:', bool(np.array_equal(pt1, pt0)), flush=True)
# 3. random permutation of val labels
rng = np.random.default_rng(7); y2 = y0.copy(); y2[val_idx] = rng.permutation(y0[val_idx])
pv2, pt2 = run(y2)
print('PERMUTE VAL LABELS -> val preds identical:', bool(np.array_equal(pv2, pv0)), 'maxabs', np.abs(pv2 - pv0).max(), flush=True)
# 4. positive control: permute FIT labels (teacher pseudo-labels remain from real-label teachers)
y3 = y0.copy(); y3[fit_idx] = rng.permutation(y0[fit_idx])
pv3, pt3 = run(y3)
print('POSITIVE CONTROL permute FIT labels -> maxabs change', np.abs(pv3 - pv0).max(), 'val AUC', roc_auc_score(y0[val_idx[elig]], pv3[elig]), '(teacher OOF still real-label based)', flush=True)
# 5. permute FIT labels AND remove teacher signal (pseudo labels replaced by constant fit-mean) -> AUC should be ~0.5
TE = ns['TEACH']
for nm_ in TE: TE[nm_] = dict(TE[nm_]); TE[nm_]['oof'] = np.where(np.isnan(TE[nm_]['oof']), np.nan, 0.3)
pv4, _ = run(y3)
print('permute FIT labels + constant teacher -> val AUC', roc_auc_score(y0[val_idx[elig]], pv4[elig]), f'{time.time()-t0:.0f}s', flush=True)
