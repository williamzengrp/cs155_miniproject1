"""Audit of r3_chp_labalbs2c, static part: re-assemble the saved block from the builder's caches, NaN pattern,
paired AUC vs chrep_bl, test_full reproduction, Spearman, correction-tuning contribution, inner-OOF sanity."""
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK); sys.path.insert(0, WORK + '/r3_chp')
import harness as H
import chp_lib as L
from chp_save import BLOCKS, CC

NAME = 'labalbs2c'
sv = H.load_res('r3_chp_' + NAME)
ref = H.load_res('chrep_bl')
y = L.y
print('saved keys/shapes', {k: v.shape for k, v in sv.items()})
cat, et, w, cc, ec, fc, fe = BLOCKS[NAME]
print('block def:', cat, et, w, cc, ec, fc, fe)
oof, aucs, tfa, have = L.assemble(lambda k: L.blend(cat, et, k, w, cc, ec))
print('repeats', have)
d = np.nanmax(np.abs(oof - sv['oof']))
print('reassembled oof vs saved: max|diff|', d, ' nan-pattern equal', np.array_equal(np.isnan(oof), np.isnan(sv['oof'])))
print('test_foldavg vs saved max|diff|', np.abs(tfa - sv['test_foldavg']).max())
# NaN pattern vs chrep_bl and vs eligibility
print('NaN pattern == chrep_bl:', np.array_equal(np.isnan(sv['oof']), np.isnan(ref['oof'])))
mk = np.zeros((5, len(y)), bool)
for k, (r, fit, val, elig) in enumerate(L.splits):
    mk[r, val[elig]] = True
print('non-NaN == eligible val rows:', np.array_equal(~np.isnan(sv['oof']), mk))
a_b = np.array([roc_auc_score(y[~np.isnan(sv['oof'][r])], sv['oof'][r][~np.isnan(sv['oof'][r])]) for r in range(5)])
a_r = np.array([roc_auc_score(y[~np.isnan(ref['oof'][r])], ref['oof'][r][~np.isnan(ref['oof'][r])]) for r in range(5)])
print('AUC per rep block', np.round(a_b, 5), 'mean', a_b.mean().round(5))
print('AUC per rep chrep', np.round(a_r, 5), 'mean', a_r.mean().round(5), ' diff', np.round(a_b - a_r, 5),
      'mean', (a_b - a_r).mean().round(5), 'pos', (a_b > a_r).sum())
print('saved aucs', np.round(sv['aucs'], 5))
# contribution of the tuned correction CC (selection bias source): same sources, Charlie's correction / none
for lab, cc_ in (('Charlie corr (CORR0)', None), ('no cat corr', (0.0, 1, 1)), ('tuned CC', CC)):
    o2, a2, _, h2 = L.assemble(lambda k: L.blend(cat, et, k, w, cc_, ec))
    print(f'  {lab:22s} AUC {np.nanmean(a2):.5f} diff vs chrep {np.nanmean(a2 - a_r):+.5f}')
# replica with tuned CC (how much CC alone gives on the replica)
o3, a3, _, _ = L.assemble(lambda k: L.blend(['rep'], ['rep'], k, w, CC, None))
print(f'  replica + tuned CC    AUC {np.nanmean(a3):.5f} diff vs chrep {np.nanmean(a3 - a_r):+.5f}')
# inner-OOF sanity: residual-correction inputs must be out-of-fold over the fit rows (AUC ~ CV level, not in-sample)
ipa = [roc_auc_score(y[L.splits[k][1]], L.C[k]['cat_ip']) for k in range(25)]
ipe = [roc_auc_score(y[L.splits[k][1]], L.C[k]['et_ip']) for k in range(25)]
print('cat_ip AUC on fit rows min/mean/max', np.round([min(ipa), np.mean(ipa), max(ipa)], 4),
      ' et_ip', np.round([min(ipe), np.mean(ipe), max(ipe)], 4))
print('len(ip)==len(fit) all:', all(len(L.C[k]['cat_ip']) == len(L.splits[k][1]) for k in range(25)))
# test_full reproduction (full refit sources, all-train labels)
import chp_full as F
tf = F.blend_full(fc, fe, w, cc, ec)
print('test_full recomputed vs saved max|diff|', np.abs(tf - sv['test_full']).max(), ' finite', np.isfinite(tf).all(),
      ' len', len(tf))
ch = ref['test_full']
print('spearman test_full vs Charlie file %.5f, test_foldavg vs Charlie %.5f, test_full vs own foldavg %.5f' % (
    spearmanr(sv['test_full'], ch)[0], spearmanr(sv['test_foldavg'], ch)[0], spearmanr(sv['test_full'], sv['test_foldavg'])[0]))
raws = [F.raw(s, 'cat') for s in fc]
print('full-refit cat sources pairwise corr (distinct seeds):',
      np.round([np.corrcoef(raws[i], raws[j])[0, 1] for i in range(5) for j in range(i + 1, 5)], 4))
print('full IP from', len(F.FULL), 'replica full refits; ip AUC on train', [round(roc_auc_score(y, d['cat_ip']), 4) for d in F.FULL])
