"""Static checks of results/r3_chx_c*.npz against the r3_chx cache, the Charlie replica and the harness splits."""
import sys, os, glob
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
WORK = './work'
for p in (WORK + '/r3_chx', WORK, WORK + '/ch', WORK + '/main', WORK + '/te'):
    sys.path.insert(0, p)
import harness as H
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
tr, te, y = H.load(); n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)
C = WORK + '/r3_chx/cache'
RC = WORK + '/ch/cache'


def md(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


B = H.load_res('r3_chx_c'); BL = H.load_res('chrep_bl')
print('keys', sorted(B), 'oof', B['oof'].shape, 'test_full', B['test_full'].shape, 'foldavg', B['test_foldavg'].shape)
print('aucs', np.round(B['aucs'], 5), 'mean', np.nanmean(B['aucs']).round(5))
print('chrep_bl aucs', np.round(BL['aucs'], 5), 'mean', np.nanmean(BL['aucs']).round(5),
      'diff', np.round(B['aucs'] - BL['aucs'], 5), 'mean diff', np.nanmean(B['aucs'] - BL['aucs']).round(5))
for nm in ('r3_chx_c_cat', 'r3_chx_c_et'):
    X = H.load_res(nm); ref = H.load_res({'r3_chx_c_cat': 'chrep_cat', 'r3_chx_c_et': 'chrep_et'}[nm])
    print(f'{nm}: aucs mean {np.nanmean(X["aucs"]):.5f} vs {np.nanmean(ref["aucs"]):.5f}; oof maxdiff vs replica '
          f'{np.nanmax(np.abs(X["oof"] - ref["oof"])):.3g}; NaN pattern equal {bool((np.isnan(X["oof"]) == np.isnan(ref["oof"])).all())}')
# NaN pattern vs chrep_bl
print('NaN pattern equal to chrep_bl (all reps):', bool((np.isnan(B['oof']) == np.isnan(BL['oof'])).all()),
      ' non-NaN per rep', (~np.isnan(B['oof'])).sum(1))
# NaN pattern vs harness eligibility
exp = np.zeros((5, n), bool)
for k, (r, fit, val, elig) in enumerate(splits):
    exp[r, val[elig]] = True
print('non-NaN == harness eligible rows:', bool(((~np.isnan(B['oof'])) == exp).all()))
# recompute oof + aucs from cache
oof = np.full((5, n), np.nan); tfa = np.zeros(m)
for k, (r, fit, val, elig) in enumerate(splits):
    d = np.load(f'{C}/c_k{k}.npz')
    assert len(d['cat_v']) == len(val) and len(d['cat_t']) == m
    oof[r, val[elig]] = (0.6 * d['cat_v'] + 0.4 * d['et_v'])[elig]
    tfa += (0.6 * d['cat_t'] + 0.4 * d['et_t']) / 25
print('oof recomputed from cache maxdiff', np.nanmax(np.abs(oof - B['oof'])), ' test_foldavg maxdiff', md(tfa, B['test_foldavg']))
aucs = [roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) for r in range(5)]
print('recomputed aucs', np.round(aucs, 5))
# ET identical to replica per fold
dv = []; dt = []; dip = []; cs = []
for k in range(25):
    a, b = np.load(f'{C}/c_k{k}.npz'), np.load(f'{RC}/chfull_std_k{k}.npz')
    dv.append(md(a['et_v'], b['et_v'])); dt.append(md(a['et_t'], b['et_t'])); dip.append(md(a['et_ip'], b['et_ip']))
    cs.append(spearmanr(a['cat_v'], b['cat_v'])[0])
print(f'ET vs replica per-fold maxdiff: val {max(dv):.3g} test {max(dt):.3g} ip {max(dip):.3g}; '
      f'cat_v spearman vs replica cat_v mean {np.mean(cs):.4f} min {np.min(cs):.4f}')
# full refit
fulls = sorted(glob.glob(f'{C}/c_full_s*.npz'))
Fl = [np.load(f) for f in fulls]
full = 0.6 * np.mean([f['cat_t'] for f in Fl], 0) + 0.4 * np.mean([f['et_t'] for f in Fl], 0)
print('full-refit files', [os.path.basename(f) for f in fulls], 'keys', sorted(Fl[0].files))
print('test_full == 0.6 mean(cat_t) + 0.4 mean(et_t) over full seeds: maxdiff', md(full, B['test_full']))
print('ip length of full refit (should be n=%d):' % n, [len(f['cat_ip']) for f in Fl])
print('seed-to-seed spearman of full cat_t:', np.round([spearmanr(Fl[0]['cat_t'], f['cat_t'])[0] for f in Fl[1:]], 4))
print('test_full vs test_foldavg: corr %.4f spearman %.4f maxdiff %.3g' % (
    np.corrcoef(B['test_full'], B['test_foldavg'])[0, 1], spearmanr(B['test_full'], B['test_foldavg'])[0],
    md(B['test_full'], B['test_foldavg'])))
chf = BL['test_full']
print('Spearman(test_full, Charlie file) %.4f ; Spearman(chrep_et test_full, r3_chx_c_et test_full) %.4f' % (
    spearmanr(B['test_full'], chf)[0], spearmanr(H.load_res('chrep_et')['test_full'], H.load_res('r3_chx_c_et')['test_full'])[0]))
# album-sharing: CV eligible val rows vs test rows
from label_features import get_ctx
alb = np.asarray(get_ctx(tr, te).key['alb'])
uri_all = pd.factorize(pd.concat([tr['uri'], te['uri']], ignore_index=True))[0]
atr, ate = alb[:n], alb[n:]
print('test uris in train:', int(np.isin(uri_all[n:], uri_all[:n]).sum()), '/', m)
sh_te = np.isin(ate, atr)
fr = []
for k, (r, fit, val, elig) in enumerate(splits):
    v = val[elig]
    fr.append(np.isin(atr[v], atr[fit]).mean())
print(f'album shared with labelled rows: test rows {sh_te.mean():.3f}; CV eligible val rows mean {np.mean(fr):.3f} '
      f'(min {np.min(fr):.3f}, max {np.max(fr):.3f})')
# also: share of rows whose release date appears among labelled rows (date-TE analogue)
dtr = tr['track_album_release_date'].astype(str).to_numpy(); dte = te['track_album_release_date'].astype(str).to_numpy()
frd = [np.isin(dtr[val[elig]], dtr[fit]).mean() for k, (r, fit, val, elig) in enumerate(splits)]
print(f'release date shared with labelled rows: test {np.isin(dte, dtr).mean():.3f}; CV eligible val {np.mean(frd):.3f}')
# variant gain by album-shared vs not (pooled over folds, AUC diff c - replica)
gs, gn = [], []
for k, (r, fit, val, elig) in enumerate(splits):
    a, b = np.load(f'{C}/c_k{k}.npz'), np.load(f'{RC}/chfull_std_k{k}.npz')
    v = val[elig]; sh = np.isin(atr[v], atr[fit])
    pa = (0.6 * a['cat_v'] + 0.4 * a['et_v'])[elig]; pb = (0.6 * b['cat_v'] + 0.4 * b['et_v'])[elig]
    for msk, L in ((sh, gs), (~sh, gn)):
        yy = y[v][msk]
        if 0 < yy.sum() < len(yy):
            L.append(roc_auc_score(yy, pa[msk]) - roc_auc_score(yy, pb[msk]))
print(f'per-fold AUC gain (c - replica): album-shared rows {np.mean(gs):+.5f}; non-shared rows {np.mean(gn):+.5f}')
