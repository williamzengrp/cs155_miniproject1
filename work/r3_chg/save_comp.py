"""Assemble cached r3_chg folds (full mode = after residual correction) into results files.
python -I save_comp.py TAG PREFIX [W ...]
  -> PREFIX (blend with Charlie's ET weight 0.4), PREFIX_cat, PREFIX_et, PREFIX_w<W*10> (cat (1-W) + et W)
Repeats whose 5 folds are not all done are left all-NaN (pubsim skips them). test_full = mean over the cached
full-refit seeds; test_foldavg = mean over the done folds."""
import os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
CACHE = WORK + '/r3_chg/cache'
tag, prefix = sys.argv[1], sys.argv[2]
ws = [float(w) for w in sys.argv[3:]]
tr, te, y = H.load()
n = len(tr)
splits = H.make_splits(tr, y, n_rep=5)
ref = H.load_res('chrep_bl')
fs = sorted(f for f in os.listdir(CACHE) if f.startswith(f'{tag}_full_s'))
Fl = [dict(np.load(f'{CACHE}/{f}')) for f in fs]
outs = [(prefix, 0.4)] + [(f'{prefix}_cat', 0.0), (f'{prefix}_et', 1.0)] + [(f'{prefix}_w{int(round(w * 10))}', w) for w in ws]
for name, w in outs:
    oof = np.full((5, n), np.nan)
    aucs = np.full(5, np.nan)
    tfa, nt = 0, 0
    for r in range(5):
        ks = range(5 * r, 5 * r + 5)
        paths = [f'{CACHE}/{tag}_k{k}.npz' for k in ks]
        if not all(os.path.exists(p) for p in paths):
            continue
        for k, p in zip(ks, paths):
            c = np.load(p)
            _, fit, val, elig = splits[k]
            oof[r, val[elig]] = ((1 - w) * c['cat_v'] + w * c['et_v'])[elig]
            tfa = tfa + (1 - w) * c['cat_t'] + w * c['et_t']
            nt += 1
        m = ~np.isnan(oof[r])
        assert (m == ~np.isnan(ref['oof'][r])).all()
        aucs[r] = roc_auc_score(y[m], oof[r, m])
    done = ~np.isnan(aucs)
    full = np.mean([(1 - w) * f['cat_t'] + w * f['et_t'] for f in Fl], axis=0) if Fl else None
    res = dict(oof=oof, aucs=aucs, auc=float(np.nanmean(aucs)), test_foldavg=tfa / max(nt, 1))
    H.save(name, res, full)
    d = aucs[done] - ref['aucs'][done]
    sp = spearmanr(full, ref['test_full'])[0] if full is not None else float('nan')
    sp2 = spearmanr(full, res['test_foldavg'])[0] if full is not None else float('nan')
    print(f'saved {name:22s} reps {np.flatnonzero(done).tolist()} AUC {np.nanmean(aucs):.5f} per-rep {np.round(aucs, 4)} '
          f'vs chrep_bl {d.mean():+.5f} ({np.round(d, 4)})  spearman(test_full, chrep_bl)={sp:.4f} '
          f'spearman(full, foldavg)={sp2:.4f}  full seeds {len(Fl)}', flush=True)
