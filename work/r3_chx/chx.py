"""r3_chx driver (Charlie's exact pipeline + extra inputs; see chx_lib.py for the variants a/b/c).

  python -I chx.py cv VARIANT K0 K1 THREADS     std folds K0..K1-1 (H.make_splits(n_rep=5), S = SEED0 + k exactly as
                                                work/ch/chfull.py) -> cache/<VARIANT>_k<k>.npz (resumable)
  python -I chx.py full VARIANT SEEDS THREADS   full refit on all train rows (S = SEED0 + 1000*(s+1), label-feature
                                                seed s as final_cv.py) -> cache/<VARIANT>_full_s<s>.npz
  python -I chx.py eval VARIANT [VARIANT ...]   paired comparison with the cached replica folds (work/ch/cache)
  python -I chx.py save VARIANT                 assemble cached folds (complete repeats only; missing repeats NaN)
                                                + full-refit seeds -> results/r3_chx_<VARIANT>.npz (0.6 cat + 0.4 et)
  VARIANT '0' = no extras (pure replica; sanity check of the copy).
"""
import sys
sys.dont_write_bytecode = True
import os, time, warnings
import numpy as np
import pandas as pd
WORK = './work'
HERE = WORK + '/r3_chx'
sys.path.insert(0, HERE)
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
sys.path.insert(0, WORK + '/main')
sys.path.insert(0, WORK + '/te')
import harness as H
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
CACHE = HERE + '/cache'
RCACHE = WORK + '/ch/cache'
mode = sys.argv[1]
tr, te, y = H.load()
n, m = len(tr), len(te)
splits = H.make_splits(tr, y, n_rep=5)


def build(variant, threads):
    from ch_common import connected_groups
    from ch_feats import charlie_pool
    from chx_lib import ReplicaX, CD_COLS
    F, BL = charlie_pool(tr, te)
    XA = CD = alb = None
    if 'a' in variant or 'b' in variant or 'e' in variant:
        from features_final import build_features_final
        X, Xt = build_features_final(tr, te)
        XA = pd.concat([X, Xt], ignore_index=True)
        CD = XA[CD_COLS].to_numpy(float)
    if 'c' in variant or 'e' in variant:
        from label_features import get_ctx
        alb = get_ctx(tr, te).key['alb']
    return ReplicaX(tr, te, y, F[BL['PROJECT']], connected_groups(tr), variant.replace('0', ''), threads=threads,
                    XA=XA, CD=CD, alb=alb)


if mode == 'cv':
    from chx_lib import SEED0
    variant, k0, k1, threads = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    R = build(variant, threads)
    for k in range(k0, min(k1, len(splits))):
        path = f'{CACHE}/{variant}_k{k}.npz'
        if os.path.exists(path):
            continue
        r, fit, val, elig = splits[k]
        S = SEED0 + k
        t0 = time.time()
        Xf, (Xv,), Xt = R.matrices(fit, [val], True, lf_seed=k)
        inner = R.inner_sets(fit, S)
        out, tim = {}, {}
        for fam, tag in (('cat', 'cat'), ('extra', 'et')):
            ta = time.time()
            (rv,), rt, ip = R.run_family(fam, fit, Xf, [Xv], [val], Xt, inner, S)
            out[f'{tag}_raw_v'], out[f'{tag}_ip'] = rv, ip
            out[f'{tag}_v'] = R.correct(fam, fit, ip, rv, tr.iloc[val])
            out[f'{tag}_raw_t'] = rt
            out[f'{tag}_t'] = R.correct(fam, fit, ip, rt, te)
            tim[tag] = time.time() - ta
        out['ncol'] = Xf.shape[1]
        np.savez(path, **out)
        print(f'[{variant} k={k}] ncol {Xf.shape[1]}  cat {tim["cat"]:.1f}s  et {tim["et"]:.1f}s  '
              f'total {time.time() - t0:.1f}s  {time.strftime("%H:%M:%S")}', flush=True)

elif mode == 'cvconf':
    # confirmation split set (H.make_splits(n_rep=3, seed0=9000), as work/ch/chfull.py 'conf'; S = SEED0 + k; no test)
    from chx_lib import SEED0
    variant, k0, k1, threads = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    R = build(variant, threads)
    csplits = H.make_splits(tr, y, n_rep=3, seed0=9000)
    for k in range(k0, min(k1, len(csplits))):
        path = f'{CACHE}/conf_{variant}_k{k}.npz'
        if os.path.exists(path):
            continue
        r, fit, val, elig = csplits[k]
        S = SEED0 + k
        t0 = time.time()
        Xf, (Xv,), _ = R.matrices(fit, [val], False, lf_seed=k)
        inner = R.inner_sets(fit, S)
        out = {}
        fams = (('extra', 'et'),) if os.environ.get('ETONLY', '0') == '1' else (('cat', 'cat'), ('extra', 'et'))
        if len(fams) == 1:
            path = f'{CACHE}/conf_et_{variant}_k{k}.npz'
        for fam, tag in fams:
            (rv,), _, ip = R.run_family(fam, fit, Xf, [Xv], [val], None, inner, S)
            out[f'{tag}_raw_v'], out[f'{tag}_ip'] = rv, ip
            out[f'{tag}_v'] = R.correct(fam, fit, ip, rv, tr.iloc[val])
        np.savez(path, **out)
        print(f'[conf {variant} k={k}] {time.time() - t0:.1f}s  {time.strftime("%H:%M:%S")}', flush=True)

elif mode in ('cvet', 'fullet'):
    # ExtraTrees family ONLY (identical code path / seeds as cv/full; CatBoost not run) -> cache/et_<VARIANT>_...
    from chx_lib import SEED0
    variant, threads = sys.argv[2], int(sys.argv[-1])
    R = build(variant, threads)
    if mode == 'cvet':
        jobs = [(f'{CACHE}/et_{variant}_k{k}.npz', k, splits[k][1], splits[k][2], SEED0 + k, k)
                for k in range(int(sys.argv[3]), min(int(sys.argv[4]), len(splits)))]
    else:
        jobs = [(f'{CACHE}/et_{variant}_full_s{s}.npz', None, np.arange(n), None, SEED0 + 1000 * (s + 1), s)
                for s in [int(x) for x in sys.argv[3].split(',')]]
    for path, k, fit, val, S, lfs in jobs:
        if os.path.exists(path):
            continue
        t0 = time.time()
        Xf, Xa, Xt = R.matrices(fit, [] if val is None else [val], True, lf_seed=lfs)
        inner = R.inner_sets(fit, S)
        ra, rt, ip = R.run_family('extra', fit, Xf, Xa, [] if val is None else [val], Xt, inner, S)
        out = dict(et_raw_t=rt, et_ip=ip, et_t=R.correct('extra', fit, ip, rt, te), ncol=Xf.shape[1])
        if val is not None:
            out['et_raw_v'] = ra[0]
            out['et_v'] = R.correct('extra', fit, ip, ra[0], tr.iloc[val])
        np.savez(path, **out)
        print(f'[ET-only {variant} {"k=" + str(k) if k is not None else "full s" + str(lfs)}] ncol {Xf.shape[1]} '
              f'{time.time() - t0:.1f}s  {time.strftime("%H:%M:%S")}', flush=True)

elif mode == 'full':
    from chx_lib import SEED0
    variant, seeds, threads = sys.argv[2], [int(s) for s in sys.argv[3].split(',')], int(sys.argv[4])
    R = build(variant, threads)
    fit = np.arange(n)
    for s in seeds:
        path = f'{CACHE}/{variant}_full_s{s}.npz'
        if os.path.exists(path):
            continue
        S = SEED0 + 1000 * (s + 1)
        t0 = time.time()
        Xf, _, Xt = R.matrices(fit, [], True, lf_seed=s)
        inner = R.inner_sets(fit, S)
        out = {}
        for fam, tag in (('cat', 'cat'), ('extra', 'et')):
            _, rt, ip = R.run_family(fam, fit, Xf, [], [], Xt, inner, S)
            out[f'{tag}_raw_t'], out[f'{tag}_ip'] = rt, ip
            out[f'{tag}_t'] = R.correct(fam, fit, ip, rt, te)
        np.savez(path, **out)
        print(f'[{variant} full seed {s}] {time.time() - t0:.1f}s  {time.strftime("%H:%M:%S")}', flush=True)

elif mode in ('eval', 'save'):
    from chx_lib import EXTRA_WEIGHT
    W = EXTRA_WEIGHT

    def load_folds(prefix):
        C = {}
        for k in range(len(splits)):
            p = f'{prefix}_k{k}.npz'
            if os.path.exists(p):
                d = dict(np.load(p))
                d['bl_v'] = (1 - W) * d['cat_v'] + W * d['et_v']
                d['bl_t'] = (1 - W) * d['cat_t'] + W * d['et_t']
                d['blraw_v'] = (1 - W) * d['cat_raw_v'] + W * d['et_raw_v']
                C[k] = d
        return C

    def rep_aucs(C, key, reps):
        out = []
        for r in reps:
            oof = np.full(n, np.nan)
            for k in range(5 * r, 5 * r + 5):
                _, fit, val, elig = splits[k]
                oof[val[elig]] = C[k][key][elig]
            mm = ~np.isnan(oof)
            out.append(roc_auc_score(y[mm], oof[mm]))
        return np.array(out)

    REP = load_folds(f'{RCACHE}/chfull_std')
    if mode == 'eval':
        for variant in sys.argv[2:]:
            C = load_folds(f'{CACHE}/{variant}')
            ks = sorted(C)
            reps = [r for r in range(5) if all(k in C for k in range(5 * r, 5 * r + 5))]
            print(f'== variant {variant}: folds {ks}  complete repeats {reps}')
            for key in ('bl_v', 'cat_v', 'et_v', 'blraw_v', 'cat_raw_v', 'et_raw_v'):
                fd = []
                for k in ks:
                    _, fit, val, elig = splits[k]
                    a = roc_auc_score(y[val[elig]], C[k][key][elig])
                    b = roc_auc_score(y[val[elig]], REP[k][key][elig])
                    fd.append(a - b)
                fd = np.array(fd)
                line = f'  {key:10s} per-fold diff mean {fd.mean():+.5f} (se {fd.std(ddof=1) / np.sqrt(len(fd)) if len(fd) > 1 else 0:.5f}, {np.sum(fd > 0)}/{len(fd)} >0)'
                if reps:
                    a, b = rep_aucs(C, key, reps), rep_aucs(REP, key, reps)
                    line += f' | repeats {reps}: {a.mean():.5f} vs replica {b.mean():.5f} diff {np.round(a - b, 5)}'
                print(line, flush=True)
            # rank correlation with replica on validation and test
            from scipy.stats import spearmanr
            cv_ = np.mean([spearmanr(C[k]['bl_v'], REP[k]['bl_v'])[0] for k in ks])
            ct_ = np.mean([spearmanr(C[k]['bl_t'], REP[k]['bl_t'])[0] for k in ks])
            print(f'  spearman vs replica fold preds: val {cv_:.4f}  test {ct_:.4f}', flush=True)
    else:
        variant = sys.argv[2]
        screen = os.environ.get('SCREEN', '0') == '1'    # screening save: no full refit, name r3_chx_scr_<variant>
        C = load_folds(f'{CACHE}/{variant}')
        reps = [r for r in range(5) if all(k in C for k in range(5 * r, 5 * r + 5))]
        fulls = sorted(f for f in os.listdir(CACHE) if f.startswith(f'{variant}_full_s'))
        assert reps and (fulls or screen), (reps, fulls)
        if fulls:
            Fl = [dict(np.load(f'{CACHE}/{f}')) for f in fulls]
            full = {t: np.mean([f[f'{t}_t'] for f in Fl], axis=0) for t in ('cat', 'et')}
        else:   # screening only: fold-averaged test predictions stand in (NOT a full refit)
            full = {t: np.mean([C[k][f'{t}_t'] for r in reps for k in range(5 * r, 5 * r + 5)], axis=0)
                    for t in ('cat', 'et')}
        full['bl'] = (1 - W) * full['cat'] + W * full['et']
        chf = H.load_res('chrep_bl')['test_full']
        from scipy.stats import spearmanr
        for tag, nm in ((('bl', ''),) if screen else (('bl', ''), ('cat', '_cat'), ('et', '_et'))):
            oof = np.full((5, n), np.nan)
            tst = np.zeros(m)
            for r in reps:
                for k in range(5 * r, 5 * r + 5):
                    _, fit, val, elig = splits[k]
                    oof[r, val[elig]] = C[k][f'{tag}_v'][elig]
                    tst += C[k][f'{tag}_t'] / (5 * len(reps))
            aucs = np.array([roc_auc_score(y[~np.isnan(oof[r])], oof[r][~np.isnan(oof[r])]) if r in reps else np.nan
                             for r in range(5)])
            res = dict(oof=oof, aucs=aucs, auc=float(np.nanmean(aucs)), test_foldavg=tst)
            name = f'r3_chx_scr_{variant}' if screen else f'r3_chx_{variant}{nm}'
            H.save(name, res, None if screen else full[tag])
            ref = H.load_res({'bl': 'chrep_bl', 'cat': 'chrep_cat', 'et': 'chrep_et'}[tag])['aucs'][reps]
            print(f'saved {name}: repeats {reps} AUC {np.nanmean(aucs):.5f} vs replica {ref.mean():.5f} '
                  f'(diff per rep {np.round(aucs[reps] - ref, 5)}); full seeds {fulls}; '
                  f'spearman(test_full, charlie file) {spearmanr(full[tag], chf)[0]:.4f}; '
                  f'corr(foldavg, full) {np.corrcoef(tst, full[tag])[0, 1]:.4f}', flush=True)
