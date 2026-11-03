"""r3_aug: augmentation variants of Charlie's own pipeline (faithful replica work/ch/chfull_lib.py, imported read-only).

  python -I chaug.py cv   FAM VARIANT K0 K1     std folds K0..K1-1 -> cache/chaug_<FAM>_<vtag>_k<k>.npz (resumable)
  python -I chaug.py full FAM VARIANT SEEDS     full refit on all train rows, SEEDS e.g. 0,1,2 -> cache/chaug_<FAM>_<vtag>_full_s<s>.npz
  FAM      cat | et   (Charlie's CatBoost / ExtraTrees configs, CFG in chfull_lib)
  VARIANT  ';'-joined options
    pl=TEACHER:MODE:W   pseudo-labelled rows added to the fit rows
        TEACHER ch   = Charlie replica blend (results/chrep_bl): CV -> its fold-k OOF prediction on the fold's eligible
                       val rows (fold-k model trained on fit rows only); final -> Charlie's file on the test rows
                chl  = 0.7*ch + 0.3*eda_chl_lgbm (label-feature LightGBM on charlie view; same fold-safe construction)
        MODE    soft (row duplicated as (x,1,W*p) + (x,0,W*(1-p))) | sharpT (p sharpened with temperature T first)
        W       weight of a pseudo row relative to a (mean) real row
    noise=S:K:W K jittered copies of every fit row (gaussian S*sd on continuous columns, sd from the fit rows), total
                copy weight W per original row (times the row's own weight)
    mix=K:W:A   within-class SMOTE/mixup among fit rows (fit labels only): partner among the 10 nearest same-class
                fit rows with another uri, lambda~U(0,A) on continuous columns; K synthetic rows per fit row, weight W
    tta=S:K     test-time augmentation: mean prediction over the clean copy + K gaussian-jittered copies (S*sd on
                continuous columns)
  Residual correction (part of Charlie's model): applied with Charlie's params; the inner-OOF predictions of the fit
  rows (fit-fold labels only) are reused from the replica cache work/ch/cache/chfull_std_k<k>.npz in CV and recomputed
  (Charlie's un-augmented inner models, 3-fold grouped inner split over all train rows) for the full refit.
Leakage: pseudo-labels for CV validation rows come only from models trained on that fold's fit rows (stored OOF);
the student never sees validation labels; test_full uses only all-train models / Charlie's file as teacher.
"""
import os, sys, time, warnings
import numpy as np
WORK = './work'
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + '/ch')
import harness as H
from ch_common import connected_groups
from ch_feats import charlie_pool
from chfull_lib import Replica, SEED0, CFG, make_model, dup_weights
warnings.filterwarnings('ignore')
THREADS = int(os.environ.get('NJ', '3'))
HERE = WORK + '/r3_aug/cache/'
CHC = WORK + '/ch/cache/'

mode, FAM, VARIANT = sys.argv[1], sys.argv[2], sys.argv[3]
opts = {}
for tok in [t for t in VARIANT.split(';') if t]:
    k_, v_ = tok.split('=')
    opts[k_] = v_.split(':')
vtag = VARIANT.replace(';', '_').replace(':', '-').replace('=', '-') or 'base'
tr, te, y = H.load()
n, m = len(tr), len(te)
F, BL = charlie_pool(tr, te)
R = Replica(tr, te, y, F[BL['PROJECT']], connected_groups(tr), threads=THREADS)
splits = H.make_splits(tr, y, n_rep=5)
fam_tag = 'cat' if FAM == 'cat' else 'et'
fam_cfg = 'cat' if FAM == 'cat' else 'extra'

TEACH = {}
if 'pl' in opts:
    TEACH['ch'] = H.load_res('chrep_bl')
    if opts['pl'][0] == 'chl':
        TEACH['lgbm'] = H.load_res('eda_chl_lgbm')


def teacher(r=None, val_idx=None, final=False):
    kind = opts['pl'][0]
    if final:
        p = TEACH['ch']['test_full']
        if kind == 'chl':
            p = 0.7 * p + 0.3 * TEACH['lgbm']['test_full']
    else:
        p = TEACH['ch']['oof'][r, val_idx]
        if kind == 'chl':
            p = 0.7 * p + 0.3 * TEACH['lgbm']['oof'][r, val_idx]
    return np.clip(p, 1e-4, 1 - 1e-4)


def mixup(A, yA, u, K, A_max, rng, cont, sd):
    from sklearn.neighbors import NearestNeighbors
    Zc = A[:, cont] / sd[cont]
    out = []
    for c in (0, 1):
        ix = np.flatnonzero(yA == c)
        _, ind = NearestNeighbors(n_neighbors=min(16, len(ix))).fit(Zc[ix]).kneighbors(Zc[ix])
        for _k in range(K):
            part = np.empty(len(ix), int)
            for a in range(len(ix)):
                cand = [ix[j] for j in ind[a, 1:] if u[ix[j]] != u[ix[a]]][:10]
                part[a] = rng.choice(cand) if cand else ix[a]
            lam = rng.uniform(0, A_max, len(ix))
            Sx = A[ix].copy()
            Sx[:, cont] = Sx[:, cont] + lam[:, None] * (A[part][:, cont] - Sx[:, cont])
            out.append((Sx, np.full(len(ix), c, int)))
    return np.vstack([o[0] for o in out]), np.concatenate([o[1] for o in out])


def student(Xf, fit, U, pu, S):
    w = dup_weights(R.uri[fit], CFG[fam_cfg]['duplicate_power'])
    A, yA, wA = Xf, y[fit].astype(int), w
    if 'noise' in opts or 'mix' in opts:
        cont, sd = cont_cols(Xf)
        rng = np.random.default_rng(S + 31)
        parts = [(A, yA, wA)]
        if 'noise' in opts:
            s_, K, W = float(opts['noise'][0]), int(opts['noise'][1]), float(opts['noise'][2])
            for _ in range(K):
                Z = Xf.copy()
                Z[:, cont] += rng.standard_normal((len(Xf), cont.sum())) * (s_ * sd[cont])[None, :]
                parts.append((Z, yA, w * W / K))
        if 'mix' in opts:
            K, W, Am = int(opts['mix'][0]), float(opts['mix'][1]), float(opts['mix'][2])
            Sx, yS = mixup(Xf, yA, R.uri[fit], K, Am, rng, cont, sd)
            parts.append((Sx, yS, np.full(len(yS), W)))
        A = np.vstack([p[0] for p in parts]); yA = np.concatenate([p[1] for p in parts])
        wA = np.concatenate([p[2] for p in parts])
    if U is not None:
        md, W = opts['pl'][1], float(opts['pl'][2])
        if md.startswith('sharp'):
            T = float(md[5:])
            a, b = pu ** (1 / T), (1 - pu) ** (1 / T)
            pu = a / (a + b)
        A = np.vstack([A, U, U])
        yA = np.concatenate([yA, np.ones(len(U), int), np.zeros(len(U), int)])
        wA = np.concatenate([wA, W * pu, W * (1 - pu)])
    mdl = make_model(fam_cfg, S, THREADS).fit(A, yA, sample_weight=wA)
    return lambda B: mdl.predict_proba(B)[:, 1]


def cont_cols(Xf):
    c = np.array([np.max(np.abs(Xf[:, j] - np.round(Xf[:, j]))) > 1e-6 and len(np.unique(Xf[:, j])) > 20
                  for j in range(Xf.shape[1])])
    sd = Xf.std(0)
    sd[sd == 0] = 1
    return c, sd


def predict(pred, B, Xf, S):
    if 'tta' not in opts:
        return pred(B)
    s, K = float(opts['tta'][0]), int(opts['tta'][1])
    c, sd = cont_cols(Xf)
    rng = np.random.default_rng(S + 99)
    out = [pred(B)]
    for _ in range(K):
        Z = B.copy()
        Z[:, c] += rng.standard_normal((len(B), c.sum())) * (s * sd[c])[None, :]
        out.append(pred(Z))
    return np.mean(out, 0)


if mode == 'cv':
    k0, k1 = int(sys.argv[4]), int(sys.argv[5])
    for k in range(k0, k1):
        path = f'{HERE}chaug_{FAM}_{vtag}_k{k}.npz'
        if os.path.exists(path):
            continue
        t0 = time.time()
        r, fit, val, elig = splits[k]
        S = SEED0 + k
        Xf, (Xv,), Xt = R.matrices(fit, [val], True)
        U = pu = None
        if 'pl' in opts:
            U, pu = Xv[elig], teacher(r, val)[elig]
            assert np.isfinite(pu).all()
        pred = student(Xf, fit, U, pu, S)
        rv, rt = predict(pred, Xv, Xf, S), predict(pred, Xt, Xf, S)
        ip = dict(np.load(f'{CHC}chfull_std_k{k}.npz'))[f'{fam_tag}_ip']
        out = dict(raw_v=rv, raw_t=rt, v=R.correct(fam_cfg, fit, ip, rv, tr.iloc[val]), t=R.correct(fam_cfg, fit, ip, rt, te))
        np.savez(path, **out)
        print(f'[{FAM} {vtag} k={k}] {time.time() - t0:.1f}s', flush=True)

elif mode == 'full':
    fit = np.arange(n)
    for s in [int(x) for x in sys.argv[4].split(',')]:
        path = f'{HERE}chaug_{FAM}_{vtag}_full_s{s}.npz'
        if os.path.exists(path):
            continue
        t0 = time.time()
        S = SEED0 + 1000 * (s + 1)
        Xf, _, Xt = R.matrices(fit, [], True)
        ipath = f'{HERE}chaug_{FAM}_ip_full_s{s}.npy'      # Charlie's un-augmented inner-OOF preds (all train rows)
        if os.path.exists(ipath):
            ip = np.load(ipath)
        else:
            ip = np.full(n, np.nan)
            for it, iv, I, J in R.inner_sets(fit, S):     # as Replica.run_family (inner part), seed S+200
                mi = make_model(fam_cfg, S + 200, THREADS).fit(
                    I, y[fit[it]], sample_weight=dup_weights(R.uri[fit[it]], CFG[fam_cfg]['duplicate_power']))
                ip[iv] = mi.predict_proba(J)[:, 1]
            assert np.isfinite(ip).all()
            np.save(ipath, ip)
        U = pu = None
        if 'pl' in opts:
            U, pu = Xt, teacher(final=True)
        pred = student(Xf, fit, U, pu, S)
        rt = predict(pred, Xt, Xf, S)
        np.savez(path, raw_t=rt, t=R.correct(fam_cfg, fit, ip, rt, te))
        print(f'[{FAM} {vtag} full s={s}] {time.time() - t0:.1f}s', flush=True)
