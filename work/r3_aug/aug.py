"""r3_aug: data-augmentation variants for Charlie-like models (Charlie's 115-col label-free view
ch_feats.charlie_pool PROJECT, optionally + the 3 out-of-fold label features te/label_features.py BEST_SPEC).

python -I aug.py MODEL VARIANT [NREP] [SAVE_NAME]
  MODEL   lgbm (teammate params) | et (charlie ET config, 1/count weights) | lgbmnl / etnl (no label features)
  VARIANT ';'-joined options:
    pl=TEACHER:SET:MODE:W     pseudo-label self-training
        TEACHER  self  = same model trained on the fold's fit rows (stage 1)
                 ch    = Charlie replica OOF on the fold's eligible val rows (results/chrep_bl, fold-k model, never
                         saw the val labels); final: Charlie's file (= chrep_bl test_full) on the test rows
                 selfch= 0.5*self + 0.5*ch
        SET      val (eligible val rows; final: test rows) | valtest (eligible val + test rows, self teacher only)
        MODE     soft (soft labels: LightGBM cross-entropy objective / ET duplicated rows weighted p, 1-p)
                 hardQ (only the Q fraction most confident rows at each end, hard 0/1 labels)
                 sharpT (soft labels sharpened with temperature T: p^(1/T) / (p^(1/T)+(1-p)^(1/T)))
        W        weight of each pseudo row relative to a real row
    noise=S:K:W               K jittered copies of every fit row, gaussian noise S*std on continuous columns,
                              total copy weight W per original row
    mix=K:W:A                 within-class SMOTE/mixup: K synthetic rows per fit row, partner drawn from the 10
                              nearest same-class fit rows (other uri), lambda~U(0,A) on continuous cols,
                              discrete cols copied from the nearer endpoint; synthetic weight W each
    tta=S:K                   test-time augmentation: average predictions over K jittered copies (noise S*std)
CV = harness std splits (seed0=1000, n_rep NREP), eligible rows scored. Full refit (3 seeds, label features with
inner seeds 0/1/2) -> test_full. Leakage: label features fit-fold labels only (cached by prep.py exactly as
main/final_cv.py); pseudo-labels for val rows come only from models trained on the fold's fit rows.
"""
import json, os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
WORK = './work'
sys.path.insert(0, WORK)
import harness as H
warnings.filterwarnings('ignore')

MODEL = sys.argv[1]
VARIANT = sys.argv[2] if len(sys.argv) > 2 else ''
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SAVE = sys.argv[4] if len(sys.argv) > 4 else None
NJ = int(os.environ.get('NJ', '3'))
SEEDOFF = int(os.environ.get('SEEDOFF', '0'))
USE_LF = not MODEL.endswith('nl')
BASE = MODEL[:-2] if MODEL.endswith('nl') else MODEL

opts = {}
for tok in [t for t in VARIANT.split(';') if t]:
    k, v = tok.split('=')
    opts[k] = v.split(':')

tr, te, y = H.load()
n, m = len(tr), len(te)
C = pickle.load(open(WORK + '/r3_aug/cache/prep.pkl', 'rb'))
X, Xt = C['X'], C['Xt']
uri = tr['uri'].to_numpy()
P = json.load(open(WORK + '/best_params_v3.json'))
splits = H.make_splits(tr, y, n_rep=5, seed0=1000)[:5 * NREP]

# continuous columns (for noise / mixup): non-integer valued with many distinct values
XA_ = np.vstack([X, Xt])
FA = np.hstack([XA_, np.vstack([C['LF'][('full', 0)][0], C['LF'][('full', 0)][2]])]) if USE_LF else XA_
cont = np.array([np.nanmax(np.abs(FA[:, j] - np.round(FA[:, j]))) > 1e-6 and len(np.unique(FA[~np.isnan(FA[:, j]), j])) > 20
                 for j in range(FA.shape[1])])
sd = np.nanstd(FA, axis=0)
sd[~np.isfinite(sd) | (sd == 0)] = 1.0

TEACH = {}
if 'pl' in opts and opts['pl'][0] in ('ch', 'selfch'):
    TEACH['ch'] = H.load_res('chrep_bl')     # NO foundation models (TabPFN) anywhere


def mats(fit_idx, apply_list, lf):
    """lf = (F_fit, [F_apply...]) cached label features."""
    A = X[fit_idx]
    if USE_LF:
        A = np.hstack([A, lf[0]])
        Bs = [np.hstack([b, f]) for b, f in zip(apply_list, lf[1])]
    else:
        Bs = list(apply_list)
    return A, Bs


def uweights(u):
    u = pd.Series(u)
    w = 1.0 / u.map(u.value_counts()).to_numpy(float)
    return w / w.mean()


def fit_model(A, yA, wA, seed):
    """yA may be soft (in [0,1]). Returns predict(B)->prob."""
    if BASE == 'lgbm':
        p = dict(P['lgbm'])
        params = dict(objective='cross_entropy', learning_rate=p['learning_rate'], num_leaves=p['num_leaves'],
                      max_depth=p['max_depth'], min_data_in_leaf=p['min_child_samples'],
                      bagging_fraction=p['subsample'], bagging_freq=p['subsample_freq'],
                      feature_fraction=p['colsample_bytree'], lambda_l1=p['reg_alpha'], lambda_l2=p['reg_lambda'],
                      seed=seed, num_threads=NJ, verbose=-1)
        ds = lgb.Dataset(A, label=yA, weight=wA)
        b = lgb.train(params, ds, num_boost_round=p['n_estimators'])
        return lambda B: b.predict(B)
    if BASE == 'et':
        med = np.nanmedian(A, axis=0)
        med = np.where(np.isfinite(med), med, 0)
        fill = lambda Z: np.where(np.isnan(Z), med, Z)
        soft = (yA > 0) & (yA < 1)
        if soft.any():   # duplicate soft rows as (x,1,w*p) + (x,0,w*(1-p))
            A2 = np.vstack([A[~soft], A[soft], A[soft]])
            y2 = np.concatenate([yA[~soft], np.ones(soft.sum()), np.zeros(soft.sum())])
            w2 = np.concatenate([wA[~soft], wA[soft] * yA[soft], wA[soft] * (1 - yA[soft])])
        else:
            A2, y2, w2 = A, yA, wA
        mdl = ExtraTreesClassifier(n_estimators=900, max_depth=15, min_samples_leaf=2,
                                   max_features=0.1827077108658931, n_jobs=NJ, random_state=seed)
        mdl.fit(fill(A2), y2.astype(int), sample_weight=w2)
        return lambda B: mdl.predict_proba(fill(B))[:, 1]
    raise ValueError(BASE)


def base_weights(fit_idx):
    if BASE == 'et':
        return uweights(uri[fit_idx])
    return np.ones(len(fit_idx))


def jitter(B, s, rng):
    Z = B.copy()
    noise = rng.standard_normal(Z.shape) * (s * sd[None, :])
    Z[:, cont] = Z[:, cont] + noise[:, cont]
    return Z


def mixup(A, yA, u, K, A_max, rng):
    """within-class interpolation (SMOTE-like) among fit rows; partner from 10 nearest same-class rows, other uri."""
    Zc = np.where(np.isnan(A[:, cont]), np.nanmedian(A[:, cont], axis=0), A[:, cont]) / sd[cont]
    out = []
    for c in (0, 1):
        ix = np.flatnonzero(yA == c)
        nn = NearestNeighbors(n_neighbors=min(16, len(ix))).fit(Zc[ix])
        _, ind = nn.kneighbors(Zc[ix])
        for _k in range(K):
            part = np.empty(len(ix), int)
            for a in range(len(ix)):
                cand = [ix[j] for j in ind[a, 1:] if u[ix[j]] != u[ix[a]]][:10]
                part[a] = rng.choice(cand) if cand else ix[a]
            lam = rng.uniform(0, A_max, len(ix))
            S = A[ix].copy()
            P2 = A[part]
            d = S[:, cont] + lam[:, None] * (P2[:, cont] - S[:, cont])
            S[:, cont] = np.where(np.isnan(d), S[:, cont], d)
            out.append((S, np.full(len(ix), c, float)))
    return np.vstack([o[0] for o in out]), np.concatenate([o[1] for o in out])


def teacher_probs(kind, fit_idx, apply_list, seed, r=None, val_idx=None, final=False):
    """Returns list of teacher prob arrays for apply_list (None where unavailable)."""
    out = []
    if kind in ('self', 'selfch'):
        A, Bs = apply_list[0], apply_list[1:]
        pred = fit_model(A, y[fit_idx].astype(float), base_weights(fit_idx), seed + 777)
        selfp = [pred(B) for B in Bs]
    if kind == 'self':
        return selfp
    ch = TEACH['ch']
    if final:
        tch = [ch['test_full']]
    else:
        tch = [ch['oof'][r, val_idx], None]
    if kind == 'selfch':
        return [None if t is None else 0.5 * t + 0.5 * s for t, s in zip(tch, selfp)]
    return tch


def train_student(fit_idx, A, U_list, seed):
    """A: fit-row matrix; U_list: list of (matrix, teacher prob) pseudo-labelled sets."""
    yA = y[fit_idx].astype(float)
    wA = base_weights(fit_idx)
    rng = np.random.default_rng(seed + 31)
    parts = [(A, yA, wA)]
    if 'noise' in opts:
        s, K, W = float(opts['noise'][0]), int(opts['noise'][1]), float(opts['noise'][2])
        for _ in range(K):
            parts.append((jitter(A, s, rng), yA, wA * W / K))
    if 'mix' in opts:
        K, W, Am = int(opts['mix'][0]), float(opts['mix'][1]), float(opts['mix'][2])
        S, yS = mixup(A, yA.astype(int), uri[fit_idx], K, Am, rng)
        parts.append((S, yS, np.full(len(yS), W)))
    if 'pl' in opts:
        mode, W = opts['pl'][2], float(opts['pl'][3])
        for U, pu in U_list:
            if mode == 'soft':
                parts.append((U, pu, np.full(len(pu), W)))
            elif mode.startswith('sharp'):
                T = float(mode[5:])
                a, b = pu ** (1 / T), (1 - pu) ** (1 / T)
                parts.append((U, a / (a + b), np.full(len(pu), W)))
            elif mode.startswith('hard'):
                q = float(mode[4:])
                # confidence by rank within the pseudo set; positives at the top q*(pos rate), negatives bottom
                o = np.argsort(pu)
                nlo, nhi = int(q * len(pu) * (1 - y.mean())), int(q * len(pu) * y.mean())
                sel = np.concatenate([o[:nlo], o[len(o) - nhi:]])
                lab = np.concatenate([np.zeros(nlo), np.ones(nhi)])
                parts.append((U[sel], lab, np.full(len(sel), W)))
    A2 = np.vstack([p[0] for p in parts])
    y2 = np.concatenate([p[1] for p in parts])
    w2 = np.concatenate([p[2] for p in parts])
    return fit_model(A2, y2, w2, seed)


def predict(pred, B, seed):
    if 'tta' not in opts:
        return pred(B)
    s, K = float(opts['tta'][0]), int(opts['tta'][1])
    rng = np.random.default_rng(seed + 99)
    return np.mean([pred(B)] + [pred(jitter(B, s, rng)) for _ in range(K)], axis=0)


def run_fold(k):
    r, fit_idx, val_idx, elig = splits[k]
    F_fit, F_val, F_te = C['LF'][('cv', k)]
    A, (Bv, Bt) = mats(fit_idx, [X[val_idx], Xt], (F_fit, [F_val, F_te]))
    seed = k + SEEDOFF
    U = []
    if 'pl' in opts:
        kind, sset = opts['pl'][0], opts['pl'][1]
        tv, tt = teacher_probs(kind, fit_idx, [A, Bv, Bt], seed, r=r, val_idx=val_idx)
        U.append((Bv[elig], tv[elig]))
        if sset == 'valtest':
            assert tt is not None
            U.append((Bt, tt))
    pred = train_student(fit_idx, A, U, seed)
    return predict(pred, Bv, seed), predict(pred, Bt, seed)


def run_full(s):
    F_fit, _, F_te = C['LF'][('full', s)]
    fit_idx = np.arange(n)
    A, (Bt,) = mats(fit_idx, [Xt], (F_fit, [F_te]))
    seed = s + SEEDOFF
    U = []
    if 'pl' in opts:
        (tt,) = teacher_probs(opts['pl'][0], fit_idx, [A, Bt], seed, final=True)
        U.append((Bt, tt))
    pred = train_student(fit_idx, A, U, seed)
    return predict(pred, Bt, seed)


t0 = time.time()
res = H.run_cv(lambda fi, vi, k: run_fold(k), splits, y, tag=f'{MODEL} {VARIANT} nrep{NREP}', verbose=False)
RUNS = WORK + '/r3_aug/cache/runs/'
tagf = f'{MODEL}__{VARIANT.replace(";", "_").replace(":", "-").replace("=", "-")}__s{SEEDOFF}'
np.savez(RUNS + tagf + '.npz', oof=res['oof'], aucs=res['aucs'], test_foldavg=res['test_foldavg'])
bf = RUNS + f'{MODEL}____s{SEEDOFF}.npz'
base = dict(np.load(bf)) if os.path.exists(bf) else H.load_res('eda_chl_lgbm' if MODEL == 'lgbm' else 'eda_chl_et'
       if MODEL == 'et' else 'ch_lgbm_proj' if MODEL == 'lgbmnl' else 'ch_et_proj')
d = res['aucs'] - base['aucs'][:NREP]
print(f'[{MODEL} | {VARIANT} | nrep{NREP}] AUC {res["auc"]:.5f} per-rep {np.round(res["aucs"], 4)}  '
      f'vs base {d.mean():+.5f} ({(d > 0).sum()}/{NREP})  ({time.time()-t0:.0f}s)', flush=True)
if SAVE:
    assert NREP == 5
    full = np.mean([run_full(s) for s in range(3)], axis=0)
    assert np.isfinite(full).all() and len(full) == m
    H.save(SAVE, res, full)
    print('saved', SAVE, ' corr(full,foldavg)', np.corrcoef(full, res['test_foldavg'])[0, 1].round(4), flush=True)
