"""Label-derived features (target encodings, neighbour labels) and collective smoothing -- agent `te`.

Row convention (same as XA = concat(train_features, test_features)): rows 0..n-1 are TRAIN (labels
y[0..n-1]), rows n..n+m-1 are TEST (never labelled).

Leakage safeguards
------------------
1. `encode(ctx, vals, wts, dst)` builds features for the `dst` rows from the rows with wts > 0 only.
   `label_features` sets wts > 0 exactly on `fit_idx`, so features for validation / test rows use
   only the labels of the rows the model is trained on. Validation labels are never read.
2. Features for the fit rows themselves are OUT-OF-FOLD: fit_idx is split into `n_inner` folds with
   StratifiedGroupKFold grouped by uri, and each inner fold is encoded from the other folds only. A fit
   row therefore never sees its own label nor the label of another copy of the same song.
3. Every statistic additionally drops source rows with the same uri as the target row (same-date pairs
   and neighbour lists exclude same-uri rows, key statistics subtract the uri's own contribution), so
   even overlapping index sets cannot leak a row's own label.
4. Everything else in `Ctx` (keys, audio space, neighbour lists) is label-free and may use test features.
(test_leakage.py asserts 1-3: flipping validation labels or a fold's own labels changes nothing.)

Usage (inside fit_predict(fit_idx, val_idx, k) of harness.run_cv; final fit: fit_idx = arange(n)):
    F_fit, (F_val, F_test) = label_features(XA, y, fit_idx, [val_idx, arange(n, n+m)], tr, te, seed=k,
                                            spec=BEST_SPEC)
    then concat F_fit / F_val / F_test column-wise to the label-free feature rows.

CV verdict (LightGBM teammate params, 5x5 eligible-row CV, 3 seeds; see log.md):
  BEST_SPEC on 67 feats: 0.83743 vs 0.83446 (+0.0030, 5/5); on fe v2 55 feats: 0.84039 vs 0.83710 (+0.0033, 5/5).
  ym / year TE, global audio kNN, audio-weighted date-mates, album size: no gain. smooth_dates: harmful.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # work/ (time_features)
from time_features import parse_date  # noqa: E402

SIM_COLS = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
            'instrumentalness', 'tempo', 'duration_ms']
# recommended: inverse-distance weighted label rate of the 40 audio-nearest labelled songs released within
# +-1 year; date target encoding smoothed (m=10) toward the year-level rate; count of labelled date-mates
BEST_SPEC = ('knnw_yr_40', 'teh_date_m10', 'cnt_date')
_CTX = {}


class Ctx:
    """Label-free precomputation over train+test rows: grouping keys, audio space, neighbour lists."""

    def __init__(self, tr, te, kmax=200, year_window=1.0, album_t=2.5):
        al = pd.concat([tr, te], ignore_index=True)
        self.n, self.m = len(tr), len(te)
        N = self.N = len(al)
        raw = al['track_album_release_date'].astype(str)
        d = parse_date(raw)
        self.uri = pd.factorize(al['uri'])[0]
        self.key = {'date': pd.factorize(raw)[0],
                    'ym': pd.factorize(d.dt.to_period('M').astype(str))[0],
                    'year': pd.factorize(d.dt.year)[0]}
        self.t = (d.dt.year + (d.dt.dayofyear - 1) / 365.25).to_numpy()
        A = al[SIM_COLS]
        Z = QuantileTransformer(n_quantiles=200, output_distribution='normal').fit_transform(
            A.fillna(A.median()))
        self.Z = Z / Z.std(axis=0)
        # all same-date pairs (i, j), i != j, different uri, with audio distance
        I, J, D = [], [], []
        for idx in pd.Series(np.arange(N)).groupby(self.key['date']).indices.values():
            if len(idx) < 2:
                continue
            Dm = np.linalg.norm(self.Z[idx, None, :] - self.Z[None, idx, :], axis=2)
            ii, jj = np.nonzero(self.uri[idx][:, None] != self.uri[idx][None, :])
            I.append(idx[ii]); J.append(idx[jj]); D.append(Dm[ii, jj])
        self.pi, self.pj, self.pd = np.concatenate(I), np.concatenate(J), np.concatenate(D)
        # global audio neighbours (sorted, same-uri rows removed)
        nn = NearestNeighbors(n_neighbors=kmax + 4).fit(self.Z)
        dist, ind = nn.kneighbors(self.Z)
        self.nbr = {'glob': self._strip(ind, dist, kmax)}
        # audio neighbours released within +-year_window years
        ind_y = np.zeros((N, kmax), int)
        dst_y = np.full((N, kmax), np.inf)
        for i in range(N):
            c = np.nonzero((np.abs(self.t - self.t[i]) <= year_window) & (self.uri != self.uri[i]))[0]
            dc = np.linalg.norm(self.Z[c] - self.Z[i], axis=1)
            o = np.argsort(dc)[:kmax]
            ind_y[i, :len(o)], dst_y[i, :len(o)] = c[o], dc[o]
        self.nbr['yr'] = (ind_y, dst_y)
        # likely albums: cluster each date group (one row per uri) on audio (label-free)
        self.key['alb'] = album_ids(al, raw, self.uri, t=album_t)
        self.albsize = np.bincount(self.key['alb'][first_by(self.uri)])[self.key['alb']].astype(float)

    def _strip(self, ind, dist, kmax):
        keep = self.uri[ind] != self.uri[:, None]
        oi = np.zeros((len(ind), kmax), int)
        od = np.full((len(ind), kmax), np.inf)
        for r in range(len(ind)):
            k = ind[r][keep[r]][:kmax]
            oi[r, :len(k)], od[r, :len(k)] = k, dist[r][keep[r]][:kmax]
        return oi, od


ALB_COLS = ['loudness', 'duration_ms', 'instrumentalness', 'speechiness', 'acousticness', 'energy',
            'danceability', 'liveness', 'valence']


def first_by(codes):
    """Index of the first row of each code (one representative row per uri)."""
    return pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()


def album_ids(al, raw, uri, t=2.5, cols=ALB_COLS):
    """Label-free album proxy (same idea as the fe agent's album_assign): within each release date,
    average-linkage clustering of the distinct songs (one row per uri) on log/standardized audio,
    cut at distance t. All copies of a uri get the same album id."""
    from scipy.cluster.hierarchy import fcluster, linkage
    first = first_by(uri)
    A = al[cols].iloc[first].astype(float).copy()
    A['duration_ms'] = np.log(A['duration_ms'])
    A['instrumentalness'] = np.log10(A['instrumentalness'] + 1e-4)
    A['speechiness'] = np.log(A['speechiness'])
    A['liveness'] = np.log(A['liveness'])
    Z = ((A - A.mean()) / A.std()).to_numpy()
    alb = np.zeros(len(first), int)
    gid = 0
    for idx in pd.Series(np.arange(len(first))).groupby(raw.iloc[first].to_numpy()).indices.values():
        if len(idx) == 1:
            alb[idx] = gid
            gid += 1
            continue
        lab = fcluster(linkage(Z[idx], 'average'), t, 'distance')
        alb[idx] = gid + lab - 1
        gid += lab.max()
    return alb[uri]   # uri codes index the representative order (first_by is sorted by code)


def get_ctx(tr, te, **kw):
    key = (id(tr), id(te), tuple(sorted(kw.items())))
    if key not in _CTX:
        _CTX[key] = Ctx(tr, te, **kw)
    return _CTX[key]


# ----------------------------------------------------------------------------------------------
# encoder: features for dst rows from "source" rows (wts > 0) with values `vals` (labels or probs)
# ----------------------------------------------------------------------------------------------
def _key_stats(ctx, key, vals, wts, dst):
    """Weighted (count, sum) over source rows sharing `key` with each dst row, same uri excluded."""
    k = ctx.key[key]
    C = np.bincount(k, weights=wts, minlength=k.max() + 1)
    S = np.bincount(k, weights=wts * vals, minlength=k.max() + 1)
    Cu = np.bincount(ctx.uri, weights=wts, minlength=ctx.uri.max() + 1)   # same uri => same date
    Su = np.bincount(ctx.uri, weights=wts * vals, minlength=ctx.uri.max() + 1)
    u = ctx.uri[dst]
    return C[k[dst]] - Cu[u], S[k[dst]] - Su[u]


def _pair_stats(ctx, vals, wts, dst, bw):
    """Audio-kernel-weighted (count, sum) over same-date source rows; bw=None -> unweighted."""
    isdst = np.zeros(ctx.N, bool)
    isdst[dst] = True
    sel = isdst[ctx.pi] & (wts[ctx.pj] > 0)
    i, j, d = ctx.pi[sel], ctx.pj[sel], ctx.pd[sel]
    w = wts[j] * (1.0 if bw is None else np.exp(-0.5 * (d / bw) ** 2))
    C = np.bincount(i, weights=w, minlength=ctx.N)
    S = np.bincount(i, weights=w * vals[j], minlength=ctx.N)
    return C[dst], S[dst]


def _knn(ctx, kind, vals, wts, dst, k, weighted=False):
    """Mean value of the k nearest source rows (audio space; 'glob' or within +-1 year 'yr')."""
    ind, dist = ctx.nbr[kind]
    ind, dist = ind[dst], dist[dst]
    ok = (wts[ind] > 0) & np.isfinite(dist)
    sel = ok & (np.cumsum(ok, axis=1) <= k)
    w = sel * (1.0 / (dist + 0.25) if weighted else 1.0) * wts[ind]
    return (w * vals[ind]).sum(1) / np.maximum(w.sum(1), 1e-12)


def encode(ctx, vals, wts, dst, spec):
    """Feature frame for rows `dst` using ONLY rows with wts > 0 as sources.

    vals: (N,) values of source rows (0/1 labels, or predicted probabilities for pseudo-labels).
    wts : (N,) source weights (0 = not a source). spec: list of feature names:
      cnt_<key>            weighted count of source rows sharing the key (date / ym / year / alb)
      te_<key>_m<M>        (sum + M*prior)/(count + M), prior = global source mean
      teh_date_m<M>        same, but prior = year-level encoding (te_year_m10)  [hierarchical]
      teh_alb_m<M>         album encoding, prior = teh_date_m<M>  (alb = label-free album clusters)
      albsize              label-free: number of distinct songs in the row's album cluster (pool)
      nbd_s<BW>_m<M>       date-mates weighted by audio kernel exp(-d^2/2BW^2), smoothed toward prior
      knn_<glob|yr>_<K>    mean value of the K audio-nearest source rows (yr: released within +-1 y)
      knnw_<glob|yr>_<K>   weighted version, weight 1/(d + 0.25), d = distance in the audio space
    Audio space = 9 SIM_COLS, quantile-normalised on train+test. Same-uri rows are never sources.
    """
    src = wts > 0
    prior = float((wts[src] * vals[src]).sum() / wts[src].sum())
    out = {}
    for name in spec:
        p = name.split('_')
        if p[0] == 'cnt':
            out[name] = _key_stats(ctx, p[1], vals, wts, dst)[0]
        elif p[0] in ('te', 'teh'):
            M = float(p[2][1:])
            C, S = _key_stats(ctx, p[1], vals, wts, dst)
            pr = prior
            if p[0] == 'teh':
                Cy, Sy = _key_stats(ctx, 'year', vals, wts, dst)
                pr = (Sy + 10 * prior) / (Cy + 10)
                if p[1] == 'alb':                      # album -> date -> year -> global
                    Cd, Sd = _key_stats(ctx, 'date', vals, wts, dst)
                    pr = (Sd + M * pr) / (Cd + M)
            out[name] = (S + M * pr) / (C + M)
        elif p[0] == 'albsize':
            out[name] = ctx.albsize[dst]                # label-free companion (album size in pool)
        elif p[0] == 'nbd':
            bw, M = float(p[1][1:]), float(p[2][1:])
            C, S = _pair_stats(ctx, vals, wts, dst, bw)
            out[name] = (S + M * prior) / (C + M)
        elif p[0] in ('knn', 'knnw'):
            out[name] = _knn(ctx, p[1], vals, wts, dst, int(p[2]), weighted=p[0] == 'knnw')
        else:
            raise ValueError(name)
    return pd.DataFrame(out)


def label_features(XA, y, fit_idx, apply_idx_list, tr, te, seed=0, spec=BEST_SPEC, n_inner=5):
    """Label features for the fit rows (out-of-fold) and for each array of apply rows.

    XA: train+test feature frame (only used to check the row convention).
    y: train labels (len n). fit_idx: train rows whose labels may be used. apply_idx_list: arrays of
    row indices into XA (validation rows and/or test rows n..n+m-1).
    Returns (F_fit, [F_apply, ...]) as DataFrames aligned with fit_idx / each apply array.
    Fit rows: inner StratifiedGroupKFold(n_inner, groups=uri) -> each inner fold encoded from the other
    folds. Apply rows: encoded from all fit_idx labels. Same-uri rows are always excluded.
    """
    ctx = get_ctx(tr, te)
    assert len(XA) == ctx.N and len(y) == ctx.n
    fit_idx = np.asarray(fit_idx)
    vals = np.zeros(ctx.N)
    vals[:ctx.n] = y
    F_fit = None
    sgkf = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
    for a, b in sgkf.split(fit_idx, y[fit_idx], ctx.uri[fit_idx]):
        wts = np.zeros(ctx.N)
        wts[fit_idx[a]] = 1.0
        e = encode(ctx, vals, wts, fit_idx[b], spec)
        if F_fit is None:
            F_fit = pd.DataFrame(np.nan, index=range(len(fit_idx)), columns=e.columns)
        F_fit.iloc[b] = e.to_numpy()
    wts = np.zeros(ctx.N)
    wts[fit_idx] = 1.0
    return F_fit, [encode(ctx, vals, wts, np.asarray(a), spec) for a in apply_idx_list]


# ----------------------------------------------------------------------------------------------
# collective (transductive) inference over release-date groups
# ----------------------------------------------------------------------------------------------
def smooth_dates(ctx, p_u, u_idx, l_idx, y, alpha, bw=None, n_iter=1, lab_w=1.0, unl_w=1.0):
    """Post-hoc smoothing of predictions toward release-date mates (labels + predictions).

    u_idx: unlabelled rows (CV: eligible validation rows + test rows; final: test rows) with model
           predictions p_u. l_idx: labelled rows (the rows the model was fit on), labels y[l_idx].
    For each unlabelled u, mates = same-date rows with a different uri in L or U; mate value = its
    label (L, weight lab_w) or its current smoothed prediction (U, weight unl_w), times an optional
    audio kernel exp(-d^2/2bw^2). Update (Jacobi, n_iter times; p_u itself stays the anchor):
        q_u = (p_u + alpha * sum_j w_uj v_j) / (1 + alpha * sum_j w_uj)
    alpha -> inf returns the plain mates' mean (rows without mates keep p_u); a fixed-weight mix
    (1-lam) p_u + lam * mates_mean is built from that in collective_eval2.py.
    Only labels of l_idx are read (never those of u_idx), and a row never uses its own uri.
    NOT RECOMMENDED: in CV (U = eligible val + real test rows) every setting lost AUC on top of the
    label features (alpha=0.05: -0.013; best fixed mix +0.0003, 3/5). See log.md.
    """
    u_idx, l_idx = np.asarray(u_idx), np.asarray(l_idx)
    isU = np.zeros(ctx.N, bool)
    isU[u_idx] = True
    isL = np.zeros(ctx.N, bool)
    isL[l_idx] = True
    assert not (isU & isL).any()
    sel = isU[ctx.pi] & (isU[ctx.pj] | isL[ctx.pj])
    i, j, d = ctx.pi[sel], ctx.pj[sel], ctx.pd[sel]
    w = np.where(isL[j], lab_w, unl_w) * (1.0 if bw is None else np.exp(-0.5 * (d / bw) ** 2))
    v = np.zeros(ctx.N)
    v[l_idx] = y[l_idx]
    p0 = np.zeros(ctx.N)
    p0[u_idx] = p_u
    v[u_idx] = p_u
    Wsum = np.bincount(i, weights=w, minlength=ctx.N)
    for _ in range(n_iter):
        S = np.bincount(i, weights=w * v[j], minlength=ctx.N)
        q = (p0 + alpha * S) / (1 + alpha * Wsum)
        v[u_idx] = q[u_idx]
    return v[u_idx]
