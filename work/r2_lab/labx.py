"""Extended out-of-fold label features -- round-2 agent `lab`.

Derived from work/te/label_features.py (copied, that file is NOT edited). Same row convention: rows 0..n-1 of
XA are TRAIN (labels y), rows n..n+m-1 are TEST (never labelled).

Leakage discipline (identical to the te module, checked by r2_lab/test_leakage_lab.py)
  1. encode(ctx, vals, wts, dst, spec) reads labels ONLY of rows with wts > 0 (the "sources").
     label_features() puts sources exclusively on fit_idx rows; validation / test labels are never read.
  2. Features of the fit rows are out-of-fold: fit_idx is split by StratifiedGroupKFold(n_inner, groups=uri),
     each inner fold is encoded from the other folds only (a row never sees its own label or a copy's label).
  3. All statistics drop source rows with the same uri as the target row (pair lists / neighbour lists exclude
     same-uri rows, key statistics subtract the uri's own contribution).
  4. Everything in Ctx (keys, audio spaces, neighbour lists, k-means clusters, album clusters) is label-free.
  dedup=True ("copies count once"): among the sources, each uri keeps ONE representative row whose value is
  the mean label of its source copies (soft label for the 32 conflicting songs). Uses only source labels.

New feature names (besides the te ones: cnt_<key>, te_<key>_m<M>, teh_<key>_m<M>, knn(w)_<list>_<K>, nbd_...):
  knnwx_<list>_<K>    as knnw but same-release-date sources excluded (style signal without album-mates)
  knn(w)_<list>_<K>   list in: yr (+-1 y, 9-d audio), yr05 (+-0.5 y), yr3 (+-3 y),
                      yrA (+-1 y, energy/loudness/acousticness/instrumentalness),
                      yrB (+-1 y, danceability/tempo/valence/speechiness), glob (all dates)
  chain_<k1-k2-..>_m<M>  hierarchical empirical Bayes: global -> k1 -> k2 -> ... (each level shrunk with M)
  tek_date_m<M>       date TE shrunk toward the knnw_yr_40 rate of the same row (kNN prior instead of year)
  dmin_hi / dmin_lo   audio distance to the nearest labelled High / Low date-mate (9.0 if none)
  teh_yc8_m<M>        TE of (year x 8 k-means audio clusters), prior = year TE
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

WORK = './work'
sys.path.insert(0, WORK)
from time_features import parse_date  # noqa: E402

SIM_COLS = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
            'instrumentalness', 'tempo', 'duration_ms']
SUBSPACES = {'A': ['energy', 'loudness', 'acousticness', 'instrumentalness'],
             'B': ['danceability', 'tempo', 'valence', 'speechiness']}
BEST_SPEC = ('knnw_yr_40', 'teh_date_m10', 'cnt_date')
BIG = 9.0
_CTX = {}


def window_nbrs(Z, t, uri, window, kmax):
    """For every row: audio-nearest rows released within +-window years (different uri), sorted."""
    N = len(Z)
    ind_y = np.zeros((N, kmax), int)
    dst_y = np.full((N, kmax), np.inf)
    for i in range(N):
        c = np.nonzero((np.abs(t - t[i]) <= window) & (uri != uri[i]))[0]
        dc = np.linalg.norm(Z[c] - Z[i], axis=1)
        o = np.argsort(dc)[:kmax]
        ind_y[i, :len(o)], dst_y[i, :len(o)] = c[o], dc[o]
    return ind_y, dst_y


class Ctx:
    """Label-free precomputation over train+test rows: grouping keys, audio spaces, neighbour lists."""

    def __init__(self, tr, te, kmax=200, kmax_yr=500, album_t=2.5):
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
        nn = NearestNeighbors(n_neighbors=kmax + 4).fit(self.Z)
        dist, ind = nn.kneighbors(self.Z)
        self.nbr = {'glob': self._strip(ind, dist, kmax)}
        # time-windowed audio neighbour lists (first 200 entries of 'yr' == te module's 'yr' list)
        self.nbr['yr'] = window_nbrs(self.Z, self.t, self.uri, 1.0, kmax_yr)
        self.nbr['yr05'] = window_nbrs(self.Z, self.t, self.uri, 0.5, kmax_yr)
        self.nbr['yr3'] = window_nbrs(self.Z, self.t, self.uri, 3.0, kmax_yr)
        for s, cols in SUBSPACES.items():
            ci = [SIM_COLS.index(c) for c in cols]
            Zs = self.Z[:, ci] / np.sqrt(len(ci) / len(SIM_COLS))
            self.nbr['yr' + s] = window_nbrs(Zs, self.t, self.uri, 1.0, kmax_yr)
        # likely albums (label-free per-date clusters) and coarse audio clusters x year
        self.key['alb'] = album_ids(al, raw, self.uri, t=album_t)
        self.albsize = np.bincount(self.key['alb'][first_by(self.uri)])[self.key['alb']].astype(float)
        km = KMeans(8, n_init=10, random_state=0).fit(self.Z).labels_
        self.key['yc8'] = pd.factorize(pd.Series(self.key['year']).astype(str) + '_' + pd.Series(km).astype(str))[0]

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
    return pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()


def album_ids(al, raw, uri, t=2.5, cols=ALB_COLS):
    """Label-free album proxy (copied from te): per-date average-linkage clusters of distinct songs."""
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
    return alb[uri]


def get_ctx(tr, te, **kw):
    key = (id(tr), id(te), tuple(sorted(kw.items())))
    if key not in _CTX:
        _CTX[key] = Ctx(tr, te, **kw)
    return _CTX[key]


# ----------------------------------------------------------------------------------------------
def _key_stats(ctx, key, vals, wts, dst):
    k = ctx.key[key]
    C = np.bincount(k, weights=wts, minlength=k.max() + 1)
    S = np.bincount(k, weights=wts * vals, minlength=k.max() + 1)
    Cu = np.bincount(ctx.uri, weights=wts, minlength=ctx.uri.max() + 1)
    Su = np.bincount(ctx.uri, weights=wts * vals, minlength=ctx.uri.max() + 1)
    u = ctx.uri[dst]
    return C[k[dst]] - Cu[u], S[k[dst]] - Su[u]


def _pair_stats(ctx, vals, wts, dst, bw):
    isdst = np.zeros(ctx.N, bool)
    isdst[dst] = True
    sel = isdst[ctx.pi] & (wts[ctx.pj] > 0)
    i, j, d = ctx.pi[sel], ctx.pj[sel], ctx.pd[sel]
    w = wts[j] * (1.0 if bw is None else np.exp(-0.5 * (d / bw) ** 2))
    C = np.bincount(i, weights=w, minlength=ctx.N)
    S = np.bincount(i, weights=w * vals[j], minlength=ctx.N)
    return C[dst], S[dst]


def _pair_min(ctx, vals, wts, dst):
    """Audio distance to the nearest same-date source labelled High (vals >= .5) / Low; BIG if none."""
    isdst = np.zeros(ctx.N, bool)
    isdst[dst] = True
    sel = isdst[ctx.pi] & (wts[ctx.pj] > 0)
    i, j, d = ctx.pi[sel], ctx.pj[sel], ctx.pd[sel]
    hi = vals[j] >= 0.5
    dH = np.full(ctx.N, BIG)
    dL = np.full(ctx.N, BIG)
    np.minimum.at(dH, i[hi], d[hi])
    np.minimum.at(dL, i[~hi], d[~hi])
    return np.minimum(dH[dst], BIG), np.minimum(dL[dst], BIG)


def _knn(ctx, kind, vals, wts, dst, k, weighted=False, xdate=False):
    ind, dist = ctx.nbr[kind]
    ind, dist = ind[dst], dist[dst]
    ok = (wts[ind] > 0) & np.isfinite(dist)
    if xdate:                                   # exclude same-release-date sources (album-mates)
        ok &= ctx.key['date'][ind] != ctx.key['date'][dst][:, None]
    sel = ok & (np.cumsum(ok, axis=1) <= k)
    w = sel * (1.0 / (dist + 0.25) if weighted else 1.0) * wts[ind]
    return (w * vals[ind]).sum(1) / np.maximum(w.sum(1), 1e-12)


def encode(ctx, vals, wts, dst, spec):
    """Feature frame for rows `dst` using ONLY rows with wts > 0 as sources (see module docstring)."""
    src = wts > 0
    prior = float((wts[src] * vals[src]).sum() / wts[src].sum())
    cache = {}

    def ks(key):
        if key not in cache:
            cache[key] = _key_stats(ctx, key, vals, wts, dst)
        return cache[key]

    def year_prior():
        Cy, Sy = ks('year')
        return (Sy + 10 * prior) / (Cy + 10)

    out = {}
    pm = None
    for name in spec:
        p = name.split('_')
        if p[0] == 'cnt':
            out[name] = ks(p[1])[0]
        elif p[0] in ('te', 'teh'):
            M = float(p[2][1:])
            C, S = ks(p[1])
            pr = prior
            if p[0] == 'teh':
                pr = year_prior()
                if p[1] == 'alb':
                    Cd, Sd = ks('date')
                    pr = (Sd + M * pr) / (Cd + M)
            out[name] = (S + M * pr) / (C + M)
        elif p[0] == 'chain':
            M = float(p[2][1:])
            pr = prior
            for key in p[1].split('-'):
                C, S = ks(key)
                pr = (S + M * pr) / (C + M)
            out[name] = pr
        elif p[0] == 'tek':
            M = float(p[2][1:])
            C, S = ks(p[1])
            pr = _knn(ctx, 'yr', vals, wts, dst, 40, weighted=True)
            out[name] = (S + M * pr) / (C + M)
        elif p[0] == 'dmin':
            if pm is None:
                pm = _pair_min(ctx, vals, wts, dst)
            out[name] = pm[0] if p[1] == 'hi' else pm[1]
        elif p[0] == 'albsize':
            out[name] = ctx.albsize[dst]
        elif p[0] == 'nbd':
            bw, M = float(p[1][1:]), float(p[2][1:])
            C, S = _pair_stats(ctx, vals, wts, dst, bw)
            out[name] = (S + M * prior) / (C + M)
        elif p[0] in ('knn', 'knnw', 'knnwx'):
            out[name] = _knn(ctx, p[1], vals, wts, dst, int(p[2]), weighted=p[0] != 'knn', xdate=p[0] == 'knnwx')
        else:
            raise ValueError(name)
    return pd.DataFrame(out)


def sources(ctx, vals, idx, dedup=False):
    """(vals, wts) with sources = rows idx. dedup: one representative row per uri, value = mean source label."""
    idx = np.asarray(idx)
    wts = np.zeros(ctx.N)
    if not dedup:
        wts[idx] = 1.0
        return vals, wts
    u = ctx.uri[idx]
    nu = ctx.uri.max() + 1
    cnt = np.bincount(u, minlength=nu)
    s = np.bincount(u, weights=vals[idx], minlength=nu)
    _, first = np.unique(u, return_index=True)
    rep = idx[first]
    v = vals.copy()
    v[rep] = s[ctx.uri[rep]] / cnt[ctx.uri[rep]]
    wts[rep] = 1.0
    return v, wts


def label_features(XA, y, fit_idx, apply_idx_list, tr, te, seed=0, spec=BEST_SPEC, n_inner=5, dedup=False):
    """Label features for the fit rows (out-of-fold, inner folds grouped by uri) and each apply array.
    Same contract as te/label_features.label_features (+ dedup option)."""
    ctx = get_ctx(tr, te)
    assert len(XA) == ctx.N and len(y) == ctx.n
    fit_idx = np.asarray(fit_idx)
    vals = np.zeros(ctx.N)
    vals[:ctx.n] = y
    F_fit = None
    sgkf = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
    for a, b in sgkf.split(fit_idx, y[fit_idx], ctx.uri[fit_idx]):
        v, wts = sources(ctx, vals, fit_idx[a], dedup)
        e = encode(ctx, v, wts, fit_idx[b], spec)
        if F_fit is None:
            F_fit = pd.DataFrame(np.nan, index=range(len(fit_idx)), columns=e.columns)
        F_fit.iloc[b] = e.to_numpy()
    v, wts = sources(ctx, vals, fit_idx, dedup)
    return F_fit, [encode(ctx, v, wts, np.asarray(a), spec) for a in apply_idx_list]
