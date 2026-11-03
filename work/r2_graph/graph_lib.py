"""Graph-based transductive features over ALL train+test songs (agent r2 `graph`).

Nodes = distinct songs (uri); every row maps to its uri node (copies of a song share one node), so a
row's own label / a copy's label can never be a seed for itself (a node is either a seed or not).

Graph (label-free, built once from train+test features):
  * date edges : all pairs of songs with the identical release-date string, weight w_date * K_audio
  * knn edges  : k nearest songs in the space [Z_audio, t / tau] (Z = 9 SIM_COLS quantile-normal,
                 standardised; t = continuous release time in years), Gaussian/inverse-distance weight
  symmetrised (W = max(W, W^T)).

Propagation (label-dependent; seeds = ONLY the given seed rows' labels):
  * harmonic(m): labelled nodes clamped to their (mean) label, unlabelled nodes iterate
        f_u = (sum_j w_uj f_j + m * prior_u) / (sum_j w_uj + m)
    (absorbing random walk / Zhu-Ghahramani harmonic function with a prior "dongle"; multi-hop through
    unlabelled nodes, e.g. unlabelled date-mates pass on their own audio neighbourhood).
  * ppr(alpha): personalised-PageRank weighted label mean (label spreading ratio form)
        score_i = (PPR_i . (L*y) + m*prior_i) / (PPR_i . L + m)
  prior_u = year-level label rate of the seeds, smoothed toward the global seed rate (m=10).

graph_label_features(...) mirrors te/label_features.label_features: fit rows get OUT-OF-FOLD values from
inner StratifiedGroupKFold(5, groups=uri) folds (inner fold b is unlabelled, seeds = other inner folds of
fit_idx); apply rows (validation / test) use all fit_idx rows as seeds. Validation and test rows are
always unlabelled nodes, exactly as at test time.
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

WORK = './work'
sys.path.insert(0, WORK)
from time_features import parse_date  # noqa: E402

SIM_COLS = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness',
            'instrumentalness', 'tempo', 'duration_ms']


class SongGraph:
    def __init__(self, tr, te):
        al = pd.concat([tr, te], ignore_index=True)
        self.n, self.m = len(tr), len(te)
        self.R = len(al)
        self.node = pd.factorize(al['uri'])[0]                      # row -> node
        self.Nn = self.node.max() + 1
        first = pd.Series(np.arange(self.R)).groupby(self.node).first().to_numpy()
        self.first = first
        raw = al['track_album_release_date'].astype(str)
        d = parse_date(raw)
        t = (d.dt.year + (d.dt.dayofyear - 1) / 365.25).to_numpy()
        self.t = t[first]
        self.year = pd.factorize(d.dt.year.to_numpy()[first])[0]
        self.date = pd.factorize(raw.to_numpy()[first])[0]
        A = al[SIM_COLS].iloc[first]
        Z = QuantileTransformer(n_quantiles=200, output_distribution='normal').fit_transform(A.fillna(A.median()))
        self.Z = Z / Z.std(axis=0)
        self._cache = {}

    # ------------------------------------------------------------------ edges
    def date_edges(self, bw=None):
        I, J, D = [], [], []
        for idx in pd.Series(np.arange(self.Nn)).groupby(self.date).indices.values():
            if len(idx) < 2:
                continue
            Dm = np.linalg.norm(self.Z[idx, None, :] - self.Z[None, idx, :], axis=2)
            ii, jj = np.nonzero(~np.eye(len(idx), dtype=bool))
            I.append(idx[ii]); J.append(idx[jj]); D.append(Dm[ii, jj])
        I, J, D = np.concatenate(I), np.concatenate(J), np.concatenate(D)
        w = np.ones(len(I)) if bw is None else np.exp(-0.5 * (D / bw) ** 2)
        return sp.csr_matrix((w, (I, J)), shape=(self.Nn, self.Nn))

    def knn_edges(self, k=20, tau=1.0, kernel='inv', bw=None, same_date=False):
        """k nearest songs in [Z, t/tau]; kernel 'inv' -> 1/(d+0.25) normalised to max 1 per row... we keep
        raw 1/(d+0.25)*0.25 (<=1); 'gauss' -> exp(-d^2/2bw^2) with bw = median kNN distance if None."""
        S = np.column_stack([self.Z, self.t / tau])
        nn = NearestNeighbors(n_neighbors=k + 1).fit(S)
        dist, ind = nn.kneighbors(S)
        dist, ind = dist[:, 1:], ind[:, 1:]
        if not same_date:
            pass
        if kernel == 'inv':
            w = 0.25 / (dist + 0.25)
        else:
            b = np.median(dist) if bw is None else bw
            w = np.exp(-0.5 * (dist / b) ** 2)
        I = np.repeat(np.arange(self.Nn), k)
        W = sp.csr_matrix((w.ravel(), (I, ind.ravel())), shape=(self.Nn, self.Nn))
        return W.maximum(W.T)

    def build(self, w_date=1.0, date_bw=None, k=20, tau=1.0, kernel='inv', w_knn=1.0):
        key = (w_date, date_bw, k, tau, kernel, w_knn)
        if key not in self._cache:
            W = sp.csr_matrix((self.Nn, self.Nn))
            if w_date > 0:
                W = W + w_date * self.date_edges(date_bw)
            if w_knn > 0 and k > 0:
                W = W + w_knn * self.knn_edges(k, tau, kernel)
            W = W.tocsr()
            W.eliminate_zeros()
            self._cache[key] = W
        return self._cache[key]

    # ------------------------------------------------------------------ seeds
    def node_seeds(self, y, seed_rows):
        """Node label mean and labelled mask from the given train rows only."""
        seed_rows = np.asarray(seed_rows)
        nd = self.node[seed_rows]
        c = np.bincount(nd, minlength=self.Nn).astype(float)
        s = np.bincount(nd, weights=y[seed_rows].astype(float), minlength=self.Nn)
        lab = c > 0
        val = np.zeros(self.Nn)
        val[lab] = s[lab] / c[lab]
        return lab, val, c

    def year_prior(self, lab, val, m=10.0):
        g = val[lab].mean()
        cy = np.bincount(self.year[lab], minlength=self.year.max() + 1).astype(float)
        sy = np.bincount(self.year[lab], weights=val[lab], minlength=self.year.max() + 1)
        return ((sy + m * g) / (cy + m))[self.year], g


def harmonic(W, lab, val, prior, m=1.0, n_iter=60):
    deg = np.asarray(W.sum(axis=1)).ravel()
    f = np.where(lab, val, prior)
    for _ in range(n_iter):
        fn = (W @ f + m * prior) / (deg + m)
        f = np.where(lab, val, fn)
    return f


def ppr_ratio(W, lab, val, prior, alpha=0.5, m=1.0, n_iter=40):
    deg = np.asarray(W.sum(axis=1)).ravel()
    P = sp.diags(1.0 / np.maximum(deg, 1e-12)) @ W
    vpos, vlab = lab * val, lab.astype(float)
    # x = sum_{t>=1} alpha^(t-1) P^t v   (exclude t=0: the node itself)
    xp, xl = P @ vpos, P @ vlab
    ap, al_ = xp.copy(), xl.copy()
    for _ in range(n_iter):
        xp, xl = alpha * (P @ xp), alpha * (P @ xl)
        ap += xp
        al_ += xl
    return (ap + m * prior) / (al_ + m), al_


def propagate(G, y, seed_rows, spec):
    """spec: dict(method='harm'|'ppr', graph kwargs, m, alpha). Returns node scores (+ support)."""
    W = G.build(**spec.get('graph', {}))
    lab, val, _ = G.node_seeds(y, seed_rows)
    prior, g = G.year_prior(lab, val)
    if spec.get('prior', 'year') == 'global':
        prior = np.full(G.Nn, g)
    if spec['method'] == 'harm':
        return harmonic(W, lab, val, prior, m=spec.get('m', 1.0)), None
    return ppr_ratio(W, lab, val, prior, alpha=spec.get('alpha', 0.5), m=spec.get('m', 1.0))


def graph_label_features(G, y, fit_idx, apply_idx_list, specs, seed=0, n_inner=5):
    """specs: dict name -> spec. Rows index train+test rows (0..n-1 train, n.. test).
    Fit rows: inner StratifiedGroupKFold by uri (out-of-fold); apply rows: seeds = all fit_idx."""
    fit_idx = np.asarray(fit_idx)
    F_fit = pd.DataFrame(np.nan, index=range(len(fit_idx)), columns=list(specs))
    sgkf = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed)
    for a, b in sgkf.split(fit_idx, y[fit_idx], G.node[fit_idx]):
        for name, s in specs.items():
            f, _ = propagate(G, y, fit_idx[a], s)
            F_fit.iloc[b, F_fit.columns.get_loc(name)] = f[G.node[fit_idx[b]]]
    outs = []
    full = {name: propagate(G, y, fit_idx, s)[0] for name, s in specs.items()}
    for ai in apply_idx_list:
        ai = np.asarray(ai)
        outs.append(pd.DataFrame({name: full[name][G.node[ai]] for name in specs}))
    return F_fit, outs


def dup_mates_features(G, alpha=0.5, n_iter=30, **gkw):
    """LABEL-FREE: duplicate-copy signal of a song's graph neighbours (uri used only to count copies).
    cnt = number of rows of the song in the train+test pool; dup = cnt >= 2 (multi-playlist songs are mostly
    High). Returns (Nn, 3): extra copies among date-mates (own copies excluded), fraction of date-mate songs
    that are duplicated (own song excluded), PPR-weighted duplicated fraction over the combined graph (walks of
    length t >= 1; return walks t >= 2 carry a small share of the song's OWN duplicate status -- label-free
    duplicate-count information, allowed by the rules, and the eligible-row CV mimics its test distribution)."""
    cnt = np.bincount(G.node, minlength=G.Nn).astype(float)
    dup = (cnt >= 2).astype(float)
    U = np.bincount(G.date, minlength=G.date.max() + 1).astype(float)[G.date]
    Sx = np.bincount(G.date, weights=cnt - 1, minlength=G.date.max() + 1)[G.date] - (cnt - 1)
    Sd = np.bincount(G.date, weights=dup, minlength=G.date.max() + 1)[G.date] - dup
    frac = np.where(U > 1, Sd / np.maximum(U - 1, 1), np.nan)
    W = G.build(**gkw)
    deg = np.asarray(W.sum(axis=1)).ravel()
    P = sp.diags(1.0 / np.maximum(deg, 1e-12)) @ W
    x = P @ dup
    acc = x.copy()
    for _ in range(n_iter):
        x = alpha * (P @ x)
        acc += x
    return np.column_stack([Sx, frac, (1 - alpha) * acc])
