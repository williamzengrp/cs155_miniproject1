"""ch agent shared helpers (own re-implementations of charlie-branch ideas; no charlie code is imported)."""
import re
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

AUDIO = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms', 'speechiness', 'acousticness',
         'instrumentalness', 'liveness', 'valence', 'key', 'mode', 'time_signature']
FULLDATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def connected_groups(df):
    """Rows joined if they share a uri OR have an identical complete audio vector (charlie's CV groups)."""
    n = len(df)
    ukey = pd.factorize(df['uri'].astype(str))[0]
    A = df[AUDIO].to_numpy(float)
    full = np.isfinite(A).all(1)
    akey = pd.Series([('|'.join(map(repr, a)) if f else f'na{i}') for i, (a, f) in enumerate(zip(A, full))])
    acode = pd.factorize(akey)[0] + ukey.max() + 1
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[ukey, acode]
    G = coo_matrix((np.ones(2 * n), (rows, cols)), shape=(n, acode.max() + 1)).tocsr()
    B = (G @ G.T)
    return connected_components(B, directed=False)[1]


class ResidualCorrector:
    """Fit on (fit rows, their inner-OOF residuals); correct query rows with same-date audio-kernel residuals."""

    def __init__(self, fit_df, resid):
        d = fit_df[['uri', 'track_album_release_date'] + AUDIO].copy()
        d['_r'] = resid
        agg = {c: 'first' for c in AUDIO + ['track_album_release_date']}
        agg['_r'] = 'mean'
        ref = d.groupby('uri', sort=False).agg(agg).reset_index()
        num = ref[AUDIO].astype(float)
        self.med = num.median()
        R = num.fillna(self.med).to_numpy()
        self.mu, self.sd = R.mean(0), R.std(0)
        self.sd[self.sd == 0] = 1.0
        self.R = (R - self.mu) / self.sd
        self.r = ref['_r'].to_numpy()
        self.uri = ref['uri'].to_numpy()
        self.by_date = ref.groupby(ref['track_album_release_date'].astype(str)).indices

    def prepare(self, q_df):
        """Pre-compute per query row the same-date reference distances (list of (D row, r) per row)."""
        Q = (q_df[AUDIO].astype(float).fillna(self.med).to_numpy() - self.mu) / self.sd
        valid = q_df[AUDIO].notna().any(axis=1).to_numpy()
        dates = q_df['track_album_release_date'].astype(str).to_numpy()
        quri = q_df['uri'].astype(str).to_numpy()
        items = [None] * len(q_df)
        for i in range(len(q_df)):
            if not valid[i] or not FULLDATE.match(dates[i]):
                continue
            ri = self.by_date.get(dates[i])
            if ri is None or len(ri) == 0:
                continue
            ri = ri[self.uri[ri] != quri[i]]          # never use the row's own song (no effect on scored rows)
            if len(ri) == 0:
                continue
            D = cdist(Q[i:i + 1], self.R[ri])[0]
            items[i] = (D, self.r[ri])
        return items

    @staticmethod
    def correction(items, bw, shrink):
        out = np.zeros(len(items))
        for i, it in enumerate(items):
            if it is None:
                continue
            D, r = it
            W = np.exp(-0.5 * (D / bw) ** 2)
            out[i] = (W @ r) / (shrink + W.sum())
        return out
