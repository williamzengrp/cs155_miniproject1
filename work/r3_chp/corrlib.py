"""r3_chp: fast, exact re-implementation of ch_common.ResidualCorrector that separates the geometry (same-date
reference pairs + standardized audio distances, computed once per (fit set, query set)) from the residual vector
and the (alpha, bandwidth, shrinkage) parameters, so the correction can be re-tuned without refits."""
import sys
sys.dont_write_bytecode = True
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

AUDIO = ['danceability', 'energy', 'loudness', 'tempo', 'duration_ms', 'speechiness', 'acousticness',
         'instrumentalness', 'liveness', 'valence', 'key', 'mode', 'time_signature']
FULLDATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


class Geo:
    """Reference = fit rows grouped by uri (first features, mean residual), exactly as ResidualCorrector."""

    def __init__(self, fit_df):
        d = fit_df[['uri', 'track_album_release_date'] + AUDIO].copy().reset_index(drop=True)
        self.codes, uniq = pd.factorize(d['uri'].astype(str), sort=False)
        self.nref = len(uniq)
        agg = {c: 'first' for c in AUDIO + ['track_album_release_date']}
        ref = d.groupby('uri', sort=False).agg(agg).reset_index()
        assert (ref['uri'].astype(str).to_numpy() == np.asarray(uniq)).all()
        num = ref[AUDIO].astype(float)
        self.med = num.median()
        Rm = num.fillna(self.med).to_numpy()
        self.mu, self.sd = Rm.mean(0), Rm.std(0)
        self.sd[self.sd == 0] = 1.0
        self.R = (Rm - self.mu) / self.sd
        self.uri = ref['uri'].to_numpy()
        self.by_date = ref.groupby(ref['track_album_release_date'].astype(str)).indices

    def ref_resid(self, resid):
        """Per-uri mean residual (resid over fit rows in fit_df order)."""
        cnt = np.bincount(self.codes, minlength=self.nref)
        return np.bincount(self.codes, weights=resid, minlength=self.nref) / cnt

    def pairs(self, q_df):
        """-> (qi, ri, D) flattened pairs for query rows."""
        Q = (q_df[AUDIO].astype(float).fillna(self.med).to_numpy() - self.mu) / self.sd
        valid = q_df[AUDIO].notna().any(axis=1).to_numpy()
        dates = q_df['track_album_release_date'].astype(str).to_numpy()
        quri = q_df['uri'].astype(str).to_numpy()
        QI, RI, DD = [], [], []
        for i in range(len(q_df)):
            if not valid[i] or not FULLDATE.match(dates[i]):
                continue
            ri = self.by_date.get(dates[i])
            if ri is None or len(ri) == 0:
                continue
            ri = ri[self.uri[ri] != quri[i]]
            if len(ri) == 0:
                continue
            D = cdist(Q[i:i + 1], self.R[ri])[0]
            QI.append(np.full(len(ri), i)); RI.append(ri); DD.append(D)
        if not QI:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0), len(q_df)
        return np.concatenate(QI), np.concatenate(RI), np.concatenate(DD), len(q_df)


def correction(pairs, rref, bw, shrink):
    qi, ri, D, nq = pairs
    W = np.exp(-0.5 * (D / bw) ** 2)
    num = np.bincount(qi, weights=W * rref[ri], minlength=nq)
    den = np.bincount(qi, weights=W, minlength=nq)
    out = np.zeros(nq)
    has = np.bincount(qi, minlength=nq) > 0
    out[has] = num[has] / (shrink + den[has])
    return out
