"""r3_chx: Charlie's exact final pipeline (Replica logic COPIED from work/ch/chfull_lib.py, unchanged: date TE,
median imputation, CatBoost/ExtraTrees params, residual correction, 0.6/0.4 mix, seeds) PLUS extra inputs appended
to his 115-col 'project' feature matrix. Variant = string of letters:
  a : + 3 out-of-fold label features knnw_yr_40, teh_date_m10, cnt_date (te/label_features.py BEST_SPEC), both
      families. Exactly as work/main/final_cv.py: label_features(XA, y, fit_idx, [apply...], seed) -> fit rows get
      inner out-of-fold values (inner StratifiedGroupKFold(5) grouped by uri), apply rows (validation / test) are
      encoded from ALL fit rows. Outer CV fold k: seed=k (== final_cv.py / zoo cache); full refit seed s: seed=s.
      Inside the residual-correction inner split (fit[it] -> fit[iv]) the label features are REBUILT from fit[it]
      labels only (like Charlie's date TE), so the inner-OOF residuals stay honest.
  b : + 6 label-free copy detectors from build_features_final (a_global_nn_dist, a_nn2_dist, n_same_date_audio,
      a_n_close10/20/30; de-duplicated-pool / global neighbour distances that Charlie's view lacks), both families.
  e : like a, plus the hierarchical out-of-fold album target encoding teh_alb_m10 (album -> date -> year -> global prior;
      album = label-free cluster id; same label_features machinery, fit labels only, same-uri rows never sources).
  c : + label-free album-cluster id (te/label_features get_ctx(tr, te).key['alb']) as a CatBoost NATIVE categorical
      (cat family only; ExtraTrees unchanged). CatBoost's ordered target statistics use the fit rows' labels only.
Everything label-derived uses only the labels of the rows the model is fit on.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
from ch_common import ResidualCorrector
from label_features import label_features, BEST_SPEC

CFG = {
    'cat': dict(rounds=2000, lr=0.008884634139277769, depth=9, l2=9.883292087592952, colsample=0.47172714535307114,
                random_strength=0.6053025553226985, borders=128, duplicate_power=0.0,
                residual_alpha=0.35, residual_bandwidth=0.75, residual_shrinkage=10.0),
    'extra': dict(rounds=900, depth=15, min_leaf=2, colsample=0.1827077108658931, duplicate_power=1.0,
                  residual_alpha=0.15, residual_bandwidth=3.0, residual_shrinkage=3.0),
}
EXTRA_WEIGHT = 0.4
SEED0 = 20261171
LF_COLS = list(BEST_SPEC)                       # knnw_yr_40, teh_date_m10, cnt_date
CD_COLS = ['a_global_nn_dist', 'a_nn2_dist', 'n_same_date_audio', 'a_n_close10', 'a_n_close20', 'a_n_close30']


def make_model(fam, seed, threads, cat_features=None):
    c = CFG[fam]
    if fam == 'cat':
        return CatBoostClassifier(iterations=c['rounds'], learning_rate=c['lr'], depth=c['depth'], l2_leaf_reg=c['l2'],
                                  rsm=c['colsample'], random_strength=c['random_strength'],
                                  border_count=c['borders'], loss_function='Logloss', verbose=False,
                                  allow_writing_files=False, thread_count=threads, random_seed=seed,
                                  cat_features=cat_features)
    return ExtraTreesClassifier(n_estimators=c['rounds'], max_depth=c['depth'], min_samples_leaf=c['min_leaf'],
                                max_features=c['colsample'], n_jobs=threads, random_state=seed)


def dup_weights(uri_sub, power):
    _, inv, cnt = np.unique(np.asarray(uri_sub).astype(str), return_inverse=True, return_counts=True)
    w = cnt[inv].astype(float) ** (-power)
    return w / w.mean()


def date_keys(dates):
    out = []
    for s in dates:
        s = (s if isinstance(s, str) else '').strip()
        year = s[:4] if len(s) >= 4 and s[:4].isdigit() else None
        month = s[:7] if len(s) >= 7 and s[4] == '-' and s[5:7].isdigit() else None
        exact = s if len(s) == 10 and s[4] == '-' and s[7] == '-' else None
        out.append((year, month, exact))
    return out


class GroupedDateTE:
    """Charlie's GroupedDateTargetEncoder(smoothing=2., folds=5, seed=42) (copied from chfull_lib)."""
    def __init__(self, smoothing=2., folds=5, seed=42):
        self.s, self.folds, self.seed = smoothing, folds, seed

    @staticmethod
    def _build(keys, y):
        prior = float(np.mean(y))
        maps = [{}, {}, {}]
        for k, lab in zip(keys, y):
            for j, key in enumerate(k):
                if key is not None:
                    b = maps[j].setdefault(key, [0., 0])
                    b[0] += int(lab)
                    b[1] += 1
        return prior, maps

    def _encode(self, keys, state):
        prior, maps = state
        out = np.empty((len(keys), 6))
        for i, k in enumerate(keys):
            for j, key in enumerate(k):
                tot, cnt = maps[j].get(key, (0., 0)) if key is not None else (0., 0)
                out[i, j] = (tot + self.s * prior) / (cnt + self.s)
                out[i, j + 3] = np.log1p(cnt)
        return out

    def fit_transform(self, keys, y, groups):
        y = np.asarray(y)
        out = np.full((len(keys), 6), np.nan)
        for a, b in StratifiedGroupKFold(self.folds, shuffle=True, random_state=self.seed).split(
                np.arange(len(keys)), y, groups):
            assert not set(groups[a]) & set(groups[b])
            out[b] = self._encode([keys[i] for i in b], self._build([keys[i] for i in a], y[a]))
        assert np.isfinite(out).all()
        self.state = self._build(keys, y)
        return out

    def transform(self, keys):
        return self._encode(keys, self.state)


class ReplicaX:
    """Pool row convention: train rows 0..n-1, test rows n..n+m-1 (apply index arrays are train indices;
    the test matrix is always all test rows)."""

    def __init__(self, tr, te, y, Fproj, groups, variant, threads=3, XA=None, CD=None, alb=None):
        self.tr, self.te, self.y, self.groups, self.threads = tr, te, y, np.asarray(groups), threads
        n, m = len(tr), len(te)
        self.n, self.m = n, m
        self.test_idx = np.arange(n, n + m)
        self.Ptr = Fproj.iloc[:n].reset_index(drop=True).astype(np.float32).to_numpy()
        self.Pte = Fproj.iloc[n:].reset_index(drop=True).astype(np.float32).to_numpy()
        self.ktr = date_keys(tr['track_album_release_date'].tolist())
        self.kte = date_keys(te['track_album_release_date'].tolist())
        self.uri = tr['uri'].astype(str).to_numpy()
        self.variant = variant
        self.use_lf = 'a' in variant or 'e' in variant
        self.lf_spec = tuple(BEST_SPEC) + (('teh_alb_m10',) if 'e' in variant else ())
        self.lf_cols = list(self.lf_spec)
        self.XA = XA
        self.CD = None
        if 'b' in variant:
            CD = np.asarray(CD, float)
            self.CD = CD
        self.alb = None
        if 'c' in variant:
            self.alb = np.asarray(alb).astype(str)       # pool-level label-free album ids
        assert not self.use_lf or XA is not None

    # ----------------------------------------------------------------------------------------- matrices
    def matrices(self, fit, applies, with_test, lf_seed):
        """Charlie: project features + fold date TE (fit rows cross-fitted) [+ extras] + median imputation fitted
        on the fit part. Extras appended AFTER Charlie's columns (b: label-free; a: label features from fit labels)."""
        enc = GroupedDateTE()
        Efit = enc.fit_transform([self.ktr[i] for i in fit], self.y[fit], self.groups[fit])
        pf = [self.Ptr[fit], Efit]
        pa = [[self.Ptr[b], enc.transform([self.ktr[i] for i in b])] for b in applies]
        pt = [self.Pte, enc.transform(self.kte)] if with_test else None
        if self.CD is not None:
            pf.append(self.CD[fit])
            for b, p in zip(applies, pa):
                p.append(self.CD[b])
            if with_test:
                pt.append(self.CD[self.test_idx])
        if self.use_lf:
            app = [np.asarray(b) for b in applies] + ([self.test_idx] if with_test else [])
            F_fit, F_app = label_features(self.XA, self.y, np.asarray(fit), app, self.tr, self.te, seed=lf_seed,
                                          spec=self.lf_spec)
            pf.append(F_fit[self.lf_cols].to_numpy(float))
            for p, Fa in zip(pa, F_app[:len(applies)]):
                p.append(Fa[self.lf_cols].to_numpy(float))
            if with_test:
                pt.append(F_app[-1][self.lf_cols].to_numpy(float))
        Xf = np.column_stack(pf)
        Xa = [np.column_stack(p) for p in pa]
        Xt = np.column_stack(pt) if with_test else None
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xf)
        return imp.transform(Xf), [imp.transform(x) for x in Xa], (imp.transform(Xt) if with_test else None)

    def inner_sets(self, fit, S):
        """Inner 3-fold split (shared by both families; seed S+100 as in charlie) + its matrices; label features of
        the inner split rebuilt from the inner fit rows only (seed S+300+j)."""
        out = []
        for j, (it, iv) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=S + 100).split(
                fit, self.y[fit], self.groups[fit])):
            I, (J,), _ = self.matrices(fit[it], [fit[iv]], False, lf_seed=S + 300 + j)
            out.append((it, iv, I, J))
        return out

    # ----------------------------------------------------------------------------------------- models
    def _frame(self, fam, X, pool_rows):
        """CatBoost + variant c: numeric matrix + album id (string) as a native categorical column."""
        if fam != 'cat' or self.alb is None:
            return X
        D = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        D['c_alb'] = self.alb[np.asarray(pool_rows)]
        return D

    def _model(self, fam, seed):
        cf = ['c_alb'] if (fam == 'cat' and self.alb is not None) else None
        return make_model(fam, seed, self.threads, cat_features=cf)

    def run_family(self, fam, fit, Xf, Xa_list, a_rows, Xt, inner, S):
        """a_rows: train indices of each apply matrix (for the album id)."""
        c = CFG[fam]
        w = dup_weights(self.uri[fit], c['duplicate_power'])
        m = self._model(fam, S).fit(self._frame(fam, Xf, fit), self.y[fit], sample_weight=w)
        raw_a = [m.predict_proba(self._frame(fam, X, r))[:, 1] for X, r in zip(Xa_list, a_rows)]
        raw_t = m.predict_proba(self._frame(fam, Xt, self.test_idx))[:, 1] if Xt is not None else None
        ip = np.full(len(fit), np.nan)
        for it, iv, I, J in inner:
            mi = self._model(fam, S + 200).fit(
                self._frame(fam, I, fit[it]), self.y[fit[it]],
                sample_weight=dup_weights(self.uri[fit[it]], c['duplicate_power']))
            ip[iv] = mi.predict_proba(self._frame(fam, J, fit[iv]))[:, 1]
        assert np.isfinite(ip).all()
        return raw_a, raw_t, ip

    def correct(self, fam, fit, inner_pred, raw, q_df):
        c = CFG[fam]
        rc = ResidualCorrector(self.tr.iloc[fit], self.y[fit] - inner_pred)
        corr = ResidualCorrector.correction(rc.prepare(q_df), c['residual_bandwidth'], c['residual_shrinkage'])
        return np.clip(raw + c['residual_alpha'] * corr, 0, 1)
