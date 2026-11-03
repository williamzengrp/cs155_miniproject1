"""ch agent: faithful re-implementation of Charlie's FINAL model (charlie branch, reproduce.py + tune_project.cv_run
+ research85/target_encode.py + reference/model_config.json). No charlie code is imported or executed.

Per outer fold (fit rows a, apply rows b, test T):
  1. features = charlie 'project' view (115 label-free cols on the train+test pool, ch_feats.charlie_pool), float32
  2. + grouped date target encodings (year / year-month / exact date, smoothing 2, prior = fit-row mean) and
     log1p counts: fit rows cross-fitted with StratifiedGroupKFold(5, shuffle, rs=42) over connected groups
     (uri + identical audio vector); apply/test rows from all fit rows. Fit-row labels only.
  3. median imputation fitted on the fit part
  4. model (seed S = 20261171 + k as in charlie's reproduce.py):
       cat  : CatBoost 2000 it, lr 0.008884634139277769, depth 9, l2 9.883292087592952, rsm 0.47172714535307114,
              random_strength 0.6053025553226985, border_count 128, Logloss; weights = 1 (duplicate_power 0)
       extra: ExtraTrees 900, max_depth 15, min_samples_leaf 2, max_features 0.1827077108658931;
              weights = 1/count per uri in the fit part, normalised to mean 1 (duplicate_power 1)
  5. residual correction (part of charlie's final submission): inner StratifiedGroupKFold(3, shuffle, rs=S+100)
     over the fit rows (groups as above); inner models (seed S+200) trained with encodings rebuilt inside the
     inner split -> inner-OOF residual y - p; same-date Gaussian audio kernel; p += alpha * sum W r/(shrink+sum W),
     clipped. cat: alpha .35, bw .75, shrink 10; extra: alpha .15, bw 3, shrink 3.
  6. blend = 0.6 * cat + 0.4 * extra (probabilities).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
from ch_common import ResidualCorrector

CFG = {
    'cat': dict(rounds=2000, lr=0.008884634139277769, depth=9, l2=9.883292087592952, colsample=0.47172714535307114,
                random_strength=0.6053025553226985, borders=128, duplicate_power=0.0,
                residual_alpha=0.35, residual_bandwidth=0.75, residual_shrinkage=10.0),
    'extra': dict(rounds=900, depth=15, min_leaf=2, colsample=0.1827077108658931, duplicate_power=1.0,
                  residual_alpha=0.15, residual_bandwidth=3.0, residual_shrinkage=3.0),
}
EXTRA_WEIGHT = 0.4
SEED0 = 20261171


def make_model(fam, seed, threads):
    c = CFG[fam]
    if fam == 'cat':
        return CatBoostClassifier(iterations=c['rounds'], learning_rate=c['lr'], depth=c['depth'], l2_leaf_reg=c['l2'],
                                  rsm=c['colsample'], random_strength=c['random_strength'],
                                  border_count=c['borders'], loss_function='Logloss', verbose=False,
                                  allow_writing_files=False, thread_count=threads, random_seed=seed)
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
    """Charlie's GroupedDateTargetEncoder(smoothing=2., folds=5, seed=42), re-implemented."""
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


class Replica:
    def __init__(self, tr, te, y, Fproj, groups, threads=4):
        self.tr, self.te, self.y, self.groups, self.threads = tr, te, y, np.asarray(groups), threads
        n = len(tr)
        self.Ptr = Fproj.iloc[:n].reset_index(drop=True).astype(np.float32).to_numpy()
        self.Pte = Fproj.iloc[n:].reset_index(drop=True).astype(np.float32).to_numpy()
        self.ktr = date_keys(tr['track_album_release_date'].tolist())
        self.kte = date_keys(te['track_album_release_date'].tolist())
        self.uri = tr['uri'].astype(str).to_numpy()

    def matrices(self, fit, applies, with_test):
        """Project features + fold date TE (fit rows cross-fitted) + median imputation fitted on fit rows."""
        enc = GroupedDateTE()
        Efit = enc.fit_transform([self.ktr[i] for i in fit], self.y[fit], self.groups[fit])
        Xf = np.column_stack([self.Ptr[fit], Efit])
        Xa = [np.column_stack([self.Ptr[b], enc.transform([self.ktr[i] for i in b])]) for b in applies]
        Xt = np.column_stack([self.Pte, enc.transform(self.kte)]) if with_test else None
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xf)
        return imp.transform(Xf), [imp.transform(x) for x in Xa], (imp.transform(Xt) if with_test else None)

    def inner_sets(self, fit, S):
        """Inner 3-fold split (shared by both families, as in charlie: same seed S+100) + its matrices."""
        out = []
        for it, iv in StratifiedGroupKFold(3, shuffle=True, random_state=S + 100).split(
                fit, self.y[fit], self.groups[fit]):
            I, (J,), _ = self.matrices(fit[it], [fit[iv]], False)
            out.append((it, iv, I, J))
        return out

    def run_family(self, fam, fit, Xf, Xa_list, Xt, inner, S):
        c = CFG[fam]
        w = dup_weights(self.uri[fit], c['duplicate_power'])
        m = make_model(fam, S, self.threads).fit(Xf, self.y[fit], sample_weight=w)
        raw_a = [m.predict_proba(X)[:, 1] for X in Xa_list]
        raw_t = m.predict_proba(Xt)[:, 1] if Xt is not None else None
        ip = np.full(len(fit), np.nan)
        for it, iv, I, J in inner:
            mi = make_model(fam, S + 200, self.threads).fit(
                I, self.y[fit[it]], sample_weight=dup_weights(self.uri[fit[it]], c['duplicate_power']))
            ip[iv] = mi.predict_proba(J)[:, 1]
        assert np.isfinite(ip).all()
        return raw_a, raw_t, ip

    def correct(self, fam, fit, inner_pred, raw, q_df):
        c = CFG[fam]
        rc = ResidualCorrector(self.tr.iloc[fit], self.y[fit] - inner_pred)
        corr = ResidualCorrector.correction(rc.prepare(q_df), c['residual_bandwidth'], c['residual_shrinkage'])
        return np.clip(raw + c['residual_alpha'] * corr, 0, 1)
