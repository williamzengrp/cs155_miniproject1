"""r3_chg: Charlie's final model (faithful replica: work/ch/chfull_lib.py + ch_common.py + ch_feats.py, copied here
unchanged) with NEW feature columns appended to his matrix. Everything else identical: 115 label-free 'project'
columns + his grouped date target encodings (fit rows cross-fitted), median imputation fitted on the fit part,
CatBoost / ExtraTrees with his params and seeds (S = SEED0 + k in CV, SEED0 + 1000 (s+1) for the full refit),
residual correction from an inner 3-fold (models S+200), blend 0.6 cat + 0.4 et.

Feature groups (GROUPS = '+'-joined names; 'none' = plain replica):
  hc2   : graph label propagation g_hc2 (r2_graph/graph_lib, copied). LABEL-DEPENDENT: seeds = labels of the
          fit rows only; fit rows get inner StratifiedGroupKFold(5, groups = uri) out-of-fold values; apply /
          test rows are unlabelled nodes. Rebuilt inside every residual-correction inner split from that split's
          training rows only (exactly like Charlie's date encodings are rebuilt there).
  hk2   : same with the kNN-only graph (g_hk2)
  dupm  : label-free duplicate structure of graph neighbours / date-mates (3 cols, train+test pool)
  spec8 : label-free 8-dim diffusion-map embedding of the song graph (8 cols, train+test pool)
  dom   : label-free domain features (whole-second duration flag, minutes, distance to 3:30, mood interactions,
          post-2015 Friday flag)
  drank : label-free within-release-date percentile rank of 10 audio features

usage:
  python -I chg.py cv TAG GROUPS K0 K1 THREADS {raw|full}   folds K0..K1-1 of H.make_splits(n_rep=5, seed0=1000)
                                                           raw = outer fits only; full = + inner fits/correction
                                                           (reuses cached raw outer predictions when present)
  python -I chg.py fullfit TAG GROUPS SEEDS THREADS         refit on all train rows, SEEDS e.g. 0,1,2
  python -I chg.py report TAG                              paired deltas vs work/ch/cache/chfull_std_k*.npz
  python -I chg.py save TAG NAME                           -> H.save(NAME, ...) (repeats w/o all 5 folds: NaN)
"""
import os
import sys
import time
import warnings

THREADS = int(sys.argv[6]) if sys.argv[1] == 'cv' else (int(sys.argv[5]) if sys.argv[1] == 'fullfit' else 1)
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = str(THREADS)
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

WORK = './work'
HERE = WORK + '/r3_chg'
sys.path.insert(0, WORK)
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
from ch_common import connected_groups  # noqa: E402
from ch_feats import charlie_pool  # noqa: E402
from chfull_lib import Replica, GroupedDateTE, CFG, make_model, dup_weights, SEED0, EXTRA_WEIGHT  # noqa: E402
from graph_lib import SongGraph, graph_label_features, dup_mates_features  # noqa: E402
from spectral import spectral_embedding  # noqa: E402
from graph_specs import GSPECS  # noqa: E402

warnings.filterwarnings('ignore')
CACHE = HERE + '/cache'
REPCACHE = WORK + '/ch/cache'
GKW = dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv')
AUD10 = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'speechiness', 'instrumentalness',
         'tempo', 'duration_ms', 'liveness']
LABEL_GROUPS = {'hc2': 'g_hc2', 'hk2': 'g_hk2'}


def lf_block(name, tr, te, G):
    """Label-free block over the train+test pool rows (train first). Uses only feature values (+ uri to count
    copies for dupm, as in r2_graph)."""
    al = pd.concat([tr, te], ignore_index=True)
    if name == 'dupm':
        a = dup_mates_features(G, **GKW)
        return pd.DataFrame(a[G.node], columns=[f'dupm_{j}' for j in range(a.shape[1])])
    if name == 'spec8':
        a = spectral_embedding(G, 8, **GKW)[0]
        return pd.DataFrame(a[G.node], columns=[f'spec8_{j}' for j in range(a.shape[1])])
    raw = al['track_album_release_date'].fillna('').astype(str).str.strip()
    full = raw.str.fullmatch(r'\d{4}-\d{2}-\d{2}').to_numpy()
    if name == 'dom':
        dur = al['duration_ms'].astype(float).to_numpy()
        e, v = al['energy'].astype(float).to_numpy(), al['valence'].astype(float).to_numpy()
        da, ac = al['danceability'].astype(float).to_numpy(), al['acousticness'].astype(float).to_numpy()
        d = pd.to_datetime(raw.where(full, None), format='%Y-%m-%d', errors='coerce')
        year = pd.to_numeric(raw.str[:4], errors='coerce').to_numpy(float)
        wd = d.dt.dayofweek.to_numpy(float)
        D = dict(
            dur_whole_sec=np.where(np.isfinite(dur), (np.mod(dur, 1000) == 0).astype(float), np.nan),
            dur_min=dur / 60000.,
            dur_dist_3m30=np.abs(np.log(np.maximum(dur, 1.) / 210000.)),
            energy_x_valence=e * v,
            dance_x_valence=da * v,
            acoustic_x_lowenergy=ac * (1 - e),
            friday_post2015=np.where(np.isfinite(wd), ((wd == 4) & (year >= 2015)).astype(float), np.nan),
        )
        return pd.DataFrame(D)
    if name == 'drank':
        key = raw.where(full, '')
        A = al[AUD10].astype(float)
        out = {}
        cnt = key.map(key.value_counts()).to_numpy(float)
        for c in AUD10:
            r = A[c].groupby(key).rank(method='average').to_numpy(float)
            pr = np.where(cnt > 1, (r - 1) / np.maximum(cnt - 1, 1), 0.5)
            out[f'drank_{c}'] = np.where(full & np.isfinite(A[c].to_numpy()), pr, np.nan)
        return pd.DataFrame(out)
    raise ValueError(name)


class RX(Replica):
    """Charlie replica + extra label-free columns (LF) + graph label features (GS) appended to his matrix."""

    def __init__(self, tr, te, y, Fproj, groups, LF, G, GS, threads):
        super().__init__(tr, te, y, Fproj, groups, threads=threads)
        n = len(tr)
        self.LFtr = None if LF is None else LF[:n]
        self.LFte = None if LF is None else LF[n:]
        self.G, self.GS = G, GS
        self.test_idx = np.arange(n, n + len(te))

    def matrices(self, fit, applies, with_test, gseed=0):
        fit = np.asarray(fit)
        enc = GroupedDateTE()
        Efit = enc.fit_transform([self.ktr[i] for i in fit], self.y[fit], self.groups[fit])
        Xf = [self.Ptr[fit], Efit]
        Xa = [[self.Ptr[b], enc.transform([self.ktr[i] for i in b])] for b in applies]
        Xt = [self.Pte, enc.transform(self.kte)] if with_test else None
        if self.LFtr is not None:
            Xf.append(self.LFtr[fit])
            for x, b in zip(Xa, applies):
                x.append(self.LFtr[b])
            if with_test:
                Xt.append(self.LFte)
        if self.GS:
            alist = [np.asarray(b) for b in applies] + ([self.test_idx] if with_test else [])
            Gf, Ga = graph_label_features(self.G, self.y, fit, alist, self.GS, seed=gseed)
            Xf.append(Gf.to_numpy(float))
            for x, g in zip(Xa, Ga[:len(applies)]):
                x.append(g.to_numpy(float))
            if with_test:
                Xt.append(Ga[-1].to_numpy(float))
        Xf = np.column_stack(Xf)
        Xa = [np.column_stack(x) for x in Xa]
        Xt = np.column_stack(Xt) if with_test else None
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(Xf)
        return imp.transform(Xf), [imp.transform(x) for x in Xa], (imp.transform(Xt) if with_test else None)

    def inner_sets(self, fit, S, gseed=0):
        out = []
        for j, (it, iv) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=S + 100).split(
                fit, self.y[fit], self.groups[fit])):
            I, (J,), _ = self.matrices(fit[it], [fit[iv]], False, gseed=gseed + 1000 * (j + 1))
            out.append((it, iv, I, J))
        return out


def fam_run(R, fam, fit, Xf, Xa_list, Xt, inner, S, outer=True):
    """= Replica.run_family, with the outer fit and the inner (residual) fits separable."""
    c = CFG[fam]
    raw_a, raw_t, ip = None, None, None
    if outer:
        w = dup_weights(R.uri[fit], c['duplicate_power'])
        m = make_model(fam, S, R.threads).fit(Xf, R.y[fit], sample_weight=w)
        raw_a = [m.predict_proba(X)[:, 1] for X in Xa_list]
        raw_t = m.predict_proba(Xt)[:, 1] if Xt is not None else None
    if inner is not None:
        ip = np.full(len(fit), np.nan)
        for it, iv, I, J in inner:
            mi = make_model(fam, S + 200, R.threads).fit(
                I, R.y[fit[it]], sample_weight=dup_weights(R.uri[fit[it]], c['duplicate_power']))
            ip[iv] = mi.predict_proba(J)[:, 1]
        assert np.isfinite(ip).all()
    return raw_a, raw_t, ip


def build(tr, te, y, groups_str, threads):
    gl = [] if groups_str == 'none' else groups_str.split('+')
    G = SongGraph(tr, te)
    lf = [g for g in gl if g not in LABEL_GROUPS]
    LF = None
    if lf:
        LFd = pd.concat([lf_block(g, tr, te, G) for g in lf], axis=1)
        print('label-free cols:', list(LFd.columns), flush=True)
        LF = LFd.to_numpy(np.float32)
    GS = {LABEL_GROUPS[g]: GSPECS[LABEL_GROUPS[g]] for g in gl if g in LABEL_GROUPS}
    F, BL = charlie_pool(tr, te)
    return RX(tr, te, y, F[BL['PROJECT']], connected_groups(tr), LF, G, GS, threads)


def blend(c, kind):
    if kind == 'raw':
        return (1 - EXTRA_WEIGHT) * c['cat_raw_v'] + EXTRA_WEIGHT * c['et_raw_v']
    return (1 - EXTRA_WEIGHT) * c['cat_v'] + EXTRA_WEIGHT * c['et_v']


def load_tag(tag, k):
    p_full, p_raw = f'{CACHE}/{tag}_k{k}.npz', f'{CACHE}/{tag}_raw_k{k}.npz'
    if os.path.exists(p_full):
        return dict(np.load(p_full)), 'full'
    if os.path.exists(p_raw):
        return dict(np.load(p_raw)), 'raw'
    return None, None


if __name__ == '__main__':
    mode = sys.argv[1]
    tr, te, y = H.load()
    n = len(tr)
    splits = H.make_splits(tr, y, n_rep=5)

    if mode == 'cv':
        tag, groups_str, k0, k1, kind = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[7]
        R = build(tr, te, y, groups_str, THREADS)
        for k in range(k0, min(k1, len(splits))):
            p_full, p_raw = f'{CACHE}/{tag}_k{k}.npz', f'{CACHE}/{tag}_raw_k{k}.npz'
            if os.path.exists(p_full) or (kind == 'raw' and os.path.exists(p_raw)):
                continue
            r, fit, val, elig = splits[k]
            S = SEED0 + k
            t0 = time.time()
            Xf, (Xv,), Xt = R.matrices(fit, [val], True, gseed=k)
            inner = R.inner_sets(fit, S, gseed=k) if kind == 'full' else None
            prev = dict(np.load(p_raw)) if os.path.exists(p_raw) else None
            out, tim = {}, {'prep': time.time() - t0, 'ncol': Xf.shape[1]}
            for fam, ft in (('cat', 'cat'), ('extra', 'et')):
                ta = time.time()
                if prev is not None:
                    _, _, ip = fam_run(R, fam, fit, Xf, [Xv], Xt, inner, S, outer=False)
                    rv, rt = prev[f'{ft}_raw_v'], prev[f'{ft}_raw_t']
                else:
                    (rv,), rt, ip = fam_run(R, fam, fit, Xf, [Xv], Xt, inner, S, outer=True)
                out[f'{ft}_raw_v'], out[f'{ft}_raw_t'] = rv, rt
                if kind == 'full':
                    out[f'{ft}_ip'] = ip
                    out[f'{ft}_v'] = R.correct(fam, fit, ip, rv, tr.iloc[val])
                    out[f'{ft}_t'] = R.correct(fam, fit, ip, rt, te)
                tim[ft] = time.time() - ta
            np.savez(p_full if kind == 'full' else p_raw, **out)
            print(f'[{tag} {kind} k={k}] ncol {tim["ncol"]} prep {tim["prep"]:.1f}s cat {tim["cat"]:.1f}s '
                  f'et {tim["et"]:.1f}s total {time.time() - t0:.1f}s', flush=True)

    elif mode == 'fullfit':
        tag, groups_str, seeds = sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(',')]
        R = build(tr, te, y, groups_str, THREADS)
        fit = np.arange(n)
        for s in seeds:
            path = f'{CACHE}/{tag}_full_s{s}.npz'
            if os.path.exists(path):
                continue
            S = SEED0 + 1000 * (s + 1)
            t0 = time.time()
            Xf, _, Xt = R.matrices(fit, [], True, gseed=s)
            inner = R.inner_sets(fit, S, gseed=s)
            out = {}
            for fam, ft in (('cat', 'cat'), ('extra', 'et')):
                _, rt, ip = fam_run(R, fam, fit, Xf, [], Xt, inner, S)
                out[f'{ft}_raw_t'], out[f'{ft}_ip'] = rt, ip
                out[f'{ft}_t'] = R.correct(fam, fit, ip, rt, te)
            np.savez(path, **out)
            print(f'[{tag} fullfit seed {s}] {time.time() - t0:.1f}s', flush=True)

    elif mode == 'report':
        tags = sys.argv[2:]
        for tag in tags:
            for r in range(5):
                ks = range(5 * r, 5 * r + 5)
                C = [load_tag(tag, k) for k in ks]
                if any(c is None for c, _ in C):
                    continue
                kinds = ['raw'] + (['full'] if all(kd == 'full' for _, kd in C) else [])
                for kind in kinds:
                    rows = []
                    for nm, f in (('blend', lambda c: blend(c, kind)),
                                  ('cat', lambda c: c['cat_raw_v' if kind == 'raw' else 'cat_v']),
                                  ('et', lambda c: c['et_raw_v' if kind == 'raw' else 'et_v'])):
                        o, b = np.full(n, np.nan), np.full(n, np.nan)
                        for k, (c, _) in zip(ks, C):
                            _, fit, val, elig = splits[k]
                            o[val[elig]] = f(c)[elig]
                            b[val[elig]] = f(dict(np.load(f'{REPCACHE}/chfull_std_k{k}.npz')))[elig]
                        m = ~np.isnan(o)
                        a1, a0 = roc_auc_score(y[m], o[m]), roc_auc_score(y[m], b[m])
                        rows.append(f'{nm} {a1:.5f} ({a1 - a0:+.5f})')
                    print(f'{tag:28s} rep{r} {kind:4s}: ' + '  '.join(rows), flush=True)

    elif mode == 'save':
        tag, name = sys.argv[2], sys.argv[3]
        oof = np.full((5, n), np.nan)
        aucs = np.full(5, np.nan)
        tfa, nt = 0, 0
        for r in range(5):
            ks = range(5 * r, 5 * r + 5)
            C = [load_tag(tag, k) for k in ks]
            if any(kd != 'full' for _, kd in C):
                continue
            for k, (c, _) in zip(ks, C):
                _, fit, val, elig = splits[k]
                oof[r, val[elig]] = blend(c, 'full')[elig]
                tfa = tfa + (1 - EXTRA_WEIGHT) * c['cat_t'] + EXTRA_WEIGHT * c['et_t']
                nt += 1
            m = ~np.isnan(oof[r])
            aucs[r] = roc_auc_score(y[m], oof[r, m])
        ref = H.load_res('chrep_bl')
        done = ~np.isnan(aucs)
        for r in np.flatnonzero(done):
            assert (np.isnan(oof[r]) == np.isnan(ref['oof'][r])).all()
        fs = sorted(f for f in os.listdir(CACHE) if f.startswith(f'{tag}_full_s'))
        Fl = [dict(np.load(f'{CACHE}/{f}')) for f in fs]
        full = np.mean([(1 - EXTRA_WEIGHT) * f['cat_t'] + EXTRA_WEIGHT * f['et_t'] for f in Fl], axis=0) if Fl else None
        res = dict(oof=oof, aucs=aucs, auc=float(np.nanmean(aucs)), test_foldavg=tfa / max(nt, 1))
        H.save(name, res, full)
        d = aucs[done] - ref['aucs'][done]
        print(f'saved {name}: repeats {np.flatnonzero(done).tolist()} AUC {np.nanmean(aucs):.5f} per-rep '
              f'{np.round(aucs, 4)}  vs chrep_bl {d.mean():+.5f} per-rep {np.round(d, 4)}  full seeds {fs}', flush=True)
        if full is not None:
            from scipy.stats import spearmanr
            print('spearman(test_full, chrep_bl test_full) =', round(spearmanr(full, ref['test_full'])[0], 4),
                  ' spearman(test_full, test_foldavg) =', round(spearmanr(full, res['test_foldavg'])[0], 4))
