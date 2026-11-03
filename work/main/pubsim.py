"""Public-leaderboard simulator. Random public-sized subsets of the eligible validation rows (std OOF, 5 repeats)
stand in for the public test rows. The AUC differences of every file vs Charlie's model are treated as jointly
Gaussian over subsets; candidates are conditioned on the public scores already observed:
  f8 0.86555  f7 0.86619  v2 0.86560  h25 0.86769  ch 0.86886
(final9 = f8 + TabPFN and T2 contain TabPFN: foundation models are NOT allowed, so neither TabPFN predictions nor
those two public scores are used anywhere.)
-> conditional mean / sd of (candidate - ch) on the public rows and P(candidate public >= TARGET).

python -I pubsim.py NPUB [cand ...]      cand = "name=tok:w,tok:w,..."  tok = ch | v2 | f8 | <result name>
Base vectors are per-repeat rank-normalised OOF predictions; a candidate is their weighted average (weights
normalised), matching how the submission files are built (rank averages)."""
import sys
import numpy as np
from scipy.stats import rankdata, norm
sys.path.insert(0, './work')
import harness as H

TARGET = 0.870
OBS = {'f8': 0.86555, 'f7': 0.86619, 'v2': 0.86560, 'h25': 0.86769}
CH_PUB = 0.86886
V2 = ['graph_ds_lgbm', 'graph_ds_xgb', 'zoo_cat_alb', 'fkd_et', 'b2_et', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
F8 = ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj']
NSUB = 3000   # subsets per repeat


def rk(v):
    return rankdata(v) / len(v)


def parse(c):
    name, spec = c.split('=')
    return name, [(t.split(':')[0], float(t.split(':')[1])) for t in spec.split(',')]


def main(argv):
    npub = int(argv[1])
    cands = dict(parse(c) for c in argv[2:])
    for row in run(cands, npub):
        pass


def run(cands, npub, verbose=True, extra_obs=None, full=False):
    """cands: {name: [(token, weight), ...]}; extra_obs: {name: public score} for already-submitted candidates.
    Returns [(name, cv_gain_vs_ch, cond_mean, cond_sd, P(>=TARGET))]."""
    obs_d = dict(OBS)
    obs_d.update(extra_obs or {})
    tr, te, y = H.load()
    names = set(V2 + F8 + ['chrep_bl'])
    for spec in cands.values():
        names |= {t for t, _ in spec if t not in ('ch', 'v2', 'f8')}
    O = {n: H.load_res(n)['oof'] for n in names}
    R = O['chrep_bl'].shape[0]
    rng = np.random.default_rng(12345)
    D = []
    for r in range(R):
        m = ~np.isnan(O['chrep_bl'][r])
        if any(np.isnan(O[n][r]).all() for n in names):   # block computed on fewer repeats: skip this repeat
            continue
        for n in names:
            assert (~np.isnan(O[n][r]) == m).all(), n
        idx_all = np.where(m)[0]
        B = {n: rk(O[n][r][m]) for n in names}
        B['ch'] = B['chrep_bl']
        B['v2'] = rk(np.mean([B[n] for n in V2], 0))
        B['f8'] = rk(np.mean([B[n] for n in F8], 0))
        B['f7'] = rk(np.mean([B[n] for n in F8 if n != 'mdl_mlp_ple'], 0))
        B['h25'] = 0.75 * B['v2'] + 0.25 * B['ch']
        for cn, spec in cands.items():
            if cn in B:
                continue
            w = np.array([x for _, x in spec]); w = w / w.sum()
            B[cn] = sum(wi * B[t] for (t, _), wi in zip(spec, w))
        keys = list(obs_d) + [c for c in cands if c not in obs_d]
        P = np.vstack([B[k] for k in ['ch'] + keys])
        yy = y[m]
        for _ in range(NSUB):
            s = rng.choice(len(idx_all), npub, replace=False)
            ys = yy[s]; n1 = ys.sum(); n0 = npub - n1
            Rs = rankdata(P[:, s], axis=1)
            auc = (Rs[:, ys == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
            D.append(auc[1:] - auc[0])
    D = np.array(D)
    ko = len(obs_d)
    cand_names = [c for c in cands if c not in obs_d]
    mu = D.mean(0); S = np.cov(D.T)
    obs = np.array([obs_d[k] - CH_PUB for k in obs_d])
    Soo = S[:ko, :ko] + np.eye(ko) * (1e-5) ** 2
    Sco = S[ko:, :ko]
    A = Sco @ np.linalg.inv(Soo)
    cm = mu[ko:] + A @ (obs - mu[:ko])
    Ccond = S[ko:, ko:] - A @ Sco.T
    cv = np.diag(Ccond)
    if full:   # names, CV-expected gains, conditional means, full conditional covariance
        return [c for c in cands if c not in obs_d], mu[ko:], cm, Ccond
    need = TARGET - CH_PUB
    out = []
    if verbose:
        print(f'npub={npub}  subsets={len(D)}  observed (file - ch) on public: ' +
              ' '.join(f'{k} {o:+.5f} (CV-expected {e:+.5f})' for k, o, e in zip(obs_d, obs, mu[:ko])))
        print(f'{"candidate":28s} {"CV gain vs ch":>13s} {"cond. mean":>10s} {"cond. sd":>8s} {"P(>=0.870)":>10s}')
    for i, cn in enumerate(cand_names):
        p = 1 - norm.cdf((need - cm[i]) / np.sqrt(max(cv[i], 1e-12)))
        out.append((cn, mu[ko + i], cm[i], np.sqrt(max(cv[i], 0)), p))
        if verbose:
            print(f'{cn:28s} {mu[ko + i]:+13.5f} {cm[i]:+10.5f} {np.sqrt(max(cv[i], 0)):8.5f} {p:10.3f}')
    return out


if __name__ == '__main__':
    main(sys.argv)
