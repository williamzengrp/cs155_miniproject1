"""r2_eda: segment-gated rank blend. Error analysis (eda5.out): the 3 label-free members (MLP, ch_et, ch_lgbm) are
0.015-0.024 AUC weaker than the label-feature trees on rows with many labelled date-mates, but as good as the best on
rows with none. Gate: per row, cnt = number of LABELLED rows (fit part in CV, all train at test time) sharing the release
date with a different uri (label-free: uses which rows are labelled, never their labels). Member weights depend on cnt.
python -I gate_blend.py SET        SET = std | c9
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0, './work')
import harness as H

S = sys.argv[1]
MEM = {'std': ['fkd_lgbm', 'fkt_xgb', 'fkt_cat', 'fkd_et', 'fkd_rf', 'mdl_mlp_ple', 'ch_et_proj', 'ch_lgbm_proj'],
       'c9': ['conf_fkd_lgbm', 'conf_fkt_xgb', 'conf_fkt_cat', 'conf_fkd_et', 'conf_fkd_rf', 'conf_mlp_ple',
              'conf_ch_et', 'conf_ch_lgbm']}[S]
NREP, SEED0 = {'std': (5, 1000), 'c9': (3, 9000)}[S]
tr, te, y = H.load()
n = len(tr)
raw = tr['track_album_release_date'].astype(str).to_numpy()
uri = tr['uri'].to_numpy()
splits = H.make_splits(tr, y, n_rep=NREP, seed0=SEED0)
CNT = np.full((NREP, n), np.nan)
dcode = pd.factorize(raw)[0]
ucode = pd.factorize(uri)[0]
for r, fi, vi, el in splits:
    c_date = np.bincount(dcode[fi], minlength=dcode.max() + 1)
    c_uri = np.bincount(ucode[fi], minlength=ucode.max() + 1)
    CNT[r, vi] = c_date[dcode[vi]] - c_uri[ucode[vi]]
oofs = [H.load_res(m)['oof'] for m in MEM]
LF = np.array([0, 0, 0, 0, 0, 1, 1, 1], bool)   # label-free members


def run(wfun):
    out = []
    for r in range(NREP):
        m = ~np.isnan(oofs[0][r])
        O = np.column_stack([rankdata(o[r, m]) / m.sum() for o in oofs])
        W = wfun(CNT[r, m])                 # (rows, 8)
        out.append(roc_auc_score(y[m], (O * W).sum(1) / W.sum(1)))
    return np.array(out)


def w_const(c):
    return np.ones((len(c), 8))


def gate(th, lf_w_hi, lf_w_lo=1.0):
    def f(c):
        W = np.ones((len(c), 8))
        W[:, LF] = np.where(c[:, None] >= th, lf_w_hi, lf_w_lo)
        return W
    return f


base = run(w_const)
print(f'[{S}] final8 {base.mean():.5f}')
for name, f in [('cnt>=4: LF w0.5', gate(4, 0.5)), ('cnt>=4: LF w0', gate(4, 0.0)), ('cnt>=1: LF w0.5', gate(1, 0.5)),
                ('cnt>=8: LF w0.5', gate(8, 0.5)), ('cnt==0: LF w2 (else 1)', gate(1, 1.0, 2.0)),
                ('cnt>=4: LF w0.5, cnt==0: LF w1.5', lambda c: np.where(c[:, None] == 0, gate(1, 1.0, 1.5)(c), gate(4, 0.5)(c)))]:
    b = run(f)
    d = b - base
    print(f'  {name:36s} {b.mean():.5f} gain {d.mean():+.5f} pos {(d > 0).sum()}/{NREP} {np.round(d, 4)}')
