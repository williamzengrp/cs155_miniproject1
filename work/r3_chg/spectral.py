"""Label-free spectral / diffusion embeddings of the song graph (train+test features only)."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
WORK = './work'
sys.path.insert(0, WORK + '/r3_chg')


def spectral_embedding(G, ndim=16, t_diff=1, **gkw):
    """Diffusion-map coordinates: top eigenvectors of D^-1/2 W D^-1/2 (excluding the trivial one),
    psi = D^-1/2 u * lambda^t. Returns (Nn, ndim) node array."""
    W = G.build(**gkw)
    deg = np.asarray(W.sum(axis=1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    S = Dm @ W @ Dm
    vals, vecs = eigsh(S, k=ndim + 1, which='LA', tol=1e-6, maxiter=5000, v0=np.ones(G.Nn) / np.sqrt(G.Nn))
    o = np.argsort(-vals)
    vals, vecs = vals[o], vecs[:, o]
    psi = (Dm @ vecs)[:, 1:] * (vals[1:] ** t_diff)
    psi = psi * np.sign(psi[np.abs(psi).argmax(axis=0), np.arange(psi.shape[1])])   # deterministic sign
    return psi / psi.std(axis=0), vals


if __name__ == '__main__':
    sys.path.insert(0, WORK)
    import harness as H
    from graph_lib import SongGraph
    tr, te, y = H.load()
    G = SongGraph(tr, te)
    for name, kw in [('comb', dict(w_date=1.0, k=20, tau=1.0)), ('knn', dict(w_date=0.0, k=20, tau=1.0))]:
        psi, vals = spectral_embedding(G, 16, **kw)
        print(name, np.round(vals[:17], 4))
        from sklearn.metrics import roc_auc_score
        yy = y
        node_rows = G.node[:len(tr)]
        print('  per-dim train AUC:', np.round([max(roc_auc_score(yy, psi[node_rows, j]), 1 - roc_auc_score(yy, psi[node_rows, j])) for j in range(16)], 3))
