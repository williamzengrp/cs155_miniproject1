import sys, numpy as np
sys.path.insert(0, './work')
import harness as H
from scipy.stats import spearmanr
ch = H.load_res('chrep_bl')
for nm in sys.argv[1:]:
    r = H.load_res(nm)
    tf = r['test_full']; fa = r['test_foldavg']
    s = spearmanr(tf, ch['test_full'])[0] if tf.size else np.nan
    sf = spearmanr(fa, ch['test_full'])[0] if fa.size else np.nan
    # OOF spearman with chrep on eligible rows (rep-wise mean)
    so = np.mean([spearmanr(r['oof'][k][~np.isnan(r['oof'][k])], ch['oof'][k][~np.isnan(ch['oof'][k])])[0] for k in range(r['oof'].shape[0])])
    print(f'{nm:28s} auc {r["aucs"].mean():.5f}  sp(test_full,ch) {s:.4f}  sp(foldavg,ch) {sf:.4f}  sp(oof,ch) {so:.4f}')
