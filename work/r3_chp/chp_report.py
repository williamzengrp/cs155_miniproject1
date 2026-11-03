"""r3_chp: summary of saved blocks: CV AUC (complete repeats) vs chrep_bl on the same repeats, Spearman of test_full
with Charlie's file, and the pubcheck rows (best E[public] / P). python -I -B chp_report.py NAME [NAME ...]"""
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import spearmanr
W = './work'
sys.path.insert(0, W); sys.path.insert(0, W + '/main')
import harness as H
import pubcheck as PC

ch = H.load_res('chrep_bl')
for name in sys.argv[1:]:
    r = H.load_res(name)
    have = [i for i in range(5) if not np.isnan(r['oof'][i]).all()]
    a = r['aucs'][have]; d = a - ch['aucs'][have]
    tf = r['test_full']
    sp = spearmanr(tf, ch['test_full'])[0] if np.size(tf) == 863 else float('nan')
    print(f'== {name}: CV {a.mean():.5f} on reps {have} (chrep_bl {ch["aucs"][have].mean():.5f}); paired d {d.mean():+.5f} '
          f'per-rep {np.round(d, 4)}; spearman(test_full, Charlie file) {sp:.5f}', flush=True)
    out = PC.check([name], verbose=False)
    out = [o for o in out if name in o[0]]
    for o in sorted(out, key=lambda o: -o[2])[:6]:
        print(f'   {o[0][:62]:62s} CVgain {o[1]:+.5f}  E[public] {PC.PS.CH_PUB + o[2]:.5f} (sd {o[3]:.5f})  P {o[4]:.3f}', flush=True)
