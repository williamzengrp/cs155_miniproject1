import sys
sys.dont_write_bytecode = True
import numpy as np, glob
from scipy.stats import spearmanr
WORK = './work'
ch = np.load(WORK + '/results/chrep_bl.npz')
cht = ch['test_full']; chfa = ch['test_foldavg']
print('chrep_bl test_full vs test_foldavg spearman', spearmanr(cht, chfa)[0])
for f in sorted(glob.glob(WORK + '/r3_chp/cache/full_s*.npz')):
    d = np.load(f)
    bl = 0.6 * d['cat_t'] + 0.4 * d['et_t']
    print(f.split('/')[-1], 'spearman vs charlie file', round(spearmanr(bl, cht)[0], 5), ' cat', round(spearmanr(d['cat_t'], cht)[0], 4), ' et', round(spearmanr(d['et_t'], cht)[0], 4))
