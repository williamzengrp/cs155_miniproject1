"""3-seed confirmation runs (seed offsets 0/100/200 averaged per fold) vs the 3-seed baseline.

usage: python confirm.py <tag>=<feat1,feat2,...> [<tag>=...]
Saves out/oof_<tag>.npy and out/res3_<tag>.npz (oof, aucs, test_foldavg) for later H.save.
"""
import sys

sys.path.insert(0, './work/te')
import numpy as np  # noqa: E402
from common import run, OUT, SEED_OFFS  # noqa: E402

for arg in sys.argv[1:]:
    tag, feats = arg.split('=')
    spec = [f for f in feats.split(',') if f]
    res = run(tag, spec, offs=SEED_OFFS, with_test=True)
    np.savez(f'{OUT}/res3_{tag}.npz', oof=res['oof'], aucs=res['aucs'], test_foldavg=res['test_foldavg'])
