"""Graph feature specs (see graph_lib.propagate) and screening variants / feature sets."""
GSPECS = {
    'g_hknn20': dict(method='harm', graph=dict(w_date=0.0, k=20, tau=1.0), m=1),
    'g_hknn40': dict(method='harm', graph=dict(w_date=0.0, k=40, tau=1.0), m=1),
    'g_hcomb': dict(method='harm', graph=dict(w_date=1.0, k=20, tau=1.0), m=1),
    'g_pknn20': dict(method='ppr', graph=dict(w_date=0.0, k=20, tau=1.0), alpha=0.5, m=1),
    # tuned on standalone AUC (std outer val): audio-weighted date edges + kNN in [Z, t/2], global prior
    'g_hc2': dict(method='harm', graph=dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'), m=0.3,
                  prior='global'),
    'g_hk2': dict(method='harm', graph=dict(w_date=0.0, k=20, tau=2.0, kernel='inv'), m=0.3, prior='global'),
}
VARIANTS = {
    'base': dict(add=[]),
    'hknn20': dict(add=['g_hknn20']),
    'hknn20_hcomb': dict(add=['g_hknn20', 'g_hcomb']),
    'pknn20': dict(add=['g_pknn20']),
    'swap_hknn20': dict(add=['g_hknn20'], drop=['knnw_yr_40']),
    'hc2': dict(add=['g_hc2']),
    'hc2_hk2': dict(add=['g_hc2', 'g_hk2']),
    # label-free diffusion-map coordinates of the graph (train+test features only)
    'spec_comb8': dict(add=[], lf=[('sc', 8, dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'))]),
    'dupm': dict(add=[], lf=[('dupm', 3, dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'))]),
    'spec_knn8': dict(add=[], lf=[('sk', 8, dict(w_date=0.0, k=20, tau=2.0, kernel='inv'))]),
}
GSETS = {
    'hc2': ['g_hc2'],
    'hc2_hk2': ['g_hc2', 'g_hk2'],
    'hknn20': ['g_hknn20'],
    # label-free: duplicate-copy signal of graph neighbours (date-mates + PPR over the combined graph)
    'dupm': dict(g=[], lf=[('dupm', 3, dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'))]),
    'spec8': dict(g=[], lf=[('sc', 8, dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'))]),
    'dupm_spec8': dict(g=[], lf=[('dupm', 3, dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv')),
                                 ('sc', 8, dict(w_date=0.3, date_bw=1.5, k=20, tau=2.0, kernel='inv'))]),
}
