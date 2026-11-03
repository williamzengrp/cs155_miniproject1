# CS155 miniproject 1: final blends 19 and 21

This branch (`final-models-19-21`, created from `charlie`) contains our modelling code and the two submissions with the best public-leaderboard scores. Charlie's original README, which documents `reproduce.py` and his model, is kept as [`README_charlie.md`](README_charlie.md).

| File | What it is (weights of a rank average) | Public LB AUC | Offline AUC |
|---|---|---|---|
| `final_submissions/21_chxCat60_charlieGraphET20_lgblinear20.csv` | 0.60 **X** + 0.20 **ET** + 0.20 **LL** | **0.87475** | 0.8450 |
| `final_submissions/19_charlieGraphET28_chxCat40_lgblinear24_charlie8.csv` | 0.40 **X** + 0.28 **ET** + 0.24 **LL** + 0.08 **Charlie** | **0.87420** | 0.8462 |
| `final_submissions/v2_primary.csv` (our best model offline) | equal-weight rank average of 8 models | 0.86560 | **0.8476** |
| `final_submissions/reference_60_40_confirmed_positive.csv` (Charlie's model) | 0.6 CatBoost + 0.4 Extra Trees | 0.86886 | 0.8388 (our re-implementation) |

The components are:

- **X** = `r3_chx_c_cat`: Charlie's CatBoost, given a label-free album-cluster ID as a categorical feature.
- **ET** = `r3_chg_hc2_et`: Charlie's Extra Trees, given one extra graph label-propagation feature (`g_hc2`).
- **LL** = `zoo_lgb_linear`: a LightGBM with linear models in its leaves, trained on our own feature set.
- **Charlie** = Charlie's submitted predictions, used as they are.

Each is described below.

**Offline AUC** is the out-of-fold ROC AUC over 5 repeats of 5-fold cross-validation (`harness.make_splits`, `seed0=1000`), computed only on "eligible" validation rows. These are the rows that could have been in the test set (see [Validation](#validation)). The rank average is rebuilt from each component's out-of-fold predictions with the same weights. **Public LB AUC** is the Kaggle public leaderboard, which uses part of the 863 test rows. The private leaderboard uses the remaining rows, and only the private leaderboard is graded.

**Selection bias.** Files 19 and 21 were found through about 15 public submissions, each informing the next (see [How we got there](#how-we-got-there)). Their public scores are therefore optimistic estimates of their private scores. The offline AUC does not use any public-leaderboard information. By that measure the three files are close, and `v2_primary` is best. This is why we recommended selecting `v2_primary` + `21` as the two final Kaggle submissions: one trusts cross-validation, the other trusts the public leaderboard. Kaggle scores the better of the two on the private set.

## Rules

- **Data:** only the competition data, `data/train.csv`, `data/test.csv` and `data/sample_submission.csv`.
- **No pretrained or foundation models.** We tried TabPFN early on and dropped it once the course rule was clarified. TabPFN code, and the scripts that evaluated blends containing TabPFN, are not included in this branch, and no component of 19, 21 or `v2_primary` uses it. The only neural network is the small MLP inside `v2_primary`, which is trained from scratch (`work/mdl/`).
- **Test covariates.** Like Charlie's pipeline, some label-free features (neighbourhood, density and graph structure) are computed on pooled train and test covariates. No test labels exist or are used.

## The components of 19 and 21

All three model components are scored with the same validation scheme (5 repeats × 5 folds, eligible rows only; see [Validation](#validation)). Each saves out-of-fold predictions plus test predictions from full refits on all 3,863 training rows. The offline AUCs below are for each component alone.

### X: `r3_chx_c_cat` (weight 0.40 in 19, 0.60 in 21)

This is Charlie's final pipeline with only the CatBoost family kept, plus one new input: `c_alb`, a label-free album-cluster ID used as a native CatBoost categorical.

- **Builders:** `work/r3_chx/chx.py` and `chx_lib.py`. The pipeline logic is copied unchanged from our re-implementation of Charlie's model in `work/ch/`.
- **Inputs:**
  - Charlie's 115-column label-free view (`ch_feats.charlie_pool`, computed on the pooled train and test rows).
  - Six fold-local grouped date target encodings: year, year-month and exact-date High rates with smoothing 2, plus log counts. Fit rows are cross-fitted with `StratifiedGroupKFold(5)` over connected groups (same URI or identical 13-dimensional audio vector).
  - `c_alb`.
  - Median imputation, fitted on the fit rows.
- **`c_alb`** (`work/te/label_features.py`, `album_ids`): within each release-date string, the distinct songs of the train and test pool are clustered by average linkage on 9 standardised audio columns, cutting at distance 2.5. This gives 3,294 IDs and uses no labels. CatBoost's target statistics for the category come only from the labels of the rows it is fitted on.
- **Model:** CatBoost with Charlie's parameters (2,000 iterations, learning rate 0.00888, depth 9, l2 9.88, rsm 0.472, random strength 0.605, Logloss).
- **Same-date residual correction** (Charlie's): residuals come from an inner 3-fold over the fit rows. A Gaussian audio kernel over other songs with the same release date then gives `p = clip(raw + 0.35·corr, 0, 1)`.
- **Test predictions:** `test_full` is the mean of 6 full refits.
- **Offline AUC:** 0.8378 alone, which is +0.0011 over the replica's CatBoost and positive in 5 of 5 repeats. The gain is concentrated on rows whose album cluster contains labelled songs. That share is slightly higher in test (0.38) than in CV (0.34).
- **Leakage audit:** flipping every validation label left validation and test predictions bit-identical (`work/r3_verify/r3_chx_mix_ca/catfull_k0.out` and `catfull_k20.out`). Shuffling the fit labels gave AUC ≈ 0.50 (`work/r3_verify/chx_c/dyn_FS.out`).

### ET: `r3_chg_hc2_et` (weight 0.28 in 19, 0.20 in 21)

This is Charlie's Extra Trees on the same 115 + 6 columns, plus one new column, `g_hc2`, with Charlie's residual correction (α = 0.15, bandwidth 3).

- **Builders:** `work/r3_chg/chg.py` fits both of Charlie's halves with `g_hc2` added. `work/r3_chg/save_comp.py` writes `_cat`, `_et` (this block) and mixes such as `_w6` = 0.4 CatBoost + 0.6 ET.
- **Model:** `ExtraTreesClassifier(n_estimators=900, max_depth=15, min_samples_leaf=2, max_features≈0.183)`. Each song's rows are weighted 1 / (number of copies of its URI).
- **`g_hc2`** (`work/r3_chg/graph_lib.py`, `work/r3_chg/graph_specs.py`): harmonic label propagation on a label-free song graph over all train and test songs, with one node per distinct URI.
  - Edges join songs with the same release-date string, weighted `0.3·exp(-0.5·(‖Δz‖/1.5)²)` on audio. They also join symmetrised 20 nearest neighbours in audio and release-year space, weighted `0.25/(d+0.25)`.
  - Only fit-row labels are used as seeds.
  - Fit rows get out-of-fold values from an inner 5-fold grouped by URI, so a song never seeds itself.
  - The column is recomputed inside every residual-correction inner split.
- **Test predictions:** `test_full` is the mean of 3 full refits.
- **Offline AUC:** 0.8387 alone. On average that matches Charlie's full 0.6/0.4 replica blend (0.8388) and is +0.0086 over the replica's Extra Trees (5 of 5 repeats).
- **Leakage audit** (`work/r3_verify/r3_chg_hc2_w6/`):
  - Flipping validation labels left every input matrix and every output bit-identical (`valflip_k0.out`, `valflip_k15.out`).
  - Permuting fit labels gave AUC 0.50–0.54 (`fitperm_k0_s777.out`, `fitperm_k15.out`).

### LL: `zoo_lgb_linear` (weight 0.24 in 19, 0.20 in 21)

This is LightGBM with linear models in its leaves (`linear_tree=True`, `linear_lambda=1.0`), using the teammates' tuned LightGBM parameters (`work/main/final_params.json`: 662 trees, learning rate 0.0165, 30 leaves, depth 9). It is model `lgb_linear` in `work/r2_zoo/zoo.py`.

- **Inputs:** the 61 columns of `work/main/final_cv.py`.
  - **58 label-free features**, computed on the pooled train and test rows:
    - 13 raw audio features;
    - 4 release-date features;
    - trend (3) and seasonality (15) features;
    - same-date, month and year counts (4);
    - album-neighbour distances (6);
    - era-detrended audio (10);
    - date-density ratios (3).
  - **3 out-of-fold label features** (`work/te/label_features.py`, `BEST_SPEC`):
    - the distance-weighted High rate of the 40 nearest labelled songs released within ±1 year;
    - a smoothed exact-date target encoding;
    - the number of labelled songs with the same release date.

    Rows with the same URI are always excluded. Fit rows get inner out-of-fold values; validation and test rows are encoded from fit-row labels only.
- **Seeds:** 3 per fold. `test_full` is the mean of 3 full refits.
- **Offline AUC:** 0.8425 alone, which is +0.0037 vs the Charlie replica (5 of 5 repeats). The same LightGBM without linear leaves scores 0.8408.
- **Selection note:** it was picked from about 13 learner variants screened on the same splits.

### Charlie: `final_submissions/reference_60_40_confirmed_positive.csv` (weight 0.08 in 19)

This is Charlie's submitted file from this branch (`reference/submission_inverted.csv`), converted to High-class scores as `1 − Popularity_Type`. In cross-validation it is represented by our faithful replica `chrep_bl` (`work/ch/chfull.py` + `work/main/hedge_eval.py`; CV 0.8388). `chrep_bl`'s `test_full` is Charlie's actual file.

## Reproducing the submission files

All commands run from the repository root, because the scripts use paths relative to it.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r work/requirements.txt
```

### 1. Rebuild 19 and 21 from saved predictions (seconds)

`work/results/<name>.npz` stores each component's out-of-fold predictions (`oof`) and its full-refit test predictions (`test_full`). `work/main/blendfile.py` builds a weighted rank average: `p = Σ w_i · rank(v_i) / 863`, with SciPy average ranks and weights normalised to sum to 1. The token `ch` means the `Popularity_Type` column of `final_submissions/reference_60_40_confirmed_positive.csv`; any other token means `test_full` from `work/results/<token>.npz`.

```bash
python -I work/main/blendfile.py 21_rebuilt "r3_chg_hc2_et:0.20,zoo_lgb_linear:0.20,r3_chx_c_cat:0.60"
python -I work/main/blendfile.py 19_rebuilt "r3_chg_hc2_et:0.28,ch:0.08,zoo_lgb_linear:0.24,r3_chx_c_cat:0.40"
```

These write `submissions/21_rebuilt.csv` and `submissions/19_rebuilt.csv`. We checked that both are identical (max abs difference 0.0) to the submitted files in `final_submissions/`.

`final_submissions/reference_60_40_confirmed_positive.csv` is exactly `1 − reference/submission_inverted.csv`. It is Charlie's submitted file converted to High-class scores. The inverted file scores 0.131 on the leaderboard, so never submit it.

### 2. Retrain the components (minutes to hours, CPU only)

These commands rebuild each component's `work/results/*.npz` from scratch. They were originally run from inside each folder with absolute paths. In this branch the paths are relative to the repository root, so run them from the root. We verified the rebuild in step 1 in this layout, but did not re-run these long jobs here.

```bash
mkdir -p work/r2_zoo/cache work/r2_zoo/raw work/r3_chg/cache work/r3_chx/cache work/ch/cache   # caches are not in git

# LL: zoo_lgb_linear (about 5 minutes)
OMP_NUM_THREADS=2 NJ=2 DO_FULL=1 python -I work/r2_zoo/zoo.py run 1000 5 zoo 3 lgb_linear

# ET: r3_chg_hc2_et (about 2-3 hours on 1 thread; fits both of Charlie's halves)
python -I work/r3_chg/chg.py cv hc2 hc2 0 25 1 full
python -I work/r3_chg/chg.py fullfit hc2 hc2 0,1,2 1
OMP_NUM_THREADS=1 python -I work/r3_chg/save_comp.py hc2 r3_chg_hc2 0.6 0.7   # writes r3_chg_hc2{,_cat,_et,_w6,_w7}

# X: r3_chx_c_cat (several hours; CatBoost takes about 3-6 minutes per fold, plus 3 inner fits)
python -I work/r3_chx/chx.py cv c 0 25 2
python -I work/r3_chx/chx.py full c 0,1,2,3,4,5 2
OMP_NUM_THREADS=1 python -I work/r3_chx/chx.py save c   # writes r3_chx_c, r3_chx_c_cat, r3_chx_c_et
```

`save_comp.py` checks its outputs against the Charlie replica file `work/results/chrep_bl.npz`. `chx.py save` also checks against `chrep_cat.npz` and `chrep_et.npz`. All three are included. The replica itself comes from `work/ch/chfull.py cv std K0 K1` (25 folds) followed by `work/main/hedge_eval.py`. Runs at 1–2 threads were bit-reproducible with the package versions in `work/requirements.txt` (Python 3.11).

## Validation

- **Why "eligible" rows** (`work/harness.py`, `make_splits`):
  - The organisers split the original rows about 80/20, then removed every test row whose Spotify URI also appears in train. So test songs are essentially all singletons.
  - Train still contains extra copies of multi-playlist songs, which are mostly High.
  - Each CV repeat therefore uses `StratifiedKFold(5, shuffle=True, random_state=1000+r)`, but only scores validation rows whose URI does not occur in that fold's fit rows. That is about 3,500 of 3,863 rows per repeat.
  - Grouped CV that scores every duplicate overstates AUC: 0.852 vs 0.833 for the same LightGBM.
- **Leakage audits:**
  - X and ET passed full-model label-flip tests: flipping all validation labels leaves validation and test predictions bit-identical.
  - They also passed fit-label permutation tests (AUC ≈ 0.5). These audits are in `work/r3_verify/`.
  - For LL, the label features are checked by `work/te/test_leakage.py` (flipping validation labels changes nothing), and LightGBM is fitted on fit rows only.
- **Confirmation splits:** gains were re-checked on fresh split seeds (`seed0=9000` and `seed0=11000`) where noted in the code (`conf_*`, `c9`, `c11`).
- **Selection effects** (not leakage): several choices were made by looking at the same std splits. These include the `g_hc2` graph settings, the choice of linear-leaf LightGBM from the learner zoo, and Charlie's correction parameters. The 19/21 blend weights were chosen with public-leaderboard feedback. Expect true private-set AUCs slightly below the offline numbers, and well below the public scores of 19 and 21.

## How we got there

1. **Cross-validation that mirrors the test set.** We found that the test set is a random 20% of the original rows, with every test row whose Spotify URI also occurs in train removed. `harness.make_splits` copies this: each validation fold is scored only on rows whose URI does not occur in that fold's training rows.
2. **Our own models (`v2_primary`).** We built about 165 configurations: gradient-boosted trees, Extra Trees and random forests, an MLP, label-free trend, density and graph features, and out-of-fold label features. The best combination offline was an equal rank average of 8 models: LightGBM and XGBoost with graph features, CatBoost with an album-cluster categorical, two Extra Trees, an MLP, and LightGBM and Extra Trees on Charlie's feature view. It scored 0.8476 offline, versus 0.8388 for Charlie's model, and the gain held up on two fresh split seeds. On the public leaderboard, however, it scored 0.8656, below Charlie's 0.8689.
3. **Building on Charlie's model.** The public leaderboard favoured Charlie's exact ranking. We re-implemented his pipeline (`work/ch/`, the `chrep_bl` replica) and added improvements to it, each checked for leakage by flipping the validation labels and confirming that the predictions do not change:
   - the `g_hc2` graph label-propagation feature (`work/r3_chg/`);
   - an album-cluster categorical in CatBoost (`work/r3_chx/`);
   - out-of-fold label features (`work/r3_chp/`);
   - augmentation: noise, mixup and pseudo-labels (`work/r3_aug/`).
4. **A public-leaderboard simulator** (`work/main/pubsim.py`). It draws thousands of random public-sized subsets of the eligible out-of-fold rows. Over those subsets it estimates the joint Gaussian distribution of every candidate blend's AUC difference from Charlie's model. It then conditions on all public scores observed so far, which gives each untested blend an expected public score and a standard deviation. The search scripts below inflate the covariance by `(1+g)/(1−g)`, with `g = npub/3521`, to account for the finite test pool. `npub` is a command-line argument; we used 431 and 259.
5. **Adaptive search.** After each submission we added its score to the simulator and searched blends around the current best (`work/main/near02.py` … `near22.py`, `next_pick.py`). We chose the blend that maximised the expected best public score (`E[max(best so far, new)]`), considering several submissions together when more than one remained. Public scores of the files allowed under the rules, in order:

   | File | Blend | Public AUC |
   |---|---|---|
   | Charlie | his model | 0.86886 |
   | G40 | 0.6 Charlie + 0.4 graph LightGBM | 0.86977 |
   | 02 | 0.6 graph-Charlie + 0.2 Charlie + 0.2 album CatBoost | 0.87079 |
   | 11 | 0.40 graph-Charlie + 0.30 Charlie + 0.15 album CatBoost + 0.15 LL | 0.87183 |
   | 12 | 0.35 graph-Charlie + 0.30 Charlie + 0.05 album CatBoost + 0.30 LL | 0.87202 |
   | 18 | 0.32 ET + 0.24 Charlie + 0.24 LL + 0.20 X | 0.87318 |
   | 19 | 0.28 ET + 0.08 Charlie + 0.24 LL + 0.40 X | 0.87420 |
   | 21 | 0.20 ET + 0.20 LL + 0.60 X | 0.87475 |

   Here "graph-Charlie" = `r3_chg_hc2_w6`, which is 0.4 × its CatBoost half + 0.6 × its Extra Trees half (ET), and "album CatBoost" = `zoo_cat_alb`. The other submitted variants scored lower: 01, 05 and 10 scored 0.8693–0.8695, and v2 + Charlie scored 0.8677. Two directions kept beating the simulator's predictions: more weight on X, and replacing Charlie's own file with his re-implemented model plus our additions.

## Repository layout (this branch)

| Path | Contents |
|---|---|
| `work/harness.py` | Data loading, eligible-row CV splits (`make_splits`), saving and loading of `results/<name>.npz` |
| `work/main/` | Final CV and v2 assembly (`final_cv.py`, `assemble.py`), blend file builder (`blendfile.py`), public-LB simulator (`pubsim.py`) and adaptive search (`near*.py`, `next_pick.py`), `final_params.json` |
| `work/ch/` | Re-implementation of Charlie's pipeline |
| `work/te/`, `work/fe/`, `work/gbm/`, `work/mdl/` | Label features (target encodings, kNN label rates), label-free features, tree models, MLP |
| `work/r2_*` | Round-2 experiments: learner zoo (`r2_zoo`), graph features (`r2_graph`), label features (`r2_lab`), feature selection, distribution shift, … |
| `work/r3_chg`, `work/r3_chx`, `work/r3_chp`, `work/r3_aug` | Round-3 variants of Charlie's pipeline: graph feature, album categorical, label features, augmentation |
| `work/r3_verify` | Leakage audits: scripts and their outputs (`*.out`) |
| `work/results/` | Saved predictions for the components of 19, 21 and `v2_primary`, plus the Charlie replica `chrep_bl` |
| `final_submissions/` | The submitted files above |

The code was developed in a scratch directory. Absolute paths were rewritten to paths relative to the repository root. Only the final components' result files and none of the caches are included, so exploratory scripts that read other `results/` or `cache/` files need those inputs regenerated first. Some old run scripts, such as `work/v2logs/*.sh`, `cd` into a subfolder and so no longer match the root-relative paths. Docstrings that mention an "agent" refer to the parallel work streams used during development. Retraining `v2_primary` also needs `torch` (for its MLP) and `interpret` (for `work/mdl/ebm_cv.py`); neither is in `work/requirements.txt`.
