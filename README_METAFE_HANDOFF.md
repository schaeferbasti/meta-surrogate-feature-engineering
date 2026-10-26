# MetaFE handoff: same-split rerun, findings, and what has to be fixed

Written 2026-10-25 on the calculco/chimay cluster (Université du Littoral) before its shutdown on 2026-10-26 13:00.
It summarizes a multi-day working session, so a new agent on a new cluster can continue.
**Status in one sentence:** Original, OpenFE and TabPrep results on the fixed OpenML split with 3 AutoGluon seeds
are done and valid. MetaFE results are **invalid** because of target leakage plus a test-set bug in the MetaFE
pipeline, and MetaFE has to be fixed and rerun (including parts of the surrogate's training metadata).

## 00. Start here (latest state, 2026-10-25 evening)

**Read `README.md` first** (layout, commands, design decisions). Everything below §00 is history.

### Three experiments (`--experiment` of every module, defined in `metafe/config.py`)

| Experiment | What | Where / how | Status |
|---|---|---|---|
| `lightgbm_reuse` | LightGBM only, AutoGluon 1.2.1. Knowledge base = only the old labels whose training feature is identical with the fixed operators (75–93 %, no new fits); `Combine`/`CombineThenFreq` etc. have no labels and are never proposed. MetaFE up to 1 h. | calculco (OAR): `scripts/submit_oar.sh` | run with `MIN_IMPROVEMENT` 0.01 done (see below), results in `results/lightgbm_reuse/` (check whether they are in git). An earlier run with 0.1 is in `results/lightgbm_reuse_min0.1/` and `data/lightgbm_reuse/fe_min0.1/`: see below |
| `lightgbm` | Same, but the missing labels (mainly `Combine`) are computed with LightGBM: the full, clean LightGBM experiment. Completes the OpenFE/TabPrep comparison (same setup, cached results). | SLURM: `scripts/submit_lightgbm.sh` | not run |
| `multimodel` | LinearModel, RandomForest, LightGBM, TabICLv2, RealMLP (exactly these; the user does not want TabM, Mitra, CatBoost, ...), AutoGluon 1.6.3; TabICL and RealMLP on GPU. MetaFE per target model; evaluation with every model, averaged over the models. | SLURM: `scripts/setup_venv_multimodel.sh`, `scripts/submit_pilot.sh`, `python -m metafe.pilot summary`, then `scripts/submit_multimodel.sh` | not run; `max_candidates=500`, `max_label_rows=50000` are placeholders to set from the pilot |

**Order the user wants:** first check that MetaFE works with LightGBM (`lightgbm_reuse` result, otherwise
`lightgbm`), only then spend the compute on `multimodel`.

### Result of `lightgbm_reuse` with `MIN_IMPROVEMENT = 0.1` (2026-10-25, all 44 datasets, 3 seeds)

Knowledge base: 166,964 of 190,641 old labels reusable (88 %); dropped mainly `Combine` (291 left) and, on some
datasets, arithmetic operators; `CombineThenFreq` is almost fully kept. With clean labels only 0.5 % of the labels
are >= 0.1 (leaky old labels: 2.4 %; median 0, 90th percentile +1.25 %), so MetaFE took **no step on 21 of 44
datasets**; the other 23 took 39 steps on average (deeply nested GroupBy chains). Test improvement over Original:

| Method | mean | median | better / datasets |
|---|---|---|---|
| TabPrep | +5.00 % | +3.58 % | 30 / 44 |
| MetaFE (100 s) | +0.90 % | 0.00 % | 16 / 44 |
| MetaFE (250 s) | +0.66 % | 0.00 % | 16 / 44 |
| MetaFE (500–3600 s) | −0.14 … −0.37 % | 0.00 % | 12–15 / 44 |
| OpenFE | −7.78 % (one outlier) | +1.21 % | 26 / 36 |

Hence the rerun with 0.01 (now the experiment default).

### Result of `lightgbm_reuse` with `MIN_IMPROVEMENT = 0.01` (finished 2026-10-26 02:24, all jobs exit 0)

MetaFE took steps on all 44 datasets (38 on average). Test improvement over Original: MetaFE mean −21.6 % (100 s)
to −30.2 % (250 s), median ≈ 0 % at every time limit, better than Original on 21–25 / 44 datasets (TabPrep, OpenFE
unchanged). The mean is ruined by a few datasets (MetaFE 1000 s: wilt −1157 %, car −110 %, phoneme −33 %, socmob
−14 %); without the 3 worst, mean +0.21 %, median +0.01 %. Best: kr_vs_kp +21 %, jungle_chess +10 %, shuttle
+6 %, yeast +5 %. So with clean labels MetaFE is currently about neutral, with rare catastrophic failures, far
behind TabPrep. The failures (deeply nested features; log loss explodes on wilt/car) are the first thing to
investigate, in the step logs `data/lightgbm_reuse/fe/MetaFE_GBM/FE_<id>_MetaFE_log.csv`. Things to look at next: the surrogate keeps stacking
nested features (more time hurts); the threshold is a relative improvement and maybe should be compared with the
label distribution per dataset; the surrogate's inputs include `dataset - id` and `feature - name`.

### What to do on the new (SLURM, GPU) cluster

1. Clone, `git lfs pull`. Check `results/lightgbm_reuse/` (plots in `results/lightgbm_reuse/plots/LightGBM/`,
   `summary.csv` = mean/median improvement over Original). If MetaFE does not beat Original there, look at the
   step logs `data/lightgbm_reuse/fe/MetaFE_GBM/FE_<id>_MetaFE_log.csv` before running anything else.
   (In front-end smoke tests with a surrogate trained on 2–3 datasets, MetaFE stacked nested features with an
   almost constant predicted improvement of 0.33–0.36; with 44 datasets this has not been checked yet.)
2. Create `~/venvs/metafe` (AutoGluon 1.2.1, see §6; needed for `lightgbm*`, because the cached
   Original/OpenFE/TabPrep LightGBM results were produced with it and every module refuses a different version).
3. `lightgbm`: `DRY_RUN=1 scripts/submit_lightgbm.sh`, then submit (44 label jobs à 8 CPUs/64 GB/≤3 days, then
   MetaFE 44 × 8 CPUs/4 h, evaluation 132 × 8 CPUs/12 h; only MetaFE data is fitted in the evaluation).
4. `multimodel`: `TORCH_INDEX=<cuda wheel index> scripts/setup_venv_multimodel.sh`, then the pilot
   (`GPU_OPTS`/`CPU_OPTS` for partitions), `python -m metafe.pilot summary --experiment multimodel`, set
   `max_candidates` / `max_label_rows` in `metafe/config.py` to fit the budget, `scripts/submit_multimodel.sh`.
5. Ask the user before submitting anything, with resources and walltimes. Never commit or push (the user does).

### Running `lightgbm_reuse` on calculco (OAR)

From the login node `ritchie` (the `calculco` front end is the wrong node for submitting):

```bash
DRY_RUN=1 scripts/submit_oar.sh     # print the oarsub commands
scripts/submit_oar.sh               # 1 knowledge-base job (16 cores, 3 h), then per dataset 1 MetaFE job
                                    # (8 cores, 2 h) and 3 evaluation jobs (8 cores, 6 h), chained with oarsub -a
oarstat -u $USER                    # progress; logs in logs/
# when all jobs are done (seconds, on the login node):
PYTHONPATH=. ~/venvs/metafe/bin/python -m metafe.evaluation.merge_folds --experiment lightgbm_reuse
PYTHONPATH=. ~/venvs/metafe/bin/python -m metafe.evaluation.plots --experiment lightgbm_reuse
```

Scheduled core-hours ≈ 48 + 704 + 6336 ≈ 7.1k (user limit 8k, 192 cores in parallel). `FROM=metafe` /
`FROM=evaluate` resubmit later stages. The knowledge-base job logs, per dataset, how many labels were left out per
operator (`logs/kb_lightgbm_reuse_<id>.log`).

## 0. Update (2026-10-25, later): MetaFE fixed and the code refactored

The fix plan of §4.4 is implemented, and the repository was restructured into the `metafe` package (see
`README.md` for the layout and how to run it). MetaFE has **not been rerun yet**: no jobs were submitted.

- **Operators (§4.4 step 1):** `metafe/operators.py`: fitted on train rows, applied to train and test rows,
  codes in sorted order (row-order independent). Verified by `tests/test_operators.py` (0 failures on 4 datasets).
- **Labels (step 2):** `metafe/core_matrix.py` recomputes the core matrix; old labels are reused where the fixed
  operators give the same training feature (checked with an exact O(n) replica of the old operators,
  `metafe/legacy_operators.py`, verified identical to the old code on 1959 candidates). New: `--models` for
  more core-matrix models (RF, LR, ...).
- **MetaFE (step 3):** `metafe/metafe.py` applies every step to train and test rows, writes all 7 time levels in
  one run, never re-applies features, and logs every step (`FE_<id>_MetaFE_log.csv`).
- **Checks (step 4):** `metafe/evaluation/check_fe.py` (FAILs both bugs on the old MetaFE files, passes OpenFE).
- **Evaluation:** `metafe/evaluation/evaluate.py` reproduces the cached Original result bit-exactly; the valid
  Original/OpenFE/TabPrep results were copied into `results/lightgbm/evaluation/GBM/`, so only MetaFE needs fitting
  (experiment `lightgbm`; the `multimodel` experiment with five models is described in README.md).
- **Naming bug found:** the surrogate was trained on `add`/`subtract`/`multiply`/`divide`, but MetaFE asked it
  about `+`/`-`/`*`/`/` (unseen categories). Now all names use the word form.
- **Deleted** (still in git history, commit `53ffa39`): MFE/d2v/TabPFN metafeatures, one-shot / MWI / NOF variants,
  BO, Motivation, OAR scripts, duplicate data. Old results and plots were moved to `results/legacy/`.

Old path → new path:

| Old | New |
|---|---|
| `src/Metadata/core/core_submatrices/*.parquet`, `Core_Matrix_Complete.parquet` | `data/legacy/core_matrix/`, `data/legacy/core_matrix.parquet` |
| `src/Metadata/pandas/Pandas_Matrix_Complete*.parquet` | `data/legacy/pandas_matrix/` |
| `src/Apply_and_Test/test_data/FE_<id>_{OpenFE,TabPrep}.parquet` | `data/fe/{OpenFE,TabPrep}/` |
| `src/Apply_and_Test/test_data/<t>_FE_<id>_pandas_CatBoost_recursion.parquet` (invalid) | `data/legacy/fe_metafe_v1/` |
| `src/Apply_and_Test/test_results_same_split_seeds/` | `results/legacy/same_split_seeds/` (valid parts copied to `results/lightgbm/evaluation/GBM/`) |
| `src/Apply_and_Test/test_results_same_split/`, `test_results/` | `results/legacy/same_split_identical_folds/`, `results/legacy/random_split/` |
| `src/Result_Analysis/test_analysis/*.pdf|png` | `results/legacy/plots_v1/` |
| `Test_FE.py`, `merge_result_folds.py`, `test_analysis.py` | `metafe.evaluation.evaluate`, `.merge_folds`, `.plots` |
| `Generate_OpenFE.py`, `tabprep_env/tabprep_data_generation.py` | `metafe.evaluation.baselines --method OpenFE/TabPrep` |
| `SurrogateModel_TabArena_Recursion_Improvement_Parallel.py` | `metafe.metafe` |
| `Operator_Model_Feature_Matrix_*_Parallel.py`, `Add_Pandas_Metafeatures.py` | `metafe.core_matrix`, `metafe.metafeatures` |

Sections 1–6 below describe the state before this update; their paths are the old ones.

---

## 1. Goal of the rerun

Fair comparison of feature-engineering (FE) methods for the paper's improvement plots:

- **Original**: no FE.
- **MetaFE**: our method, "pandas" metafeatures, recursive surrogate (`pandas_CatBoost_recursion`), at 7 time limits:
  100, 250, 500, 1000, 1800, 3600, 7200 s. These are the `<t>_pandas` methods.
- **OpenFE** (openfe 0.0.12).
- **TabPrep** ([atschalz/tabprep](https://github.com/atschalz/tabprep), paper arXiv:2606.02384).

All methods are evaluated on the **same fixed OpenML train/test split** (`get_openml_dataset_split_and_metadata`,
default OpenML split of each task). The downstream model is AutoGluon LightGBM, and the reported score is
`LightGBM_BAG_L1`.

**Why the rerun:** the old results (`src/Apply_and_Test/test_results/`) were unfair. Original used the OpenML split,
but MetaFE/OpenFE used `split_data()`, an **unseeded random 80/20 re-split of the FE file per fold**. So the
methods were scored on different test rows, and all fold-to-fold variation came from those random splits.
In the old results, Original had a test std of exactly 0 on all 41 datasets.

---

## 2. Final valid results (frozen)

Folder: `src/Apply_and_Test/test_results_same_split_seeds/` (committed in `53ffa39`).

- **Per method, dataset and fold:** `<method>_Result_<id>_<fold>.parquet`.
- **Per dataset and fold, all methods combined:** `Result_<id>_<fold>.parquet`.
- **Averaged over the 3 folds:** `Result_<id>.parquet`, with columns `score_{val,test}_{mean,std}` and `n_folds`.
  Produced by `python merge_result_folds.py test_results_same_split_seeds`.
- **Folds = AutoGluon seeds 0, 1, 2 on the same split.** The seed sets the bagging fold split
  (`learner_kwargs={"random_state": seed}`) and LightGBM `seed_value`.
  - Seed 0 equals AutoGluon's defaults; I verified it reproduces the earlier unseeded scores bit-exactly.
  - Median test std across seeds: Original 0.0015, TabPrep 0.0016, OpenFE 0.002.
- **Coverage:**
  - 44 datasets × 3 folds for Original, TabPrep and the 7 MetaFE levels.
  - OpenFE for 36 datasets. OpenFE never finished on 8 datasets (`NO_OPENFE_DATASETS` in `Test_FE.py`):
    2073, 359930, 359931, 359935, 359938, 359959, 359962, 359983.

Improvement over Original, in % of the test score (positive = better), across datasets:

| Method | Datasets | Mean | Median | Better than Original | Median (classification / regression) |
|---|---|---|---|---|---|
| TabPrep | 44 | +5.0 % | +3.6 % | 30 / 44 | +4.4 % / +3.6 % |
| OpenFE | 36 | −7.8 % | +1.2 % | 26 / 36 | +1.2 % / +2.1 % |
| MetaFE (any level) | 44 | **invalid**, see §4 | | | |

Scores are `-log_loss` for classification (`eval_metric="log_loss"`, `problem_type="multiclass"`, also for binary
tasks) and `-RMSE` for regression. The large max std values come from regression RMSE scales.

**Plots:** `src/Result_Analysis/test_analysis/*.pdf|png` were regenerated from `test_results_same_split_seeds`
(commit `ffd75bd`/`53ffa39`). **Their MetaFE parts are invalid.** By request, the plots were not regenerated
again. The old (pre-rerun) plots are in git history, but they are also leakage-inflated, see §4.

---

## 3. What was done (chronological)

1. **Data.** `src/Apply_and_Test/test_data/` (Git LFS):
   - 308 MetaFE files `<t>_FE_<id>_pandas_CatBoost_recursion.parquet` and 44 `FE_<id>_TabPrep.parquet`.
   - 36 `FE_<id>_OpenFE.parquet`, generated here.
   - All 44 datasets were validated: each FE file maps onto the OpenML split (train rows first, then test rows).
2. **TabPrep** for the two missing datasets 359938 and 359962 was generated with
   `tabprep_env/tabprep_data_generation.py --dataset ...` (venv `~/venvs/tabprep`, see §6).
3. **OpenFE generation** (`src/Apply_and_Test/Generate_OpenFE.py`, `run_fe_openfe.oar`), one job per dataset.
   - **Bug found and fixed:** OpenFE's `transform()` (`openfe/utils.py`) ignores `tmp_save_path` and always writes
     `./openfe_tmp_data.feather` in the working directory. Parallel jobs overwrote each other's file:
     `ArrowInvalid: Field named ... not found` crashes, or silently wrong features.
   - OpenFE also catches worker errors and calls `exit()`, so failed jobs exited 0.
   - Fix (`src/utils/get_data.py::get_openfe_data`): run `fit` and `transform` inside `contextlib.chdir(mkdtemp())`.
   - Fix (`Generate_OpenFE.py`): turn OpenFE's `SystemExit` into exit 1.
   - All 36 files were regenerated into `test_data_openfe_v2/` and compared with the earlier ones:
     - 28 identical.
     - 7 new (359936, 359949, 359954, 359963, 359971, 359975, 359982).
     - 1 changed (**359981**: the earlier file was corrupted; it is kept in `~/2_metafe/openfe_suspect/` on the
       old cluster).
   - `run_openfe_reeval.oar` swapped the files in and re-evaluated.
4. **Identical folds explained.** AutoGluon is deterministic with its defaults (bagging `random_state=0`, LightGBM
   `seed_value=0`), so 3 "folds" on a fixed split were bit-identical. The user chose to vary the AutoGluon seed per
   fold: `seed = fold`, see `src/utils/run_models.py::with_seed` and the `seed` arguments.
5. **Evaluation.**
   - `Test_FE.py --dataset <id> --fold <f> --skip-missing-fe [--skip-openfe] [--results_dir <dir>]`, one OAR job
     per dataset and fold (`run_test_fe.oar`).
   - Results are cached per method file, so reruns only fit what is missing.
   - Evaluation is fast: median about 2 min, max about 30 min per dataset and fold on 8 cores.
6. **Plots.**
   - `src/Result_Analysis/test_analysis/test_analysis.py` now reads `test_results_same_split_seeds`.
   - The `only_pandas` plots are skipped, because there are no one-shot SM results in the rerun.
   - It needs `adjustText` (installed in `~/venvs/metafe`).
   - PNGs were made with `pdftoppm -png -r 100 -singlefile`, since the script only writes PDFs.

---

## 4. MetaFE findings: why it must be fixed (the main point of this handoff)

### 4.1 Bug 1: MetaFE features are never computed on the test rows

`src/SurrogateModel/SurrogateModel_TabArena_Recursion_Improvement.py`, `main()`, `pandas` branch
(around lines 211-218; the `mfe` branches have the same pattern):

```python
X_train, y_train = recursive_feature_addition(X_train, y_train, ...)   # features added to X_train only
data = concat_data(X_train, y_train, X_test, y_test, "target")          # X_test unchanged -> NaN in new columns
```

- Every generated feature is NaN on all test rows.
- This affects **159 of 308** MetaFE files: every file where MetaFE actually added features.
- The other 149 files contain no generated features (142 with no new columns, 7 with only renamed original
  columns). They are identical to Original, and their improvement is exactly 0 %.
- The operators in `src/utils/create_feature_and_featurename.py` have **no fit/transform separation**:
  - `Combine` = `pd.factorize`.
  - `GroupByThen*` statistics are computed on whatever rows they are given.
  - `min`/`max`/`freq` and the non-numeric `factorize` paths work the same way.
- So the test rows cannot simply be fed through the same functions. They need train-fitted versions of the
  operators.

### 4.2 Bug 2: target leakage through row order (`Combine`)

- `task.get_train_test_split_indices()` returns the OpenML **train rows grouped in class blocks**. On binary
  datasets the class changes only 18 times along the train rows, against about 300-2000 times in random order.
- `Combine(a, b)` = `pd.factorize(a_str + "_" + b_str)` numbers values **in order of first appearance**.
  For high-cardinality combinations the code is roughly the row position (correlation with row index 0.98-1.0).
  So the code encodes the class block, which is **target leakage**.
- Evidence on the evaluation side:
  - australian (146818), MetaFE 7200: validation log loss 0.065 against 0.31 for Original. Test log loss 1.85
    against 0.30, because the test rows have NaN features (bug 1).
  - New fixed split, MetaFE files with new features: improvement mean −287 %, median −17.5 %.
  - Old random re-split (`test_results/`), MetaFE files **with** new features: mean +22 %, median +13 %.
    About 90 % of the old test rows came from the leaky train part.
  - Old random re-split, files **without** new features: mean −3 %, median −2 % (just split noise).
  - So **the previously reported ~15 % MetaFE gain was leakage**, not a real effect.

### 4.3 Bug 3: the surrogate's training metadata is contaminated by the same leak

- **Core matrix:** `src/Metadata/core/Core_Matrix_Complete.parquet`, 190,641 rows. Columns: `dataset - id`,
  `feature - name`, `operator`, `model`, `improvement`. It covers the same 44 datasets as the test set; MetaFE
  leaves the target dataset out.
- **How it was built:** `src/Metadata/core/Operator_Model_Feature_Matrix_{Classification,Regression}_Parallel.py`.
  - Each candidate feature is built separately on `X_train` and `X_test` by `create_feature_and_featurename`
    (lines 137-139).
  - Label = relative improvement of AutoGluon **`score_val`** (bagging on the class-ordered OpenML train rows)
    over the original score (lines 55-71).
- **Label by operator:** `Combine` has mean +14 %. Every other operator is at most +0.6 % (`subtract` +0.6 %,
  `add` +0.55 %, `GroupBy*` ≤ +0.13 %).
- **`Combine` labels rise with code uniqueness** (share of distinct codes among the train rows):

  | Distinct `Combine` codes | Labels | Mean label | Median label |
  |---|---|---|---|
  | < 1 % | 3,475 | +1.0 % | +0.02 % |
  | 1-10 % | 2,238 | −0.1 % | +0.01 % |
  | 10-50 % | 4,058 | +3.8 % | +0.5 % |
  | 50-90 % | 2,521 | +26 % | +22 % |
  | > 90 % (row-ID-like) | 1,832 | **+61 %** | **+76 %** |

- So the surrogate learned "near-unique `Combine` = large improvement", and MetaFE then selected exactly those
  features.
- `CombineThenFreq` (+0.4 % mean) does not look affected, since counts don't depend on row order.
- The `src/Metadata/pandas/Pandas_Matrix_Complete*.parquet` files (core matrix + pandas metafeatures, used by
  `recursive_feature_addition`) inherit the contaminated labels.

### 4.4 Suggested fix plan for MetaFE (needs more compute than was left)

1. **Make the operators order-independent and fit/transform-capable.**
   - Shuffle the train rows (seeded) before any FE and before computing labels, or at least use
     `pd.factorize(..., sort=True)`, or hash/sort-based codes.
   - Better: give every operator a fit-on-train / apply-to-test form:
     - `Combine`: train code map; unseen combinations become NaN or a new code.
     - `GroupBy*`: train group statistics mapped by the test key.
     - `min`/`max`/`freq`: train statistics.
     - Arithmetic on factorized non-numeric columns: train category maps.
   - Also check whether bagged-validation labels are still biased in other ways once the order leak is gone.
2. **Recompute the surrogate labels.** At minimum the 14,124 `Combine` rows of the core matrix. Ideally re-check
   all operators with the fixed implementation and shuffled rows, since some non-numeric paths also `factorize`.
   Each label is one AutoGluon run (8-fold bagged LightGBM); the scripts run per dataset in parallel.
   Then rebuild `Pandas_Matrix_Complete*` (`src/Metadata/pandas/Add_Pandas_Metafeatures.py`).
3. **Fix the MetaFE generation** (`SurrogateModel_TabArena_Recursion_Improvement*.py` / `Run_MetaFE.py`): record
   the chosen operations and apply them, train-fitted, to `X_test` before `concat_data`.
4. **Sanity checks before trusting the new MetaFE data:**
   - No new column may be all-NaN on the test rows.
   - Validation and test improvement must agree in sign and size. A large validation gain with a test loss means
     leakage.
   - Correlation of every generated feature with the train row index should be about 0.
5. **Regenerate the 7 time levels for all 44 datasets.**
   - Time limits sum to 13,950 s per dataset, about 170 job-hours in total if every level runs to its limit.
     Levels and datasets can run in parallel.
   - Then evaluate with the existing pipeline:

     ```
     Test_FE.py --dataset <id> --fold <0|1|2> --results_dir <new dir> --skip-missing-fe
     ```

   - Copy or reuse the valid Original/TabPrep/OpenFE result files from `test_results_same_split_seeds` (same
     seeds), so only MetaFE needs fitting.
   - Delete the old `<t>_pandas_Result_*` files first, otherwise the per-method cache reuses them.
   - Merge with `merge_result_folds.py <dir>`, then point `RESULTS_DIR` in `test_analysis.py` at it.
6. **Paper:** the earlier MetaFE numbers (~15 % improvement) should not be used.

---

## 5. Code changes made in this session (all committed by the user)

| File | Change |
|---|---|
| `src/utils/get_data.py` | Lazy `openfe`/`tabprep` imports. `get_openfe_data(..., n_jobs)` runs in a private temp working dir (OpenFE collision fix). |
| `src/utils/run_models.py` | `seed` argument, default 0 = previous behaviour, for the `*_origin_*` / `improvement` predictors. `learner_kwargs={"random_state": seed}` and `with_seed()` setting LightGBM `seed_value`. |
| `src/Apply_and_Test/Test_FE.py` | Fixed OpenML split for all methods. `seed = fold`. New options `--dataset`, `--fold`, `--skip-missing-fe`, `--skip-openfe`, `--results_dir`. Default 3 folds. |
| `src/Apply_and_Test/Generate_OpenFE.py` | New. One dataset per call, `--n_jobs`, `--out_dir`, atomic write, fails on OpenFE's silent exit. |
| `src/Apply_and_Test/merge_result_folds.py` | Optional results-dir argument. |
| `tabprep_env/tabprep_data_generation.py` | `--dataset` option, atomic write. |
| `run_test_fe.oar`, `run_fe_openfe.oar`, `run_fe_tabprep.oar`, `run_openfe_reeval.oar`, `submit_same_split.sh` | OAR job scripts, see §6. |
| `src/Result_Analysis/test_analysis/test_analysis.py` | Reads `test_results_same_split_seeds`. Skips `only_pandas` plots without one-shot results. |

---

## 6. Environment / cluster notes (old cluster; adapt on the new one)

- **Old cluster:** OAR scheduler. Submit from the login node `ritchie`; the `calculco` front end showed every node
  as Dead and was the wrong node.
- **CPU only.** Node filter used:

  ```
  -p "(gpu='' OR gpu IS NULL) AND type='default' AND cputype<>'GraceA02'"
  ```

  This gives x86 Xeon/EPYC nodes. GraceA02 nodes are ARM, where the x86 venvs don't work. Each job prints its
  host and CPU model.
- **Per-user limits there:** 192 cores in parallel and 8000 core-hours scheduled, so about 24 jobs of 8 cores.
- **Job sizes:**

  | Job | Cores | Walltime | Notes |
  |---|---|---|---|
  | Evaluation | 8 | 6 h | matches AutoGluon `num_cpus=8`, `num_gpus=0`, `memory_limit=48` GB, `time_limit` 4 h per fit, `presets="best_quality"`, `num_bag_folds=8`, `num_stack_levels=0`, no weighted ensemble, LightGBM zeroshot config only |
  | OpenFE | 8 | 2 h | longest run was 268 s |
  | TabPrep | 4 | 4 h | |

- **Venv `~/venvs/metafe`** (evaluation + OpenFE), Python 3.11.2:
  1. torch 2.6.0+cpu / torchvision 0.21.0 from `https://download.pytorch.org/whl/cpu`.
  2. `pip install --no-deps -r requirements.txt`, without `causalnex` (no Python 3.11 wheel) and with the six
     `@ file://` conda pins unpinned. The freeze is internally inconsistent (`huggingface-hub==1.0.0` vs
     `transformers`), hence `--no-deps`.
  3. `pillow<12`, `packaging<25`, `greenlet`, `wheel`, `adjustText`.
  4. Resulting key versions: AutoGluon 1.2.1b20250430, openfe 0.0.12.
- **Venv `~/venvs/tabprep`:** TabPrep needs AutoGluon from source.
  - AutoGluon master commit `9ad8e9a9` (2026-10-22), editable `common/`, `core/`, `features/`.
  - Plus `openml` and `pyarrow`.
  - `tabprep.py` from atschalz/tabprep commit `fee7f189`, copied into site-packages.
  - It is unknown which AutoGluon commit produced the 42 pre-existing TabPrep files.
- **Git LFS:** the data and result parquet files are LFS objects. On a fresh clone:

  ```
  git lfs install --local --skip-repo
  git lfs pull --include "data/**,results/lightgbm/**"
  ```

  Without it, you only get 130-byte pointer files.
- **OpenML:** datasets are cached in `~/.openml`. Pre-download all 44 tasks once before running parallel jobs.
- **Only on the old cluster, not in git:**
  - `~/2_metafe/openfe_suspect/` (corrupted earlier OpenFE file for 359981).
  - Job logs in `~/2_metafe/logs/`.
  - The venvs.
