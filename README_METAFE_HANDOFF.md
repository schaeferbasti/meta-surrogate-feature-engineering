# MetaFE handoff: same-split rerun, findings, and what has to be fixed

Written 2026-10-25 on the calculco/chimay cluster (Université du Littoral) before its shutdown on 2026-10-26 13:00.
It summarizes a multi-day working session, so a new agent on a new cluster can continue.
**Status in one sentence:** Original, OpenFE and TabPrep results on the fixed OpenML split with 3 AutoGluon seeds
are done and valid. MetaFE results are **invalid** because of target leakage plus a test-set bug in the MetaFE
pipeline, and MetaFE has to be fixed and rerun (including parts of the surrogate's training metadata).

## 000. Newest (2026-10-26 evening, new cluster kis3bat1 / SLURM): why MetaFE fails, and the plan

**Diagnosis of the third `lightgbm_reuse` run** (`results/lightgbm_reuse/plots/LightGBM/summary.csv`: MetaFE mean
−7.5 … −9.4 %, median ≈ 0 %, 20–23/44 better; TabPrep +5.0 % / +3.6 %, 30/44):
1. **The surrogate does not rank candidates on unseen datasets.** Leave-one-dataset-out (LightGBM surrogate on the
   metafeature matrix, without `dataset - id` / `feature - name`): median per-dataset Spearman 0.006; the median real
   label of its top-10 picks is 0.0 %, the same as random; the best candidate per dataset is worth a median of +3.5 %.
2. **The metafeatures cannot carry that information:** only operator, task type and pandas `describe()` of the
   candidate on its own. There is nothing about y, the other columns or the dataset, so there is no actual
   "dataset similarity".
3. **Labels are mostly noise:** 39 % are exactly 0 (trees ignore monotone transforms); the non-zero labels have
   median |label| 0.59 % and are 53 % positive / 47 % negative; Original's seed std is 0.75 % (val). Single seed-0
   labels. Dataset 359930 (quake) has all 75 labels at +0.89 (broken baseline score? check).
4. **The recursion goes out of distribution:** the logs show copies like `subtract(add(X, Mean_R), Mean_R)` and
   `GroupByThenMean(persons, persons)`; the predictions are nearly constant (e.g. 0.042952 repeated).

**Plan the user approved** (run in order, each only if the previous one works; do NOT recompute the knowledge base
before):
1. Target-aware metafeatures for the existing KB rows (MI with y, correlation ratio with the residual of an OOF
   LightGBM on the original columns, gain over the operands, redundancy with the existing columns, scale-free stats,
   dataset-level metafeatures), then rerun the leave-one-dataset-out check (old vs new vs both). Gate: top-10 picks
   clearly beat random (Wilcoxon over datasets) and Spearman on the non-zero labels > ~0.15.
2. Baseline without meta-learning: rank by the residual-gain statistic alone, same metrics.
3. MetaFE v2: depth ≤ 2, near-duplicate filter (|Spearman| > 0.98), surrogate/heuristic only pre-selects the top-k,
   each one is verified by a 5-fold LightGBM CV on the training rows; evaluate with a proxy (plain LightGBM, 3 seeds, OpenML
   split) on Original/TabPrep/OpenFE/old MetaFE/new MetaFE.

**State of the code:** `experiments/surrogate_check/common.py` (metafeature code, `DatasetContext`) and
`step1_metafeatures.py` (tested on 146818: 8 s). Pending fix in `common.py` `mutual_info`: replace non-finite
values by the median of the **finite** values (`log(A9)` failed with inf). Not yet written: the step 1 LODO check,
step 2, step 3, and the SLURM driver.
**Environment:** venv `/work/dlclarge1/matusd-schaefeb-metafe/venvs/analysis` (pandas 3, lightgbm, sklearn, openml,
no AutoGluon); all 44 OpenML tasks cached in `/work/dlclarge1/matusd-schaefeb-metafe/openml_cache` (`common.py`
sets it). Run with `PYTHONPATH=.` from the repo. Login node has 16 cores; use SLURM for the full run
(`mldlc2_cpu-epyc9655`, 192 cores/node, or `gki_cpu-cascadelake`; both accept 32 CPUs/23 h). The user authorized
running the experiments overnight.

**Result (2026-10-26 22:20): the step 1 gate FAILED, so step 3 was not run.** Step 1a (`mf_v2`, all 44 datasets,
0 failures, 2 min on SLURM; `mutual_info` fix and nested operands done) and the LODO check
`step1_lodo.py` (also step 2) → `results/surrogate_check/step1_lodo/{summary,per_dataset}.csv, gate.json`. On the
add-feature candidates, median per-dataset Spearman on the non-zero labels is 0.03–0.06 for every surrogate
(old/new/both, label or per-dataset rank target) and every heuristic (gate: > 0.15). Top-10 picks vs random
(Wilcoxon on top-10 mean label minus the dataset's mean label): surrogates p = 0.09–0.43 (new 0.09, both 0.43, old
0.08); heuristics `eta_res` p = 0.005 (30/44 above the mean), `eta_y_gain` 0.009, `eta_res_gain` 0.018. Their
gain is tiny (median top-10 excess +0.2 % vs an oracle +2.3 %). NaNs in `mf_v2` are only skew/kurt of
constant features. Step 3 is written and smoke-tested (146818, `--ranker heu_eta_res_gain`): `step3_metafe_v2.py`
(depth ≤ 2, |Spearman| > 0.98 filter, top-10 → 5-fold LightGBM CV verification, greedy add),
`step3_evaluate.py` (plain LightGBM, 3 seeds, OpenML split; `--summary`), SLURM wrapper `run.sh` (array index →
dataset). To run it anyway with a heuristic: `sbatch --array=0-43 ... run.sh step3_metafe_v2.py --ranker heu_eta_res`.

**Label reliability test and new knowledge base (2026-10-27).** `experiments/label_reliability/label_reliability.py`
reran the knowledge-base label process exactly (AutoGluon 1.2.1 LightGBM, 8-fold bagging; a seed-0 rerun reproduced
5/5 labels bit-exactly) with seeds 1–3 on 200 random + top-20 candidates of 10 datasets. It was stopped by the user
at ~10 % (enough): on the add-feature candidates of 146818, 168757, 359950 the labels agree with themselves across seeds
only with Spearman 0.21 (non-zero labels 0.27), sign agreement 0.60, ICC 0.23 (seed noise sd 0.60 % > real spread
between candidates 0.42 %). The top-20 candidates hold up (+2.0 % in the KB, +1.7 % with new seeds, 89 % stay
positive); delete labels are reliable (ICC 0.94). Plain LightGBM 5-fold CV × 10 repeats: ICC of the mean ≈ 0.77
(only 146818 measured). Hence the knowledge base is being recomputed with that label: experiment `lightgbm_cv`
(see README.md), submitted 2026-10-27 07:46 (`UNTIL=mf_merge`, jobs 30096235 labels, 30096236 merge, 30096255/6
metafeatures; venv `/work/dlclarge1/matusd-schaefeb-metafe/venvs/ag12` = AutoGluon 1.2.1b20250430 with the original
pins, `OPENML_CACHE=/work/dlclarge1/matusd-schaefeb-metafe/openml_cache_np1` because the cache pickles depend on the numpy
major version). The package code reproduces the reliability test's CV labels to 1e-16.

**Result (finished 2026-10-27 11:49, all 90 jobs COMPLETED, 0 failed candidates):** `data/lightgbm_cv/core_matrix.parquet`
and `metafeature_matrix.parquet` have 119,932 labels on 44 datasets (95,453 shared with the old KB, 24,479 new). The
reliability of the mean of 10 repeats on the add-feature candidates (from `label_repeats`, per dataset in
`results/label_reliability/lightgbm_cv_kb_reliability.csv`) has a median ICC of 0.84 and a split-half Spearman of 0.72
(0.84 after Spearman-Brown); 36/44 datasets have ICC ≥ 0.5. It is low on 167120 (numerai, no signal),
359935, 359949, 359956 and 359987. Per dataset, the new labels agree with the old ones with a median Spearman of 0.26,
about the old labels' own reliability. **Problem: 359987 (shuttle, 7 classes, some with 4–5 rows)**: LightGBM multiclass diverges at random (fold log loss of
the original data jumps between 0.003 and 4.3), giving labels down to −1039; 543/576 of its labels are < −0.5. These
labels are noise and must be dropped or relabelled (e.g. with the rare classes merged, or a smaller learning rate) before the KB is used.

**Fix and full rerun (2026-10-27 afternoon):** `cv_losses` now uses `min_sum_hessian_in_leaf=1.0` for every dataset
(no special case): shuttle's fold log loss becomes stable (0.003), on 146818 the original loss even drops (0.369 vs
0.392) and the per-repeat labels vary less; no effect on regression. The first KB was moved to
`data/lightgbm_cv_unstable_v1/`. The whole pipeline was resubmitted (labels → metafeatures → MetaFE 100/250/500/1000 s →
evaluation → plots): jobs 30103248 (labels) … 30103275 (plots), same venv/cache as above. The ag12 venv got
catboost 1.2.8 and setuptools < 80 (AutoGluon 1.2 imports `pkg_resources`). Original/OpenFE/TabPrep are evaluated
again in this venv instead of copied from `lightgbm`: the cached test score of 146818 fold 0 differs in the 3rd
decimal (-0.2960 cached vs -0.2967 here, validation equal). Results: `results/lightgbm_cv/plots/`.

**Result (2026-10-28, all jobs done; plots rerun locally after installing adjustText 1.3.0 into ag12):** test error
change vs Original, median / mean / better-of-n: TabPrep +3.7 % / +4.4 % / 29 of 44 (p 0.006), OpenFE +1.1 % / −8.5 % /
25 of 36, MetaFE 100 s 0.0 % / +0.2 % / 16 of 44, 1000 s 0.0 % / −1.3 % / 15 of 44 (significantly worse, p 0.04).
Why: MetaFE's first pick has a median true label of −0.1 % (predicted +2.9 %), median percentile 45 among the
dataset's candidates (chance); every step is predicted ≥ 1 %, so 36/44 runs go to the time limit and build deep
GroupBy chains on ID columns. **LODO on the new labels** (`SURROGATE_CHECK_KB=lightgbm_cv
SURROGATE_CHECK_DIR=results/surrogate_check_cv`, see `experiments/surrogate_check/common.py`): the step-1 gate now
passes. Target-aware surrogate with rank target: median Spearman 0.31 (old labels 0.02), top-10 excess +0.24 % (oracle
+2.32 %), p 3e-6; the `describe()` surrogate as in MetaFE: Spearman 0.08; the residual heuristics have the best top 10
(+0.35 %). Write-up: https://claude.ai/artifact/T74Kew7xE8MJ1s4DbAC3Ve. Proposed next step: MetaFE v2
(`step3_metafe_v2.py`, ranker `sur_new_rank`, with CV verification of every pick) next to the heuristic with the same verification.

**Two follow-up tests (2026-10-29; MetaFE v2 not run, the user doubts it can help):** both run locally in minutes
(analysis venv, same env variables as above).
- `experiments/surrogate_check/within_dataset.py`: surrogate trained on 4/5 of a dataset's own candidates, tested on
  the rest. Median Spearman 0.81 (random folds) / 0.74 (grouped by first operand), every dataset ≥ 0.43; top-10
  excess +1.39 % (oracle +2.32 %); LODO `sur_new_rank` on the same candidates 0.31. Label ceiling (split-half of the
  10 repeats, Spearman-Brown) 0.89. So neither the labels nor the metafeatures are the bottleneck: candidate quality
  does not transfer across datasets.
- `experiments/surrogate_check/dataset_selection.py`: dataset-level KB (44 rows: size, columns, classes, categorical
  share, baseline difficulty, max/q90 residual gain of the candidates → test gain of TabPrep/OpenFE), random-forest
  selector LODO. Original vs TabPrep: meta mean +3.9 % (always TabPrep +4.4 %, oracle +8.9 %, permutation p 0.36); it
  only skips amazon and brazilian, both wrongly. Choosing by the validation score (no meta-learning): +7.6 %, catches
  kr_vs_kp and shuttle. With OpenFE (36 datasets): meta +2.7 % (p 0.21); validation −4.1 % (picks OpenFE on
  click_pred/okcupid, which fail on test). Dataset-level meta-learning does not work with 44 datasets.
- 2026-10-29: the user stopped the MetaFE experiments ("MetaFE as described in the paper is not working") and plans
  an ESANN 2027 paper (deadline 18 Nov 2026) on why it fails. `experiments/surrogate_check/learning_curve.py`: 25
  labels of the new dataset rank as well as the KB of 43 datasets (Spearman 0.30), 100 labels 0.58; the KB surrogate
  as an extra input adds <= 0.015. Paper figures, captions and all numbers: `experiments/paper/figures.py` ->
  `results/paper/` (README.md there).

## 00. Start here (2026-10-25 evening)

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

### Why MetaFE failed with 0.01, and the third run (2026-10-26 morning)

The catastrophic datasets (wilt −1157 %, car, phoneme) came from GroupBy features with a **continuous group key**:
on the training rows almost every key value is its own group, on the test rows almost every key value is unseen,
so the features were 86–89 % NaN on the test rows (fit-on-train mapping of group statistics). Two changes:
1. `candidate_featurenames(X)` allows GroupBy operators only with a categorical (non-numeric or boolean) group key,
   like TabPrep (`GroupByFeatureGenerator`: keys = category/object columns) and OpenFE.
2. Unseen numeric group keys on the test rows get the statistic of the nearest training key value (not
   transductive). With change 1 this only matters for features built before.
(A replay of the logged steps with change 2 alone fixed wilt: test log loss 0.711 → 0.073, Original 0.065.)

Third run, submitted 2026-10-26 07:15 (jobs 77128–77304): knowledge base rebuilt without numeric-key GroupBy labels,
MetaFE with `MIN_IMPROVEMENT` 0.01 and time limits 100–1000 s (calculco deadline), LightGBM on 3 seeds. Results in
`results/lightgbm_reuse/`. Earlier runs are kept: `results/lightgbm_reuse_min0.1/`,
`results/lightgbm_reuse_min0.01_testnan/`, `data/lightgbm_reuse_numeric_keys/`. `metafe.metafe --replay` rebuilds
MetaFE files from the step logs after changes that only affect the test rows.

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
