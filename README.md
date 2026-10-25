# MetaFE: model-aware feature engineering with a meta-learned surrogate

MetaFE engineers features for a tabular dataset without training the downstream model for every candidate. A
surrogate model, trained on other datasets, predicts how much each candidate feature would improve a given model,
and MetaFE applies the best candidates recursively until no candidate is predicted to help or the time is up.

- **Candidates:** the OpenFE operator set on the current columns (`add(a, b)`, `GroupByThenMean(a, b)`,
  `freq(a)`, …) and deleting a column (`without - a`), see `metafe/operators.py`.
- **Label** of a candidate on a benchmark dataset and model: relative improvement of the model's AutoGluon validation
  score (8-fold bagging, out-of-fold) when the feature is added or the column removed.
- **Metafeatures** of a candidate: pandas `describe()` of the feature on the training rows.
- **Surrogate:** CatBoost regressor from metafeatures (+ operator, model, task type) to label, trained
  leave-one-dataset-out on the 44 benchmark datasets of `metafe/config.py`, on the labels of all models. MetaFE asks
  it about one target model.

All operators are fitted on the training rows and applied to the training **and** test rows, and they do not
depend on the row order. (The first implementation leaked the target through the row order and never applied
features to the test rows; see `README_METAFE_HANDOFF.md`.)

## Experiments

Everything that depends on the models is an experiment in `metafe/config.py` (`--experiment` of every module):

| | `lightgbm` | `multimodel` |
|---|---|---|
| Models | LightGBM | Linear model, RandomForest, LightGBM, TabICLv2, RealMLP |
| AutoGluon / venv | 1.2.1, `~/venvs/metafe` (CPU) | 1.6.3, `~/venvs/metafe-ag16` (TabICL, RealMLP on GPU) |
| Labels | all candidates; old LightGBM labels reused where unchanged (75–95 %) | ≤ `max_candidates` per dataset (balanced over operators, same for all models), ≤ `max_label_rows` training rows |
| MetaFE | one run per dataset | one run per dataset and target model |
| Evaluation | LightGBM; Original/OpenFE/TabPrep results are cached, only MetaFE is fitted | each model on Original/OpenFE/TabPrep, MetaFE for model m with model m; plots per model and averaged over the models |
| Submit | `scripts/submit_lightgbm.sh` | `scripts/submit_pilot.sh`, then `scripts/submit_multimodel.sh` |

A third experiment, `lightgbm_reuse`, is `lightgbm` without any new label: only the reusable old labels (operators
without one, mainly `Combine`/`CombineThenFreq`, are never proposed) and MetaFE up to 1 h. It needs no model fits
before MetaFE and runs in hours on calculco with OAR (`scripts/submit_oar.sh`, see `README_METAFE_HANDOFF.md` §00).

**Order:** run `lightgbm` first. It completes the OpenFE/TabPrep experiments with the same setup and shows whether
MetaFE works before the expensive labels of the other models are computed. Then run the `multimodel` pilot, set
`max_candidates` / `max_label_rows` from its estimate, and run `multimodel`.

## Repository layout

```
metafe/                      the method (python package)
  config.py                  paths, datasets, time limits, AutoGluon resources, experiments
  data.py                    OpenML split loading, label rows, FE file format (train rows, then test rows; "target")
  operators.py               feature operators, feature names, candidates
  legacy_operators.py        the first (leaky) operators, only to reuse old labels
  models.py                  the five models and the AutoGluon fits (labels and evaluation)
  core_matrix.py             step 1: labels                 -> data/<exp>/core_matrix/<model>/<id>.parquet
  metafeatures.py            step 2: metafeatures           -> data/<exp>/metafeature_matrix[/<id>].parquet
  surrogate.py               step 3: CatBoost surrogate
  metafe.py                  step 4: recursive MetaFE       -> data/<exp>/fe/MetaFE_<model>/FE_<id>_MetaFE_<t>.parquet
  pilot.py                   timing pilot and compute estimate (multimodel) -> results/<exp>/pilot/
  evaluation/
    baselines.py             OpenFE / TabPrep data          -> data/fe/{OpenFE,TabPrep}/
    check_fe.py              sanity checks of FE data
    evaluate.py              AutoGluon on every FE dataset  -> results/<exp>/evaluation/<model>/
    merge_folds.py           mean/std over the 3 seeds      -> results/<exp>/evaluation/Result_<id>.parquet
    plots.py                 plots, summary, LaTeX tables   -> results/<exp>/plots/{<model>,average}/
tests/test_operators.py      operator self-test (order independence, test rows, reuse of old labels)
scripts/run_step.sh          runs one step (venv, array index -> dataset/fold)
scripts/submit_slurm.sh      submits the pipeline of an experiment (used by the two scripts below)
scripts/submit_lightgbm.sh   experiment lightgbm
scripts/submit_pilot.sh      multimodel timing pilot
scripts/submit_multimodel.sh experiment multimodel
scripts/setup_venv_multimodel.sh   venv of the multimodel experiment (+ TabICLv2 checkpoints, OpenML cache)
scripts/submit_oar.sh        experiment lightgbm_reuse on calculco (OAR), with oar_step.sh, build_knowledge_base.sh
data/fe/{OpenFE,TabPrep}/    baseline FE data (valid, original OpenML split, model-independent)
data/legacy/                 first core matrix and pandas matrix (leaky labels), first MetaFE data (invalid)
results/lightgbm/evaluation/GBM/   cached valid Original/OpenFE/TabPrep LightGBM results (AutoGluon 1.2.1, seeds 0-2)
results/legacy/              earlier results and plots (MetaFE parts invalid)
```

Data and results are Git LFS files: `git lfs pull` after cloning.

## Running it

Environments: `~/venvs/metafe` for `lightgbm` (see `README_METAFE_HANDOFF.md` §6) and
`scripts/setup_venv_multimodel.sh` for `multimodel` (set `TORCH_INDEX` to the cluster's CUDA version). Every module
checks that the installed AutoGluon matches the experiment. Run everything from the repository root with
`PYTHONPATH=.`; every module has `--help`.

```bash
E="--experiment lightgbm"                       # or multimodel (then MetaFE/evaluate need --model)
# 0. checks (minutes, no model training)
python -m tests.test_operators
python -m metafe.core_matrix build $E --dataset 146818 --dry_run   # labels to reuse / to compute
# (multimodel) timing pilot on metafe.config.PILOT_DATASETS, then the estimate
python -m metafe.pilot run --experiment multimodel --dataset 168757 --model TABICL
python -m metafe.pilot summary --experiment multimodel
# 1. labels (one AutoGluon fit per missing label; resumable)
python -m metafe.core_matrix build $E --dataset <id> [--models GBM]
python -m metafe.core_matrix merge $E
# 2. metafeatures
python -m metafe.metafeatures build $E --dataset <id>
python -m metafe.metafeatures merge $E
# 3+4. MetaFE: one run writes the data for all time limits (100 ... 7200 s)
python -m metafe.metafe $E --dataset <id> [--model GBM]
# 5. evaluation
python -m metafe.evaluation.check_fe $E
python -m metafe.evaluation.evaluate $E --dataset <id> --fold <0|1|2> [--model GBM]
python -m metafe.evaluation.merge_folds $E
python -m metafe.evaluation.plots $E
```

On SLURM, the submit scripts submit all steps as dependent array jobs (one task per dataset, or per dataset and fold;
one job array per model). Try `DRY_RUN=1` first; `FROM=<stage>` / `UNTIL=<stage>` run a part, `DATASETS`/`MODELS`
restrict it, `SBATCH_OPTS` (all jobs), `CPU_OPTS` (CPU jobs) and `GPU_OPTS` (GPU jobs, default `--gres=gpu:1`) add
cluster options, `MODULES` loads modules in every job.

```bash
DRY_RUN=1 scripts/submit_lightgbm.sh
CPU_OPTS="--partition=cpu" scripts/submit_lightgbm.sh
# multimodel
GPU_OPTS="--partition=gpu --gres=gpu:1" CPU_OPTS="--partition=cpu" scripts/submit_pilot.sh
python -m metafe.pilot summary --experiment multimodel        # then adjust max_candidates / max_label_rows
GPU_OPTS="--partition=gpu --gres=gpu:1" CPU_OPTS="--partition=cpu" scripts/submit_multimodel.sh
```

| Stage | Jobs (44 datasets) | Resources per job |
|---|---|---|
| core matrix | 44 per model | 8 CPUs (+1 GPU for TabICL/RealMLP), 64 GB, `CORE_TIME` (default 3 days; resubmit to continue) |
| metafeatures | 44 | 2 CPUs, 32 GB, 6 h |
| MetaFE | 44 per model | 8 CPUs, 72 GB, `METAFE_TIME` (default 4 h = 2 h limit + surrogate training) |
| evaluation | 132 per model | 8 CPUs (+1 GPU for TabICL/RealMLP), 64 GB, `EVAL_TIME` (default 12 h) |
| pilot | 4 per model | 8 CPUs (+1 GPU), 64 GB, `PILOT_TIME` (default 12 h) |

## Design decisions to know

- **Labels are reused** (experiment `lightgbm` only) from `data/legacy/core_matrix` where the fixed operators give the
  same training feature (a label depends only on the training rows); typically 75–95 % of the labels.
- **Label sample** (experiment `multimodel`): at most `max_candidates` candidates per dataset, round robin over the
  operators, the same candidates for every model; datasets with more than `max_label_rows` training rows are labelled
  on a fixed stratified row sample (also used for their metafeatures). The evaluation always uses all rows.
- **A model that fails** on a dataset's original data (e.g. a size limit) is skipped for that dataset; the average
  over the models then uses the other models.
- **Averaging over models:** per dataset, the improvement over Original of each model, then the mean over the models;
  for MetaFE, model m's improvement comes from the MetaFE data for model m.
- **MetaFE time** is counted from the end of the surrogate training (the surrogate is trained once per dataset,
  leave-one-out; its training time is logged). `--count_surrogate_training` counts it as well. MetaFE never fits
  the target model, so its time limits mean the same for every model.
- **MetaFE never re-applies** a feature it added or deleted before, deletes only original columns, and skips
  candidates that duplicate an existing column (all three made earlier runs loop or stall).
- **Evaluation folds** are AutoGluon seeds 0–2 on the fixed OpenML split; seed 0 reproduces the earlier
  results bit-exactly, so the cached LightGBM results stay valid. MetaFE files that are identical for several time
  limits are fitted once.
- **The surrogate's inputs** are all matrix columns except the label, as in the first experiments, including
  `dataset - id` and `feature - name`.
