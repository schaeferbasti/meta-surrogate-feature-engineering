"""Paths and settings shared by all MetaFE steps.

Everything that depends on the models lives in an experiment (EXPERIMENTS, chosen with --experiment):
  lightgbm    LightGBM only, AutoGluon 1.2 (CPU). Completes the experiments with OpenFE and TabPrep: their cached
              evaluation results were computed with this setup, and the LightGBM labels of the first core matrix are
              reused where the training feature is unchanged (the other labels are computed).
  lightgbm_reuse  Like lightgbm, but without any new label: only the reusable old labels (75-93 %), MetaFE up to
              1 h, MIN_IMPROVEMENT 0.01 (with 0.1, MetaFE took no step on 21 of 44 datasets: only 0.5 % of the
              clean labels are >= 0.1). Needs no AutoGluon fits before MetaFE, so it runs in hours (calculco,
              scripts/submit_oar.sh).
              Operators without any reusable label (Combine, CombineThenFreq) are never proposed by MetaFE.
  multimodel  Linear model, random forest, LightGBM, TabICLv2 and RealMLP, AutoGluon 1.6 (TabICL and RealMLP on GPU).
              MetaFE runs once per target model; the evaluation uses all models and averages over them.

Layout (the environment variables METAFE_DATA_DIR / METAFE_RESULTS_DIR move data/ and results/, e.g. for tests):
  data/fe/{OpenFE,TabPrep}/FE_<id>_<method>.parquet                  baselines (model-independent, shared)
  data/<experiment>/core_matrix/<model>/<id>.parquet                 1. labels
  data/<experiment>/metafeature_matrix[/<id>].parquet                2. surrogate training data
  data/<experiment>/fe/MetaFE_<model>/FE_<id>_MetaFE_<t>.parquet     4. MetaFE data per target model and time limit
  results/<experiment>/pilot/                                        timing pilot (multimodel)
  results/<experiment>/evaluation/<model>/<method>_Result_<id>_<fold>.parquet, Result_<id>.parquet
  results/<experiment>/plots/<model name>/, plots/average/
"""
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("METAFE_DATA_DIR", REPO / "data"))
RESULTS_DIR = Path(os.environ.get("METAFE_RESULTS_DIR", REPO / "results"))

# Feature-engineered data of the baselines: <BASELINE_FE_DIR>/<method>/FE_<id>_<method>.parquet
BASELINE_FE_DIR = DATA_DIR / "fe"
BASELINES = ["OpenFE", "TabPrep"]
# Old (leaky) LightGBM labels of the first core matrix (experiment lightgbm reuses them where possible)
LEGACY_CORE_MATRIX_DIR = DATA_DIR / "legacy" / "core_matrix"

TARGET = "target"

# MetaFE: time limits (s) of one run; a FE file is written for every limit
TIME_LIMITS = [100, 250, 500, 1000, 1800, 3600, 7200]
# MetaFE: stop when the best candidate's predicted relative improvement is below this value
MIN_IMPROVEMENT = 0.1

# AutoGluon resources of every fit (core matrix and evaluation); GPUs are used if the job has one (metafe.models)
NUM_CPUS = 8
MEMORY_LIMIT_GB = 48
FIT_TIME_LIMIT_S = 4 * 60 * 60

# Evaluation: number of repeats ("folds") on the fixed OpenML split; fold k uses AutoGluon seed k
FOLDS = [0, 1, 2]


@dataclass(frozen=True)
class Experiment:
    name: str
    models: tuple                       # AutoGluon model keys (metafe.models): labels, MetaFE targets, evaluation
    autogluon: str                      # required AutoGluon version (prefix), so that results are never mixed
    max_candidates: int | None = None   # labelled candidates per dataset (balanced over operators); None = all
    max_label_rows: int | None = None   # training rows used for labels and metafeatures (stratified); None = all
    reuse_legacy_labels: bool = False   # reuse first-core-matrix LightGBM labels where the training feature is unchanged
    compute_labels: bool = True         # False: only reused labels, candidates without one are left out
    time_limits: tuple = tuple(TIME_LIMITS)  # MetaFE time limits (s); one run writes the data of every limit
    baseline_results_from: str | None = None  # experiment whose Original/OpenFE/TabPrep results are reused
    min_improvement: float = MIN_IMPROVEMENT  # MetaFE stops when no candidate is predicted to improve by this much

    @property
    def core_matrix_dir(self):
        return DATA_DIR / self.name / "core_matrix"

    @property
    def core_matrix(self):
        return DATA_DIR / self.name / "core_matrix.parquet"

    @property
    def metafeature_matrix_dir(self):
        return DATA_DIR / self.name / "metafeature_matrix"

    @property
    def metafeature_matrix(self):
        return DATA_DIR / self.name / "metafeature_matrix.parquet"

    def metafe_dir(self, model_key):
        return DATA_DIR / self.name / "fe" / f"MetaFE_{model_key}"

    @property
    def pilot_dir(self):
        return RESULTS_DIR / self.name / "pilot"

    @property
    def evaluation_dir(self):
        return RESULTS_DIR / self.name / "evaluation"

    @property
    def plots_dir(self):
        return RESULTS_DIR / self.name / "plots"


EXPERIMENTS = {
    "lightgbm": Experiment("lightgbm", models=("GBM",), autogluon="1.2", reuse_legacy_labels=True),
    "lightgbm_reuse": Experiment("lightgbm_reuse", models=("GBM",), autogluon="1.2", reuse_legacy_labels=True,
                                 compute_labels=False, time_limits=(100, 250, 500, 1000, 1800, 3600),
                                 baseline_results_from="lightgbm", min_improvement=0.01),
    # max_candidates / max_label_rows: set from the pilot (python -m metafe.pilot summary)
    "multimodel": Experiment("multimodel", models=("LR", "RF", "GBM", "TABICL", "REALMLP"), autogluon="1.6",
                             max_candidates=500, max_label_rows=50_000),
}
DEFAULT_EXPERIMENT = "lightgbm"


def add_experiment_argument(parser):
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default=DEFAULT_EXPERIMENT,
                        help=f"Experiment (default {DEFAULT_EXPERIMENT}), see metafe/config.py")


# Datasets on which OpenFE did not finish
NO_OPENFE_DATASETS = [2073, 359930, 359931, 359935, 359938, 359959, 359962, 359983]

# OpenML task IDs of the benchmark (classification and regression) and their dataset names
DATASETS = {
    146818: "australian", 146820: "wilt", 167120: "numerai28.6", 167210: "moneyball", 168350: "phoneme",
    168757: "credit-g", 168784: "steel", 189354: "airlines", 190146: "vehicle", 2073: "yeast", 233211: "diamonds",
    359930: "quake", 359931: "sensory", 359932: "socmob", 359933: "space_ga", 359935: "wine_quality",
    359936: "elevators", 359937: "black_friday", 359938: "brazilian", 359944: "abalone", 359949: "house_sales",
    359950: "boston", 359952: "house_16H", 359954: "eucalyptus", 359955: "blood", 359956: "qsar_biodeg",
    359958: "pc4", 359959: "cmc", 359960: "car", 359962: "kc1", 359963: "segment", 359965: "kr_vs_kp",
    359968: "churn", 359971: "pishing", 359972: "sylvine", 359974: "wine_white", 359975: "satellite",
    359979: "amazon", 359981: "jungle_chess", 359982: "bank", 359983: "adult", 359987: "shuttle",
    359992: "click_pred", 359993: "okcupid",
}
# Pilot datasets (multimodel): small, medium and large classification, and a regression dataset
PILOT_DATASETS = [168757, 359968, 359983, 359952]
