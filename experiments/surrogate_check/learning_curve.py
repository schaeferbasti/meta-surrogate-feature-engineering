"""Learning curve of a surrogate trained on labels of the target dataset itself, and whether the knowledge base of
the other datasets still helps when only a few target labels exist.

Per dataset (add-feature candidates): a fixed test set (TEST_SHARE of the candidates, seed 0); from the rest, n labelled
candidates are drawn (n in N_LABELS, SEEDS draws). Rankers, scored by the Spearman with the label on the test set:
  local       LightGBM on NEW_MF + operator, trained on the n target labels (percentile target)
  local+kb    the same plus the LODO score sur_new_rank as an input (trained on the other 43 datasets only), i.e. the
              knowledge base as a warm start
  kb          sur_new_rank alone (n = 0, from step1_lodo)
Ceiling: the label-noise ceiling of within_dataset.py.

    SURROGATE_CHECK_KB=lightgbm_cv SURROGATE_CHECK_DIR=results/surrogate_check_cv \
        python experiments/surrogate_check/learning_curve.py [--workers 14]
Output: <OUT_DIR>/learning_curve/{per_dataset.csv, summary.csv}
"""
import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, NEW_MF  # noqa: E402
from step1_lodo import LABEL  # noqa: E402
from within_dataset import load_kb  # noqa: E402

N_LABELS = [10, 25, 50, 100, 200, 400]
SEEDS = range(5)
TEST_SHARE = 0.3
COLUMNS = NEW_MF + ["operator"]


def fit_predict(train, test, columns, seed):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=min(5, max(2, len(train) // 5)),  # n = 10: 2, else 5
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                              n_jobs=1)
    model.fit(train[columns], train[LABEL].rank(pct=True))
    return model.predict(test[columns])


def curve(dataset_id):
    d = KB[KB["dataset - id"] == dataset_id].reset_index(drop=True)
    rng = np.random.RandomState(0)
    is_test = np.zeros(len(d), bool)
    is_test[rng.choice(len(d), int(TEST_SHARE * len(d)), replace=False)] = True
    test, pool = d[is_test], d[~is_test]
    rows = [dict(dataset=dataset_id, n=0, seed=0, ranker="kb",
                 spearman=spearmanr(test["sur_new_rank"], test[LABEL]).correlation)]
    for n in N_LABELS:
        if n > len(pool):
            continue
        for seed in SEEDS:
            train = pool.sample(n, random_state=seed)
            for ranker, columns in [("local", COLUMNS), ("local+kb", COLUMNS + ["sur_new_rank"])]:
                pred = fit_predict(train, test, columns, seed)
                sp = spearmanr(pred, test[LABEL]).correlation if np.std(pred) > 0 else 0.0
                rows.append(dict(dataset=dataset_id, n=n, seed=seed, ranker=ranker, spearman=sp))
    return rows


def init(kb):
    global KB
    KB = kb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out_dir = OUT_DIR / "learning_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = load_kb()
    lodo = pd.read_parquet(OUT_DIR / "step1_lodo" / "scores.parquet",
                           columns=["dataset - id", "feature - name", "sur_new_rank"])
    kb = kb.merge(lodo, on=["dataset - id", "feature - name"], how="left", validate="one_to_one")
    with multiprocessing.get_context("fork").Pool(a.workers, initializer=init, initargs=(kb,)) as pool:
        rows = [r for rs in pool.map(curve, sorted(kb["dataset - id"].unique())) for r in rs]
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_dataset.csv", index=False)
    # mean over draws per dataset, then median over the datasets that have this n
    by_ds = per.groupby(["ranker", "n", "dataset"])["spearman"].mean().reset_index()
    summary = by_ds.groupby(["ranker", "n"])["spearman"].agg(["median", "mean", "count"]).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.pivot(index="n", columns="ranker", values="median").round(3).to_string())
    print(summary.pivot(index="n", columns="ranker", values="count").to_string())


if __name__ == "__main__":
    main()
