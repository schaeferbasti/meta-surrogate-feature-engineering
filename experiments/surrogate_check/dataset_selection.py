"""Dataset-level meta-learning: choose the FE method per dataset (instead of ranking candidates).

Knowledge base: one row per dataset. Inputs = dataset metafeatures (from mf_v2: size, columns, classes, categorical
share, baseline difficulty, task type, and how strongly the best candidates explain the LightGBM residual), targets =
test gain (%) of each FE method over Original (results/<experiment>/evaluation/GBM, mean of the 3 folds).

Selectors, scored by the test gain of the method they choose on the held-out dataset:
  always_<m>     always the same method
  validation     the method with the best validation score on the dataset itself (no meta-learning)
  meta           random forest per method predicting its gain (clipped to +-CLIP %) from the other datasets (LODO);
                 the method with the highest predicted gain, Original if none is predicted > 0
  oracle         the best method per dataset (test)
Two settings: "TabPrep" (Original vs TabPrep, 44 datasets), "TabPrep+OpenFE" (36 datasets where OpenFE finished).
Permutation test of meta: the gains are shuffled across datasets (the inputs stay), the LODO repeated; p = share of
permutations with a mean gain >= the real one.

    SURROGATE_CHECK_KB=lightgbm_cv SURROGATE_CHECK_DIR=results/surrogate_check_cv \
        python experiments/surrogate_check/dataset_selection.py [--experiment lightgbm_cv] [--permutations 500]
Output: <OUT_DIR>/dataset_selection/{dataset_table.csv, per_dataset.csv, summary.csv}
"""
import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR  # noqa: E402
from metafe import config  # noqa: E402

CLIP = 20.0
DATASET_MF = ["n_rows_log", "n_cols", "n_classes", "frac_cat_cols", "baseline_difficulty"]  # n_classes 0: regression
CANDIDATE_AGG = ["eta_res_gain", "mi_y_gain"]  # max and 90 % quantile over the candidates of the dataset
SETTINGS = {"TabPrep": ["TabPrep"], "TabPrep+OpenFE": ["TabPrep", "OpenFE"]}


def gains(experiment):
    """Gain (%) over Original per dataset and method, for the validation and the test score (mean over folds)."""
    files = sorted(config.EXPERIMENTS[experiment].evaluation_dir.glob("GBM/*_Result_*_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files])
    mean = df.groupby(["dataset", "origin"])[["score_val", "score_test"]].mean().unstack("origin")
    out = {}
    for split in ["val", "test"]:
        err = -mean[f"score_{split}"]  # scores are negative losses
        out[split] = err.drop(columns="Original").apply(lambda c: 100 * (err["Original"] - c) / err["Original"])
    return out


def dataset_features():
    rows = []
    for p in sorted((OUT_DIR / "mf_v2").glob("*.parquet")):
        mf = pd.read_parquet(p)
        row = {"dataset": int(mf["dataset - id"].iloc[0]), **mf[DATASET_MF].iloc[0].to_dict()}
        for c in CANDIDATE_AGG:
            row[f"{c}_max"] = mf[c].max()
            row[f"{c}_q90"] = mf[c].quantile(0.9)
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")


def meta_choice(X, G, methods, seed=0):
    """LODO choice per dataset: method with the highest predicted (clipped) gain, Original if none > 0."""
    choice = pd.Series("Original", index=X.index)
    for d in X.index:
        train = X.index != d
        pred = {}
        for m in methods:
            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, random_state=seed,
                                       n_jobs=1)
            rf.fit(X[train], G.loc[train, m].clip(-CLIP, CLIP))
            pred[m] = rf.predict(X.loc[[d]])[0]
        best = max(pred, key=pred.get)
        if pred[best] > 0:
            choice[d] = best
    return choice


def realised(choice, G):
    return pd.Series([0.0 if m == "Original" else G.at[d, m] for d, m in choice.items()], index=choice.index)


def permuted(args):
    X, G, methods, seed = args
    perm = np.random.RandomState(seed).permutation(len(G))
    Gp = pd.DataFrame(G.to_numpy()[perm], index=G.index, columns=G.columns)
    return realised(meta_choice(X, Gp, methods), Gp).mean()


def stats(name, setting, g):
    return dict(setting=setting, selector=name, datasets=len(g), mean_gain=g.mean(), median_gain=g.median(),
                better=int((g > 0.05).sum()), worse=int((g < -0.05).sum()), worse_5pct=int((g < -5).sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="lightgbm_cv")
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--workers", type=int, default=14)
    a = p.parse_args()
    out_dir = OUT_DIR / "dataset_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    g = gains(a.experiment)
    X = dataset_features()
    X = X.loc[g["test"].index]
    table = X.join(g["test"].add_prefix("test_")).join(g["val"].add_prefix("val_"))
    table.insert(0, "name", [config.DATASETS[d] for d in table.index])
    table.to_csv(out_dir / "dataset_table.csv")

    summary, per = [], []
    for setting, methods in SETTINGS.items():
        G = g["test"][methods].dropna()
        V = g["val"].loc[G.index, methods]
        Xs = X.loc[G.index]
        choices = {f"always_{m}": pd.Series(m, index=G.index) for m in ["Original"] + methods}
        choices["validation"] = V.apply(lambda r: r.idxmax() if r.max() > 0 else "Original", axis=1)
        choices["meta"] = meta_choice(Xs, G, methods)
        choices["oracle"] = G.apply(lambda r: r.idxmax() if r.max() > 0 else "Original", axis=1)
        for name, choice in choices.items():
            r = realised(choice, G)
            summary.append(stats(name, setting, r))
            per.append(pd.DataFrame({"setting": setting, "selector": name, "dataset": G.index,
                                     "name": [config.DATASETS[d] for d in G.index], "choice": choice.values,
                                     "gain": r.values}))
        real = realised(choices["meta"], G).mean()
        with multiprocessing.Pool(a.workers) as pool:
            null = np.array(pool.map(permuted, [(Xs, G, methods, s) for s in range(1, a.permutations + 1)]))
        for row in summary:
            if row["setting"] == setting and row["selector"] == "meta":
                row["perm_p"] = float((null >= real).mean())
                row["perm_null_mean"] = float(null.mean())
                row["perm_null_q95"] = float(np.quantile(null, 0.95))
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.concat(per).to_csv(out_dir / "per_dataset.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
