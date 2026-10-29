"""Within-dataset check: can the target-aware metafeatures rank a dataset's candidates when the surrogate is trained on
other candidates of the SAME dataset? Separates "the metafeatures cannot describe candidate quality" from "candidate
quality does not transfer across datasets" (step1_lodo trains on the other datasets only).

Per dataset (add-feature candidates, as in step1_lodo): 5-fold CV over the candidates, LightGBM surrogate on NEW_MF +
operator with the per-dataset percentile of the label as target. Two fold splits:
  random   candidates assigned at random (candidates sharing an operand can be in train and test)
  operand  grouped by the first operand of the candidate (no operand of a test candidate's first slot is seen)
Ceiling: Spearman between the 10-repeat label and its true value, estimated from the split-half Spearman r of repeats
0-4 vs 5-9 (Spearman-Brown: reliability of the 10-repeat mean = 2r/(1+r), ceiling = sqrt of it).
Compared with sur_new_rank of step1_lodo (scores.parquet of the same OUT_DIR).

    SURROGATE_CHECK_KB=lightgbm_cv SURROGATE_CHECK_DIR=results/surrogate_check_cv \
        python experiments/surrogate_check/within_dataset.py [--workers 16]
Output: <OUT_DIR>/within_dataset/{per_dataset.csv, summary.csv}
"""
import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, KB_MATRIX, NEW_MF  # noqa: E402
from step1_lodo import LABEL, TOP_K, metrics  # noqa: E402

COLUMNS = NEW_MF + ["operator"]
N_SPLITS = 5


def load_kb():
    kb = pd.read_parquet(KB_MATRIX, columns=["dataset - id", "feature - name", "operator", LABEL])
    kb = kb[kb["operator"] != "delete"]
    mf = pd.concat([pd.read_parquet(p) for p in sorted((OUT_DIR / "mf_v2").glob("*.parquet"))])
    kb = kb.merge(mf, on=["dataset - id", "feature - name"], how="left", validate="one_to_one")
    kb["operator"] = kb["operator"].astype("category")
    kb["first_operand"] = kb["feature - name"].map(first_operand)
    return kb.reset_index(drop=True)


def first_operand(name):
    """First top-level argument of a candidate name "op(a, b)" (a and b may be nested candidates)."""
    inner, depth = name[name.index("(") + 1:-1], 0
    for i, ch in enumerate(inner):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            return inner[:i].strip()
    return inner.strip()


def ceiling(dataset_id, names):
    path = KB_MATRIX.parent / "label_repeats" / "GBM" / f"{dataset_id}.parquet"
    rep = pd.read_parquet(path).set_index("feature - name").reindex(names)
    a = rep[[f"repeat_{i}" for i in range(5)]].mean(axis=1)
    b = rep[[f"repeat_{i}" for i in range(5, 10)]].mean(axis=1)
    r = spearmanr(a, b, nan_policy="omit").correlation
    return r, np.sqrt(max(2 * r / (1 + r), 0)) if r > -1 else np.nan


def within(dataset_id):
    import lightgbm as lgb

    d = KB[KB["dataset - id"] == dataset_id].reset_index(drop=True)
    target = d[LABEL].rank(pct=True)
    out = d[["dataset - id", "feature - name", "operator", LABEL]].copy()
    n_groups = d["first_operand"].nunique()  # datasets with few columns: fewer operand folds
    splits = {"random": KFold(N_SPLITS, shuffle=True, random_state=0).split(d),
              "operand": GroupKFold(min(N_SPLITS, n_groups)).split(d, groups=d["first_operand"])}
    for name, split in splits.items():
        pred = np.full(len(d), np.nan)
        for tr, te in split:
            model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0,
                                      verbose=-1, n_jobs=1)
            model.fit(d.loc[tr, COLUMNS], target[tr])
            pred[te] = model.predict(d.loc[te, COLUMNS])
        out[f"within_{name}"] = pred
    r, ceil = ceiling(dataset_id, d["feature - name"])
    return out, dict(dataset=dataset_id, split_half_spearman=r, ceiling_spearman=ceil,
                     n_operands=d["first_operand"].nunique())


def init(kb):
    global KB
    KB = kb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out_dir = OUT_DIR / "within_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = load_kb()
    ids = sorted(kb["dataset - id"].unique())
    with multiprocessing.get_context("fork").Pool(a.workers, initializer=init, initargs=(kb,)) as pool:
        results = pool.map(within, ids)
    scored = pd.concat([r[0] for r in results])
    ceil = pd.DataFrame([r[1] for r in results])
    lodo = pd.read_parquet(OUT_DIR / "step1_lodo" / "scores.parquet",
                           columns=["dataset - id", "feature - name", "sur_new_rank", "heu_eta_res_gain"])
    scored = scored.merge(lodo, on=["dataset - id", "feature - name"], how="left", validate="one_to_one")
    rankers = ["within_random", "within_operand", "sur_new_rank", "heu_eta_res_gain"]
    rows = [dict(dataset=d, ranker=r, **metrics(f, r)) for d, f in scored.groupby("dataset - id") for r in rankers]
    per = pd.DataFrame(rows).merge(ceil, on="dataset")
    per.to_csv(out_dir / "per_dataset.csv", index=False)

    lodo_sp = per[per["ranker"] == "sur_new_rank"].set_index("dataset")["spearman"]
    summary = []
    for r, g in per.groupby("ranker"):
        g = g.set_index("dataset")
        diff = (g["spearman"] - lodo_sp).dropna()
        summary.append(dict(
            ranker=r, datasets=len(g), median_spearman=g["spearman"].median(),
            median_spearman_nz=g["spearman_nz"].median(), median_top10_excess=g["top10_excess"].median(),
            mean_top10_excess=g["top10_excess"].mean(),
            median_oracle_top10_excess=g["oracle_top10_excess"].median(),
            datasets_spearman_above_lodo=int((diff > 0).sum()),
            wilcoxon_p_vs_lodo=wilcoxon(diff).pvalue if r != "sur_new_rank" else np.nan))
    summary = pd.DataFrame(summary).sort_values("median_spearman", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
        print(f"\nlabel ceiling: median split-half Spearman {ceil['split_half_spearman'].median():.3f}, "
              f"median ceiling Spearman {ceil['ceiling_spearman'].median():.3f}")


if __name__ == "__main__":
    main()
