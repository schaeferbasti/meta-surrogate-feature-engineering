"""Step 1b + step 2: leave-one-dataset-out (LODO) check of the surrogate with the old, the new (target-aware) and
both metafeature sets, and of the heuristic rankers without meta-learning (step 2).

For every held-out dataset, a LightGBM surrogate is trained on the knowledge base of the other 43 datasets and ranks
the held-out dataset's candidates. Metrics per dataset (on the add-feature candidates; deleting a column is not a
candidate of MetaFE v2, and the heuristics are meaningless for it):
  spearman, spearman_nz   Spearman of score and label (all labels / non-zero labels)
  top10, top10_excess     mean label of the 10 best-scored candidates, minus the dataset's mean label (= random pick)
  top10_median            median label of the top 10 (the handoff's metric; random: median label of the dataset)
Gate (handoff 000): top-10 picks clearly beat random (one-sided Wilcoxon over datasets of top10_excess, p < 0.05,
median excess > 0) and median spearman_nz > 0.15.

    python experiments/surrogate_check/step1_lodo.py [--workers 16]
Output: results/surrogate_check/step1_lodo/{per_dataset.csv, summary.csv, gate.json}
"""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, KB_MATRIX, OLD_MF, NEW_MF  # noqa: E402

LABEL = "improvement"
TOP_K = 10
GATE_P = 0.05
GATE_SPEARMAN_NZ = 0.15
FEATURE_SETS = {"old": OLD_MF + ["operator", "task_type"], "new": NEW_MF + ["operator"],
                "both": OLD_MF + NEW_MF + ["operator", "task_type"]}
# Surrogates: (feature set, target); target "rank" = per-dataset percentile of the label (scale-free across datasets)
SURROGATES = {"old": ("old", "label"), "new": ("new", "label"), "both": ("both", "label"),
              "new_rank": ("new", "rank"), "both_rank": ("both", "rank")}
# Step 2: rank by a single statistic (residual gain first = the handoff's heuristic)
HEURISTICS = ["eta_res_gain", "eta_res", "rho_res_gain", "mi_y_gain", "eta_y_gain"]


def load_kb():
    kb = pd.read_parquet(KB_MATRIX)
    mf = pd.concat([pd.read_parquet(p) for p in sorted((OUT_DIR / "mf_v2").glob("*.parquet"))])
    missing = set(kb["dataset - id"]) - set(mf["dataset - id"])
    if missing:
        raise SystemExit(f"mf_v2 missing for datasets {sorted(missing)}")
    kb = kb.merge(mf, on=["dataset - id", "feature - name"], how="left", validate="one_to_one")
    for c in ["operator", "task_type"]:
        kb[c] = kb[c].astype("category")
    kb["rank_label"] = kb.groupby("dataset - id")[LABEL].rank(pct=True)
    return kb.reset_index(drop=True)


def fit_surrogate(train, columns, target, seed=0):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
                              n_jobs=4)
    model.fit(train[columns], train["rank_label" if target == "rank" else LABEL])
    return model


def lodo_scores(dataset_id):
    """Scores of every ranker on the held-out dataset: frame (feature - name, label, operator, <ranker>...)."""
    kb = KB
    test_mask = kb["dataset - id"] == dataset_id
    train, test = kb[~test_mask], kb[test_mask]
    out = test[["dataset - id", "feature - name", "operator", LABEL]].copy()
    for name, (fs, target) in SURROGATES.items():
        model = fit_surrogate(train, FEATURE_SETS[fs], target)
        out[f"sur_{name}"] = model.predict(test[FEATURE_SETS[fs]])
    for h in HEURISTICS:
        out[f"heu_{h}"] = test[h].fillna(-np.inf).to_numpy()
    out["random"] = np.random.RandomState(dataset_id % 2 ** 31).rand(len(out))
    return out


def metrics(frame, score):
    y, s = frame[LABEL].to_numpy(), frame[score].to_numpy()
    order = np.argsort(-s, kind="stable")
    top = y[order[:TOP_K]]
    nz = y != 0
    sp = spearmanr(s, y).correlation if np.std(y) > 0 and np.std(s) > 0 else np.nan
    sp_nz = spearmanr(s[nz], y[nz]).correlation if nz.sum() >= 10 and np.std(y[nz]) > 0 and np.std(s[nz]) > 0 \
        else np.nan
    return dict(spearman=sp, spearman_nz=sp_nz, top10=top.mean(), top10_excess=top.mean() - y.mean(),
                top10_median=np.median(top), median_label=np.median(y), top1=top[0],
                oracle_top10_excess=np.sort(y)[::-1][:TOP_K].mean() - y.mean(), n=len(y))


def init(kb):
    global KB
    KB = kb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    out_dir = OUT_DIR / "step1_lodo"
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = load_kb()
    print(f"KB: {len(kb)} rows, {kb['dataset - id'].nunique()} datasets; NaN share of new metafeatures: "
          f"{kb[NEW_MF].isna().any(axis=1).mean():.3f}", flush=True)
    ids = sorted(kb["dataset - id"].unique())
    with multiprocessing.get_context("fork").Pool(a.workers, initializer=init, initargs=(kb,)) as pool:
        scored = pd.concat(pool.map(lodo_scores, ids))
    scored.to_parquet(out_dir / "scores.parquet")
    rankers = [c for c in scored.columns if c.startswith(("sur_", "heu_")) or c == "random"]
    rows = []
    for subset in ["add", "all"]:
        sub = scored if subset == "all" else scored[scored["operator"] != "delete"]
        for d, frame in sub.groupby("dataset - id"):
            for r in rankers:
                if subset == "all" and not r.startswith("sur_"):
                    continue
                rows.append(dict(subset=subset, dataset=d, ranker=r, **metrics(frame, r)))
    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_dataset.csv", index=False)

    summary = []
    for (subset, r), g in per.groupby(["subset", "ranker"]):
        ex = g["top10_excess"].to_numpy()
        try:
            pval = wilcoxon(ex, alternative="greater").pvalue
        except ValueError:
            pval = np.nan
        vs_random = g.merge(per[(per["ranker"] == "random") & (per["subset"] == "add")][["dataset", "top10"]],
                            on="dataset", suffixes=("", "_rand"))
        summary.append(dict(
            subset=subset, ranker=r, datasets=len(g), median_spearman=g["spearman"].median(),
            median_spearman_nz=g["spearman_nz"].median(), mean_top10_excess=g["top10_excess"].mean(),
            median_top10_excess=g["top10_excess"].median(), median_top10_median=g["top10_median"].median(),
            datasets_top10_above_mean=int((ex > 0).sum()), wilcoxon_p=pval,
            datasets_beat_random_draw=int((vs_random["top10"] > vs_random["top10_rand"]).sum()),
            median_oracle_top10_excess=g["oracle_top10_excess"].median(),
            gate=bool(pval < GATE_P and np.median(ex) > 0 and g["spearman_nz"].median() > GATE_SPEARMAN_NZ)))
    summary = pd.DataFrame(summary).sort_values(["subset", "median_top10_excess"], ascending=[True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False), flush=True)

    add = summary[summary["subset"] == "add"].set_index("ranker")
    step1_pass = [r for r in add.index if r.startswith("sur_") and r != "sur_old" and add.at[r, "gate"]]
    step2_pass = [r for r in add.index if r.startswith("heu_") and add.at[r, "gate"]]
    passing = add.loc[step1_pass + step2_pass].sort_values("median_top10_excess", ascending=False)
    gate = dict(step1_pass=step1_pass, step2_pass=step2_pass, step1_gate=bool(step1_pass),
                ranker_for_step3=passing.index[0] if step1_pass else None,
                criteria=f"wilcoxon(top10_excess) p < {GATE_P}, median excess > 0, median spearman_nz > "
                         f"{GATE_SPEARMAN_NZ}; add-feature candidates")
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
