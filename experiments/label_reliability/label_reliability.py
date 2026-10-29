"""Label reliability test: are the knowledge-base labels a repeatable signal or mostly seed noise?

Two arms on a sample of candidates of DATASETS (CANDIDATES_PER_DATASET random ones, balanced over the operators, plus
the TOP_N best by their knowledge-base label to measure regression to the mean):
  ag     the label process of the knowledge base exactly (metafe.core_matrix: AutoGluon 1.2.1 LightGBM, 8-fold
         bagging, relative improvement of the out-of-fold score), with AutoGluon seeds 1-3. Seed 0 is the knowledge
         base itself (a seed-0 rerun reproduced 5 of 5 labels of 146818 bit-exactly). Needs the AutoGluon 1.2 venv.
  lgbcv  a cheaper candidate label: plain LightGBM, 5-fold CV repeated REPEATS times (fold split and model seed r),
         paired with the original data on the same folds. Can averaging give a reliable label?

    python experiments/label_reliability/label_reliability.py candidates
    python experiments/label_reliability/label_reliability.py ag --dataset 146818 --seed 1 --chunk 0 --n_chunks 4
    python experiments/label_reliability/label_reliability.py lgbcv --dataset 146818 --workers 32
    python experiments/label_reliability/label_reliability.py analyze
Output: results/label_reliability/{candidates,ag,lgbcv}/..., analysis/{per_dataset.csv, summary.txt}
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "label_reliability"
KB_MATRIX = REPO / "data" / "lightgbm_reuse" / "metafeature_matrix.parquet"

DATASETS = [146818, 168757, 168784, 359950, 359960, 359968, 359936, 359983, 359944, 359930]
CANDIDATES_PER_DATASET = 200
TOP_N = 20
AG_SEEDS = [1, 2, 3]
REPEATS = 10
CV_ROWS = 20000


def set_openml_cache():
    import numpy
    import openml

    # The pickles of the cache depend on the numpy major version (analysis venv: numpy 2, AutoGluon 1.2: numpy 1)
    suffix = "" if numpy.__version__.startswith("2") else "_np1"
    openml.config.set_root_cache_directory(str(REPO.parent / f"openml_cache{suffix}"))


# ------------------------------------------------------------------------------------------------------ candidates

def make_candidates():
    from metafe.core_matrix import sample_candidates

    kb = pd.read_parquet(KB_MATRIX, columns=["dataset - id", "feature - name", "operator", "improvement"])
    out = []
    for d in DATASETS:
        k = kb[kb["dataset - id"] == d].reset_index(drop=True)
        sample = set(sample_candidates(list(k["feature - name"]), CANDIDATES_PER_DATASET, seed=0))
        top = set(k.nlargest(TOP_N, "improvement")["feature - name"])
        k = k[k["feature - name"].isin(sample | top)].copy()
        k["in_random_sample"] = k["feature - name"].isin(sample)
        k["in_top"] = k["feature - name"].isin(top)
        out.append(k.rename(columns={"improvement": "kb_label"}))
        print(f"{d}: {k['in_random_sample'].sum()} random + {(~k['in_random_sample']).sum()} extra top candidates")
    path = OUT / "candidates" / "candidates.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_parquet(path)


def candidates(dataset_id):
    c = pd.read_parquet(OUT / "candidates" / "candidates.parquet")
    return c[c["dataset - id"] == dataset_id].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------------ ag arm

def run_ag(dataset_id, seed, chunk, n_chunks):
    from metafe import config
    from metafe.core_matrix import with_candidate
    from metafe.data import label_data, is_classification, write_parquet
    from metafe.models import fit_leaderboard, validation_scores, model_name, relative_improvement

    out = OUT / "ag" / f"{dataset_id}_seed{seed}_chunk{chunk}.parquet"
    names = list(candidates(dataset_id)["feature - name"])[chunk::n_chunks]
    X, y, md = label_data(dataset_id, config.EXPERIMENTS["lightgbm_reuse"])  # rows of the knowledge-base labels
    classification = is_classification(md)
    model = model_name("GBM")

    def score(X_):
        try:
            return validation_scores(fit_leaderboard(X_, y, None, None, classification, ["GBM"], seed=seed))[model]
        except Exception as e:  # noqa: BLE001
            print(f"[Warning] AutoGluon failed: {e}", flush=True)
            return np.nan

    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    base = done["original_score"].iloc[0] if len(done) else score(X)
    rows = done.to_dict("records")
    todo = [n for n in names if n not in set(done.get("feature - name", []))]
    print(f"{dataset_id} seed {seed} chunk {chunk}: original {base}, {len(todo)} of {len(names)} to do", flush=True)
    for n in todo:
        t = time.time()
        s = score(with_candidate(n, X))
        label = np.nan if np.isnan(s) else relative_improvement(base, s)
        rows.append({"dataset - id": dataset_id, "feature - name": n, "seed": seed, "label": label, "score": s,
                     "original_score": base})
        write_parquet(pd.DataFrame(rows), out)
        print(f"{n}: {label} ({time.time() - t:.0f} s)", flush=True)


# --------------------------------------------------------------------------------------------------------- lgbcv arm

def _lgbcv_task(args):
    """Per-fold losses of the original data (name None) or a candidate, for every repeat: array (REPEATS, 5)."""
    name = args
    import lightgbm as lgb
    from metafe.core_matrix import with_candidate

    X = G["X"] if name is None else with_candidate(name, G["X"])
    Xp = G["prepare"](X)
    Xp.columns = [f"f{i}" for i in range(Xp.shape[1])]
    y, cls, k = G["y"], G["classification"], G["n_classes"]
    losses = np.zeros((REPEATS, 5))
    for r, folds in enumerate(G["folds"]):
        for f, (tr, va) in enumerate(folds):
            p = G["params"](cls, k, seed=r, n_estimators=300)
            m = (lgb.LGBMClassifier if cls else lgb.LGBMRegressor)(**p)
            m.fit(Xp.iloc[tr], y[tr])
            if cls:
                proba = np.full((len(va), k), 1e-6)
                proba[:, m.classes_.astype(int)] = m.predict_proba(Xp.iloc[va])
                proba /= proba.sum(1, keepdims=True)
                losses[r, f] = -np.mean(np.log(np.clip(proba[np.arange(len(va)), y[va]], 1e-15, 1)))
            else:
                losses[r, f] = np.sqrt(np.mean((y[va] - m.predict(Xp.iloc[va])) ** 2))
    return name, losses


def run_lgbcv(dataset_id, workers):
    import multiprocessing
    from sklearn.model_selection import KFold, StratifiedKFold
    sys.path.insert(0, str(REPO / "experiments" / "surrogate_check"))
    from common import encode_y, lgb_params, prepare_X  # noqa: E402  (sets the numpy-2 OpenML cache)
    from metafe import config
    from metafe.data import label_data, is_classification, write_parquet

    X, y_raw, md = label_data(dataset_id, config.EXPERIMENTS["lightgbm_reuse"])
    cls = is_classification(md)
    y, k = encode_y(y_raw) if cls else (pd.to_numeric(y_raw).to_numpy(dtype="float64"), 0)
    if len(X) > CV_ROWS:
        keep = np.sort(np.random.RandomState(0).choice(len(X), CV_ROWS, replace=False))
        X, y = X.iloc[keep], y[keep]
    folds = []
    for r in range(REPEATS):
        try:
            assert cls
            folds.append(list(StratifiedKFold(5, shuffle=True, random_state=r).split(X, y)))
        except (AssertionError, ValueError):
            folds.append(list(KFold(5, shuffle=True, random_state=r).split(X)))
    global G
    G = dict(X=X, y=y, classification=cls, n_classes=k, folds=folds, prepare=prepare_X, params=lgb_params)
    names = [None] + list(candidates(dataset_id)["feature - name"])
    t = time.time()
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = dict(pool.map(_lgbcv_task, names, chunksize=1))
    base = results.pop(None)
    rows = []
    for n, loss in results.items():
        gain = (base - loss) / base  # paired relative loss decrease per repeat and fold
        for r in range(REPEATS):
            rows.append({"dataset - id": dataset_id, "feature - name": n, "repeat": r, "label": gain[r].mean(),
                         "folds_better": int((gain[r] > 0).sum())})
    write_parquet(pd.DataFrame(rows), OUT / "lgbcv" / f"{dataset_id}.parquet")
    print(f"{dataset_id}: {len(results)} candidates x {REPEATS} repeats in {time.time() - t:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------- analysis

def icc1(M):
    """ICC(1) of a (candidates x replicates) matrix: share of the variance between candidates (one-way ANOVA)."""
    M = M[~np.isnan(M).any(1)]
    n, k = M.shape
    if n < 3 or k < 2:
        return np.nan
    msb = k * np.var(M.mean(axis=1), ddof=1)
    msw = np.sum((M - M.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if msb + (k - 1) * msw > 0 else np.nan


def spearman_brown(r, k):
    return k * r / (1 + (k - 1) * r) if np.isfinite(r) else np.nan


def pairwise(M, method):
    cols = range(M.shape[1])
    vals = [pd.Series(M[:, i]).corr(pd.Series(M[:, j]), method=method) for i in cols for j in cols if i < j]
    return np.nanmean(vals) if vals else np.nan


def analyze():
    from scipy.stats import spearmanr

    cand = pd.read_parquet(OUT / "candidates" / "candidates.parquet")
    ag_files = sorted((OUT / "ag").glob("*.parquet"))
    ag = pd.concat([pd.read_parquet(p) for p in ag_files]) if ag_files else pd.DataFrame()
    lg_files = sorted((OUT / "lgbcv").glob("*.parquet"))
    lg = pd.concat([pd.read_parquet(p) for p in lg_files]) if lg_files else pd.DataFrame()
    rows, lines = [], []
    for d, subset in [(d, subset) for subset in ["add", "all"] for d in DATASETS]:
        c = cand[cand["dataset - id"] == d].set_index("feature - name")
        if subset == "add":  # add-feature candidates only (MetaFE v2 never deletes; delete labels are large)
            c = c[~c.index.str.startswith("without - ")]
        row = {"subset": subset, "dataset": d, "candidates": int(c["in_random_sample"].sum())}
        if len(ag) and (ag["dataset - id"] == d).any():
            a = ag[ag["dataset - id"] == d].pivot_table(index="feature - name", columns="seed", values="label")
            a[0] = c["kb_label"]
            a = a[[0] + [s for s in AG_SEEDS if s in a.columns]]
            rnd = a.loc[a.index.intersection(c.index[c["in_random_sample"]])].dropna()
            M = rnd.to_numpy()
            nz = M[(M != 0).all(1)]
            row.update(ag_seeds_done=M.shape[1] - 1, ag_rows=len(M),
                       ag_spearman=pairwise(M, "spearman"), ag_pearson=pairwise(M, "pearson"),
                       ag_spearman_nonzero=pairwise(nz, "spearman") if len(nz) >= 10 else np.nan,
                       ag_icc=icc1(M), ag_icc_mean4=spearman_brown(icc1(M), M.shape[1]),
                       kb_zero_share=float((M[:, 0] == 0).mean()),
                       zero_consistent=float(((M == 0).all(1) | (M != 0).all(1)).mean()),
                       between_sd=float(np.std(M.mean(axis=1))), within_sd=float(np.sqrt(np.mean(np.var(M, 1, ddof=1)))))
            pairs = [(M[:, i], M[:, j]) for i in range(M.shape[1]) for j in range(i + 1, M.shape[1])]
            agree = [np.mean(np.sign(u[(u != 0) & (v != 0)]) == np.sign(v[(u != 0) & (v != 0)])) for u, v in pairs
                     if ((u != 0) & (v != 0)).sum() > 0]
            row["ag_sign_agreement"] = float(np.mean(agree)) if agree else np.nan
            top = a.loc[a.index.intersection(c.index[c["in_top"]])].dropna()
            if len(top):
                new = top.drop(columns=0).mean(axis=1)
                row.update(top_kb_mean=top[0].mean(), top_new_mean=new.mean(), top_new_positive=float((new > 0).mean()),
                           top_mean_label_all=rnd.drop(columns=0).mean(axis=1).mean())
            base = ag[ag["dataset - id"] == d].groupby("seed")["original_score"].first()
            row["original_score_rel_sd"] = float(base.std() / base.mean()) if len(base) > 1 else np.nan
        if len(lg) and (lg["dataset - id"] == d).any():
            L = lg[lg["dataset - id"] == d].pivot_table(index="feature - name", columns="repeat", values="label")
            L = L.loc[L.index.intersection(c.index[c["in_random_sample"]])]
            Lm = L.to_numpy()
            half = REPEATS // 2
            split = spearmanr(Lm[:, :half].mean(axis=1), Lm[:, half:].mean(axis=1)).correlation
            r1 = icc1(Lm)
            row.update(lgbcv_icc_single=r1, lgbcv_icc_mean10=spearman_brown(r1, REPEATS),
                       lgbcv_split_half_spearman=split,
                       lgbcv_vs_kb_spearman=spearmanr(L.mean(axis=1), c.loc[L.index, "kb_label"]).correlation)
            if len(ag) and (ag["dataset - id"] == d).any():
                ag_mean = rnd.mean(axis=1)
                common_idx = L.index.intersection(ag_mean.index)
                row["lgbcv_vs_ag_mean_spearman"] = spearmanr(L.loc[common_idx].mean(axis=1),
                                                             ag_mean.loc[common_idx]).correlation
        rows.append(row)
    per = pd.DataFrame(rows)
    out = OUT / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_dataset.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        text = per.to_string(index=False)
        lines.append(text)
        main = per[per["dataset"] != 359930]
        lines.append("\nMedian over datasets (without 359930 quake), per subset:")
        lines.append(main.drop(columns=["dataset"]).groupby("subset").median(numeric_only=True).T.to_string())
    (out / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["candidates", "ag", "lgbcv", "analyze"])
    p.add_argument("--dataset", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--n_chunks", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.command in ("candidates", "ag"):
        set_openml_cache()
    if a.command == "candidates":
        make_candidates()
    elif a.command == "ag":
        from metafe.models import check_setup
        from metafe import config
        check_setup(config.EXPERIMENTS["lightgbm_reuse"], ["GBM"])
        run_ag(a.dataset, a.seed, a.chunk, a.n_chunks)
    elif a.command == "lgbcv":
        run_lgbcv(a.dataset, a.workers)
    else:
        analyze()


if __name__ == "__main__":
    main()
