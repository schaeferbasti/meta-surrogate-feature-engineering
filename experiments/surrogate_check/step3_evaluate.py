"""Step 3 evaluation with a proxy: plain LightGBM (seeds 0-2, early stopping on 10 % of the training rows) on the
original OpenML split, for Original, TabPrep, OpenFE, old MetaFE (lightgbm_reuse, 100 s and 1000 s) and MetaFE v2.
Test loss: log loss (classification) or RMSE (regression).

    python experiments/surrogate_check/step3_evaluate.py --dataset 146818
    python experiments/surrogate_check/step3_evaluate.py --summary
Output: results/surrogate_check/eval_v2/<id>.csv, summary.csv, per_dataset.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, REPO, encode_y, lgb_params, prepare_X, load_task  # noqa: E402
from metafe.config import TARGET  # noqa: E402
from metafe.data import is_classification, split_like_original  # noqa: E402
from metafe.metafe import sanitize_column_name  # noqa: E402

SEEDS = [0, 1, 2]
METHODS = {
    "TabPrep": REPO / "data/fe/TabPrep/FE_{id}_TabPrep.parquet",
    "OpenFE": REPO / "data/fe/OpenFE/FE_{id}_OpenFE.parquet",
    "MetaFE_old_100": REPO / "data/lightgbm_reuse/fe/MetaFE_GBM/FE_{id}_MetaFE_100.parquet",
    "MetaFE_old_1000": REPO / "data/lightgbm_reuse/fe/MetaFE_GBM/FE_{id}_MetaFE_1000.parquet",
    "MetaFE_v2": OUT_DIR / "fe_v2/FE_{id}_MetaFE_v2.parquet",
}


def fit_eval(X_train, y_train, X_test, y_test, classification, n_classes, seed, n_jobs):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    Xtr, Xte = prepare_X(X_train), prepare_X(X_test, reference=X_train)
    Xtr.columns = Xte.columns = [f"f{i}" for i in range(Xtr.shape[1])]
    try:
        tr, va = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=seed,
                                  stratify=y_train if classification else None)
    except ValueError:
        tr, va = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=seed)
    p = lgb_params(classification, n_classes, seed, n_estimators=5000)
    p["n_jobs"] = n_jobs
    model = (lgb.LGBMClassifier if classification else lgb.LGBMRegressor)(**p)
    model.fit(Xtr.iloc[tr], y_train[tr], eval_set=[(Xtr.iloc[va], y_train[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    if classification:
        proba = np.full((len(Xte), n_classes), 1e-6)
        proba[:, model.classes_.astype(int)] = model.predict_proba(Xte)
        proba /= proba.sum(1, keepdims=True)
        return -np.mean(np.log(np.clip(proba[np.arange(len(Xte)), y_test], 1e-15, 1)))
    return float(np.sqrt(np.mean((y_test - model.predict(Xte)) ** 2)))


def evaluate(dataset_id, n_jobs):
    X_train, y_train, X_test, y_test, metadata = load_task(dataset_id)
    classification = is_classification(metadata)
    if classification:
        cats = np.sort(y_train.astype(str).unique())
        codes = {c: i for i, c in enumerate(cats)}
        ytr, n_classes = encode_y(y_train)
        yte = y_test.astype(str).map(codes).fillna(-1).astype(int).to_numpy()
        keep = yte >= 0  # a test class never seen in training cannot be scored
    else:
        ytr, n_classes = pd.to_numeric(y_train).to_numpy(dtype="float64"), 0
        yte, keep = pd.to_numeric(y_test).to_numpy(dtype="float64"), np.ones(len(y_test), bool)
    data = {"Original": (X_train, X_test)}
    for method, path in METHODS.items():
        path = Path(str(path).format(id=dataset_id))
        if path.exists():
            fe = pd.read_parquet(path)
            Xs_tr, _, Xs_te, _ = split_like_original(fe, X_train.rename(columns=sanitize_column_name),
                                                    X_test.rename(columns=sanitize_column_name)) \
                if "MetaFE" in method else split_like_original(fe, X_train, X_test)
            data[method] = (Xs_tr.drop(columns=[TARGET], errors="ignore"), Xs_te.drop(columns=[TARGET], errors="ignore"))
    rows = []
    for method, (Xtr, Xte) in data.items():
        for seed in SEEDS:
            loss = fit_eval(Xtr, ytr, Xte[keep], yte[keep], classification, n_classes, seed, n_jobs)
            rows.append(dict(dataset=dataset_id, method=method, seed=seed, test_loss=loss, n_cols=Xtr.shape[1]))
            print(f"[{dataset_id}] {method} seed {seed}: {loss:.5f}", flush=True)
    out = OUT_DIR / "eval_v2" / f"{dataset_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def summary():
    d = OUT_DIR / "eval_v2"
    res = pd.concat([pd.read_csv(p) for p in sorted(d.glob("[0-9]*.csv"))])
    loss = res.groupby(["dataset", "method"])["test_loss"].mean().unstack()
    imp = (loss.rsub(loss["Original"], axis=0)).div(loss["Original"], axis=0) * 100  # % loss decrease
    imp.to_csv(d / "per_dataset.csv")
    rows = []
    for m in imp.columns.drop("Original"):
        v = imp[m].dropna()
        rows.append(dict(method=m, datasets=len(v), mean_pct=v.mean(), median_pct=v.median(),
                         better=int((v > 0).sum()), worse=int((v < 0).sum()), worst_pct=v.min()))
    s = pd.DataFrame(rows)
    s.to_csv(d / "summary.csv", index=False)
    print(s.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=int)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    if a.summary:
        return summary()
    if not (OUT_DIR / "fe_v2" / f"FE_{a.dataset}_MetaFE_v2.parquet").exists():
        print(f"No MetaFE v2 data for {a.dataset} (gate failed or step 3 failed), skip")
        return
    evaluate(a.dataset, a.n_jobs)


if __name__ == "__main__":
    main()
