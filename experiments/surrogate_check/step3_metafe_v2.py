"""Step 3: MetaFE v2. Runs only if the step 1 gate passed (results/surrogate_check/step1_lodo/gate.json).

Per round (at most MAX_ROUNDS, so features are nested at most twice):
  1. candidates = all add-feature candidates of the current columns with depth <= 2 that were not tried before,
  2. describe them with the target-aware metafeatures (residual of an OOF LightGBM on the CURRENT columns) and rank
     them with the ranker chosen by the gate (LightGBM surrogate trained on the other 43 datasets, or a heuristic),
  3. walk down the ranking, skip near-duplicates (|Spearman| > 0.98 with a current column or an earlier pick) until
     TOP_K candidates are picked,
  4. verify each pick by 5-fold LightGBM CV on the training rows (same folds as the current data): it passes if the
     mean relative loss decrease is >= MIN_GAIN and it improves >= 4 of 5 folds,
  5. add the passing picks greedily (best first), each re-verified on top of the features added before.
Stops when a round adds nothing. Only the training rows are used; the features are applied to the test rows.

    python experiments/surrogate_check/step3_metafe_v2.py --dataset 146818
Output: results/surrogate_check/fe_v2/FE_<id>_MetaFE_v2.parquet (format of the FE files) and FE_<id>_log.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT_DIR, KB_MATRIX, NEW_MF, DatasetContext, load_task, lgb_params, prepare_X, rank01,  # noqa: E402
                    numeric)
from step1_lodo import FEATURE_SETS, SURROGATES, fit_surrogate, load_kb  # noqa: E402
from metafe.data import concat_data, write_parquet  # noqa: E402
from metafe.metafe import sanitize_column_name  # noqa: E402
from metafe.operators import DELETE, apply_features, candidate_featurenames, compute_feature, parse_featurename  # noqa: E402

MAX_ROUNDS = 2
MAX_DEPTH = 2
MAX_CANDIDATES = 20000  # per round (random subset beyond that)
TOP_K = 10
DUP_SPEARMAN = 0.98
MIN_GAIN = 0.002        # mean relative CV loss decrease
MIN_FOLDS_BETTER = 4
MAX_ADD_PER_ROUND = 5
CV_ROWS = 20000
# Constant features (min/max of a column) can never help
SKIP_OPERATORS = {"min", "max", DELETE}


class CV:
    """5-fold LightGBM CV loss per fold on (a subsample of) the training rows, with fixed folds."""

    def __init__(self, ctx, n_jobs, seed=0):
        from sklearn.model_selection import KFold, StratifiedKFold

        self.ctx, self.n_jobs = ctx, n_jobs
        rng = np.random.RandomState(seed)
        n = len(ctx.X)
        self.rows = np.sort(rng.choice(n, CV_ROWS, replace=False)) if n > CV_ROWS else np.arange(n)
        self.y = ctx.y[self.rows]
        try:
            assert ctx.classification
            self.folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(self.rows, self.y))
        except (AssertionError, ValueError):
            self.folds = list(KFold(5, shuffle=True, random_state=seed).split(self.rows))

    def loss(self, X):
        import lightgbm as lgb

        Xs = prepare_X(X.iloc[self.rows])
        Xs.columns = [f"f{i}" for i in range(Xs.shape[1])]
        out = []
        for tr, va in self.folds:
            p = lgb_params(self.ctx.classification, self.ctx.n_classes, 0, n_estimators=300)
            p["n_jobs"] = self.n_jobs
            model = (lgb.LGBMClassifier if self.ctx.classification else lgb.LGBMRegressor)(**p)
            model.fit(Xs.iloc[tr], self.y[tr])
            if self.ctx.classification:
                proba = np.full((len(va), self.ctx.n_classes), 1e-6)
                proba[:, model.classes_.astype(int)] = model.predict_proba(Xs.iloc[va])
                proba /= proba.sum(1, keepdims=True)
                out.append(-np.mean(np.log(np.clip(proba[np.arange(len(va)), self.y[va]], 1e-15, 1))))
            else:
                out.append(np.sqrt(np.mean((self.y[va] - model.predict(Xs.iloc[va])) ** 2)))
        return np.array(out)


def depth(name):
    return name.count("(")


def make_ranker(choice, dataset_id):
    """score(frame of NEW_MF + operator + old metafeatures) -> array; higher = better."""
    if choice.startswith("heu_"):
        stat = choice[len("heu_"):]
        return lambda mf: mf[stat].fillna(-np.inf).to_numpy()
    fs, target = SURROGATES[choice[len("sur_"):]]
    kb = load_kb()
    train = kb[kb["dataset - id"] != dataset_id]
    model = fit_surrogate(train, FEATURE_SETS[fs], target)
    categories = {c: train[c].cat.categories for c in ["operator", "task_type"]}

    def score(mf):
        mf = mf.copy()
        for c, cats in categories.items():
            mf[c] = pd.Categorical(mf[c], categories=cats)
        return model.predict(mf[FEATURE_SETS[fs]])
    return score


def old_metafeatures(feature):
    """pandas describe() metafeatures of the knowledge base (metafe.metafeatures), for the 'old'/'both' sets."""
    s = pd.Series(np.asarray(feature))
    try:
        s = pd.to_numeric(s, errors="raise")
        d = s.describe()
        return {"feature - count": d["count"], "feature - mean": d["mean"], "feature - std": d["std"],
                "feature - min": d["min"], "feature - 25%": d["25%"], "feature - 50%": d["50%"],
                "feature - 75%": d["75%"], "feature - max": d["max"], "feature - unique": np.nan,
                "feature - freq": np.nan}
    except (ValueError, TypeError):
        d = s.astype(str).describe()
        return {"feature - count": d["count"], "feature - unique": d["unique"], "feature - freq": d["freq"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=int, required=True)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--ranker", default=None, help="Override the gate's ranker (e.g. heu_eta_res_gain)")
    a = p.parse_args()
    choice = a.ranker
    if choice is None:
        gate = json.loads((OUT_DIR / "step1_lodo" / "gate.json").read_text())
        if not gate["step1_gate"]:
            print("Step 1 gate failed, MetaFE v2 is not run:", json.dumps(gate))
            return
        choice = gate["ranker_for_step3"]
    out_dir = OUT_DIR / "fe_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    X_train, y_train, X_test, y_test, metadata = load_task(a.dataset)
    X_train = X_train.rename(columns=sanitize_column_name)
    X_test = X_test.rename(columns=sanitize_column_name)
    task_type = metadata["task_type"]
    rank = make_ranker(choice, a.dataset)
    log, added, tried = [], [], set()
    ctx = DatasetContext(X_train, y_train, metadata)
    cv = CV(ctx, a.n_jobs)
    base = cv.loss(X_train)
    log.append(dict(round=0, feature=None, cv_loss=base.mean(), gain=np.nan, accepted=None, score=np.nan,
                    elapsed_s=time.time() - start))
    print(f"[{a.dataset}] ranker {choice}, CV loss {base.mean():.5f}", flush=True)
    for rnd in range(1, MAX_ROUNDS + 1):
        names = [n for n in candidate_featurenames(X_train) if n not in tried and depth(n) <= MAX_DEPTH
                 and parse_featurename(n)[0] not in SKIP_OPERATORS]
        if len(names) > MAX_CANDIDATES:
            names = list(np.random.RandomState(rnd).choice(names, MAX_CANDIDATES, replace=False))
        ctx = DatasetContext(X_train, y_train, metadata) if rnd > 1 else ctx  # residual of the current columns
        rows, values = [], {}
        for n in names:
            try:
                f, _ = compute_feature(n, X_train)
                mf = ctx.describe(n, f)
            except Exception:  # noqa: BLE001
                continue
            values[n] = f
            rows.append({"feature - name": n, "operator": parse_featurename(n)[0], "task_type": task_type, **mf,
                         **old_metafeatures(f)})
        tried.update(names)
        if not rows:
            break
        mf = pd.DataFrame(rows)
        mf["score"] = rank(mf)
        mf = mf.sort_values("score", ascending=False)
        col_ranks = [ctx.column_rank(c) for c in X_train.columns]
        picks, pick_ranks = [], []
        for n, s in zip(mf["feature - name"], mf["score"]):
            r = rank01(numeric(values[n].iloc[ctx.stat_rows]))
            if np.std(r) == 0:
                continue
            r = (r - r.mean()) / r.std()
            if any(abs(np.mean(r * c)) > DUP_SPEARMAN for c in col_ranks + pick_ranks):
                continue
            picks.append((n, s))
            pick_ranks.append(r)
            if len(picks) == TOP_K:
                break
        verified = []
        for n, s in picks:
            loss = cv.loss(apply_features(X_train, None, [n])[0])
            gain = (base - loss) / base
            ok = gain.mean() >= MIN_GAIN and (gain > 0).sum() >= MIN_FOLDS_BETTER
            log.append(dict(round=rnd, feature=n, cv_loss=loss.mean(), gain=gain.mean(), accepted="verified" if ok
                            else "rejected", score=s, elapsed_s=time.time() - start))
            print(f"[{a.dataset}] round {rnd} {n}: score {s:.4f}, gain {gain.mean():+.4f} "
                  f"({(gain > 0).sum()}/5) {'ok' if ok else ''}", flush=True)
            if ok:
                verified.append((gain.mean(), n, s, loss))
        n_added = 0
        for _, n, s, loss in sorted(verified, key=lambda v: -v[0]):
            if n_added == MAX_ADD_PER_ROUND:
                break
            X_new = apply_features(X_train, None, [n])[0]
            if n_added > 0:  # the best one was verified on the current data already
                loss = cv.loss(X_new)
            gain = (base - loss) / base
            if gain.mean() >= MIN_GAIN and (gain > 0).sum() >= MIN_FOLDS_BETTER:
                X_train, base = X_new, loss
                added.append(n)
                n_added += 1
                log.append(dict(round=rnd, feature=n, cv_loss=loss.mean(), gain=gain.mean(), accepted="added",
                                score=s, elapsed_s=time.time() - start))
                print(f"[{a.dataset}] round {rnd}: added {n} (CV loss {loss.mean():.5f})", flush=True)
        if n_added == 0:
            break
    X_tr, X_te = load_task(a.dataset)[0].rename(columns=sanitize_column_name), X_test
    X_tr, X_te = apply_features(X_tr, X_te, added)
    write_parquet(concat_data(X_tr, y_train, X_te, y_test), out_dir / f"FE_{a.dataset}_MetaFE_v2.parquet")
    pd.DataFrame(log).to_csv(out_dir / f"FE_{a.dataset}_log.csv", index=False)
    print(f"[{a.dataset}] done: {len(added)} features added in {time.time() - start:.0f} s: {added}", flush=True)


if __name__ == "__main__":
    main()
