"""Step 1a: target-aware metafeatures (common.NEW_MF) of every knowledge-base candidate of one or all datasets.

    python experiments/surrogate_check/step1_metafeatures.py --dataset 146818
    python experiments/surrogate_check/step1_metafeatures.py --all --workers 40
Output: results/surrogate_check/mf_v2/<id>.parquet (one row per feature name of the knowledge base).
"""
import argparse
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, KB_MATRIX, NEW_MF, DatasetContext, load_task  # noqa: E402


def build(dataset_id):
    out = OUT_DIR / "mf_v2" / f"{dataset_id}.parquet"
    if out.exists():
        return f"{dataset_id}: exists"
    start = time.time()
    kb = pd.read_parquet(KB_MATRIX, columns=["dataset - id", "feature - name"])
    names = pd.unique(kb.loc[kb["dataset - id"] == dataset_id, "feature - name"])
    X_train, y_train, _, _, metadata = load_task(dataset_id)
    ctx = DatasetContext(X_train, y_train, metadata)
    rows, failed = [], 0
    for name in names:
        try:
            mf = ctx.describe(name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            if failed <= 3:
                print(f"[{dataset_id}] {name}: {e}", flush=True)
            mf = {k: np.nan for k in NEW_MF}
        rows.append({"dataset - id": dataset_id, "feature - name": name, **mf})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(str(out) + ".tmp")
    Path(str(out) + ".tmp").rename(out)
    return f"{dataset_id}: {len(names)} features, {failed} failed, {time.time() - start:.0f} s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.dataset:
        print(build(a.dataset), flush=True)
        return
    kb = pd.read_parquet(KB_MATRIX, columns=["dataset - id"])
    ids = kb["dataset - id"].value_counts().index.tolist()  # largest first
    with multiprocessing.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(build, ids):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
