"""Feature-engineered data of the baselines OpenFE and TabPrep, on the original OpenML split (fitted on train rows).

Writes data/fe/<method>/FE_<id>_<method>.parquet (existing files are skipped).
TabPrep needs its own environment (AutoGluon from source, see README.md); this module only imports the method used.

    python -m metafe.evaluation.baselines --method OpenFE --dataset 146818 [--n_jobs 8]
    python -m metafe.evaluation.baselines --method TabPrep --dataset 146818 168350
"""
import argparse
import contextlib
import os
import re
import shutil
import sys
import tempfile
import time

import pandas as pd

from metafe import config
from metafe.data import load_task, concat_data, write_parquet, is_classification


def openfe(X_train, y_train, X_test, metadata, n_jobs=1):
    from openfe import OpenFE, transform

    # OpenFE writes ./openfe_tmp_data.feather (transform() ignores tmp_save_path), so parallel jobs in one directory
    # overwrite each other's file: run it in a private temporary working directory
    tmp_dir = tempfile.mkdtemp()
    try:
        with contextlib.chdir(tmp_dir):
            features = OpenFE().fit(data=X_train, label=y_train, n_jobs=n_jobs,
                                    tmp_save_path=os.path.join(tmp_dir, "openfe_tmp_data.feather"))
            return transform(X_train, X_test, features, n_jobs=n_jobs)
    except SystemExit as e:
        # OpenFE catches errors in its worker processes and calls exit(), i.e. SystemExit with status 0
        raise RuntimeError(f"OpenFE exited ({e.code!r})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def tabprep(X_train, y_train, X_test, metadata, random_state=0):
    from sklearn.preprocessing import LabelEncoder
    from tabprep import TabPrepFeatureGenerator

    if is_classification(metadata):
        target_type = "binary" if y_train.nunique() == 2 else "multiclass"
        y_fit = pd.Series(LabelEncoder().fit_transform(y_train), index=y_train.index)  # only for fitting TabPrep
    else:
        target_type = "regression"
        y_fit = y_train.astype(float)
    generator = TabPrepFeatureGenerator(target_type=target_type, random_state=random_state)
    X_train_fe = generator.fit_transform(X_train, y_fit)
    return X_train_fe, generator.transform(X_test)


def unique_clean_columns(X_train, X_test):
    # Column names with only [A-Za-z0-9_] (LightGBM), made unique again
    mapping, seen = {}, {}
    for col in X_train.columns:
        new = re.sub("[^A-Za-z0-9_]+", "_", str(col))
        seen[new] = seen.get(new, -1) + 1
        mapping[col] = new if seen[new] == 0 else f"{new}_{seen[new]}"
    return X_train.rename(columns=mapping), X_test.rename(columns=mapping)


METHODS = {"OpenFE": openfe, "TabPrep": tabprep}


def generate(method, dataset_id, out_dir=None, n_jobs=1):
    out_dir = out_dir or str(config.BASELINE_FE_DIR / method)
    out_path = os.path.join(out_dir, f"FE_{dataset_id}_{method}.parquet")
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipped")
        return
    X_train, y_train, X_test, y_test, metadata = load_task(dataset_id)
    start = time.time()
    kwargs = {"n_jobs": n_jobs} if method == "OpenFE" else {}
    X_train_fe, X_test_fe = METHODS[method](X_train, y_train, X_test, metadata, **kwargs)
    if method == "TabPrep":
        X_train_fe, X_test_fe = unique_clean_columns(X_train_fe, X_test_fe)
    print(f"{method} on dataset {dataset_id}: {time.time() - start:.0f} s")
    write_parquet(concat_data(X_train_fe, y_train, X_test_fe, y_test), out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--dataset", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", default=None, help="Default: data/fe/<method>")
    parser.add_argument("--n_jobs", type=int, default=1, help="OpenFE only")
    args = parser.parse_args()
    failed = []
    for dataset_id in args.dataset:
        try:
            generate(args.method, dataset_id, args.out_dir, args.n_jobs)
        except Exception as e:
            print(f"[Error] {args.method} failed on dataset {dataset_id}: {e}")
            failed.append(dataset_id)
    if failed:
        sys.exit(f"{args.method} failed on datasets {failed}")


if __name__ == "__main__":
    main()
