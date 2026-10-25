import argparse
import os
import sys
import time

from src.utils.get_data import get_openml_dataset_split_and_metadata, get_openfe_data, concat_data


def main(dataset_id, n_jobs, out_dir="test_data"):
    # Same output as the inline generation in Test_FE.py, but run once per dataset before the (parallel) evaluation
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/FE_{dataset_id}_OpenFE.parquet"
    if os.path.exists(out_path):
        print(f"{out_path} exists, skipped.")
        return
    # Use the original OpenML train/test split (same rows as "Original" in Test_FE.py)
    X_train, y_train, X_test, y_test, dataset_metadata = get_openml_dataset_split_and_metadata(dataset_id)
    start = time.time()
    try:
        X_tr, y_tr, X_te, y_te = get_openfe_data(X_train, y_train, X_test, y_test, n_jobs=n_jobs)
    except SystemExit as e:
        # OpenFE catches errors in its worker processes and calls exit(), i.e. SystemExit(None) = status 0
        sys.exit(f"OpenFE exited ({e.code!r}) for dataset {dataset_id}, no data written")
    print(f"OpenFE Time: {time.time() - start}")
    # Write to a temporary file first, so that an interrupted job leaves no truncated FE data behind
    concat_data(X_tr, y_tr, X_te, y_te, "target").to_parquet(out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--out_dir", default="test_data")
    args = parser.parse_args()
    main(args.dataset, args.n_jobs, args.out_dir)
    # OpenFE catches errors in its worker processes and calls exit(), which ends the job with status 0
    if not os.path.exists(f"{args.out_dir}/FE_{args.dataset}_OpenFE.parquet"):
        sys.exit(f"OpenFE produced no data for dataset {args.dataset}")
