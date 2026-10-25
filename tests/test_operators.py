"""Self-test of metafe.operators on real datasets (no model training; a few minutes).

For every operator and pair of the first --max_columns columns (on --max_rows sampled rows):
  1. order independence: shuffling the training rows gives every row the same feature value,
  2. test rows: a feature with values on the training rows has values on the test rows (for group operators: if a
     test row belongs to a training group with a value),
and it reports how often the training feature differs from the first (leaky) implementation, i.e. how many old
core-matrix labels cannot be reused.

    python -m tests.test_operators [dataset_id ...]   (default: 146818 168350 359937 359954)
"""
import argparse
import itertools
import sys
from collections import Counter

import numpy as np

from metafe.data import load_task
from metafe.legacy_operators import legacy_train_feature
from metafe.operators import UNARY_OPERATORS, BINARY_OPERATORS, fit_apply_operator, same_feature


def check_dataset(dataset_id, max_columns=8, max_rows=3000, seed=0):
    X_train, _, X_test, _, _ = load_task(dataset_id)
    # Random rows (the legacy implementation is slow on large datasets; the training rows are sorted by class)
    X_train = X_train.sample(n=min(max_rows, len(X_train)), random_state=seed)
    X_test = X_test.sample(n=min(max_rows, len(X_test)), random_state=seed)
    perm = np.random.default_rng(seed).permutation(len(X_train))
    X_shuffled = X_train.iloc[perm]
    columns = list(X_train.columns[:max_columns])
    candidates = [(op, a, None) for a in columns for op in UNARY_OPERATORS]
    candidates += [(op, a, b) for a, b in itertools.product(columns, columns) for op in BINARY_OPERATORS]
    failures, changed, total = [], Counter(), Counter()
    for op, a, b in candidates:
        operands = lambda X: [X[a]] + ([] if b is None else [X[b]])
        name = f"{op}({a}{'' if b is None else ', ' + b})"
        second = lambda X: None if b is None else X[b]
        f_train, f_test = fit_apply_operator(op, X_train[a], second(X_train), X_test[a], second(X_test))
        f_shuffled, _ = fit_apply_operator(op, X_shuffled[a], second(X_shuffled))
        if not same_feature(f_train.iloc[perm], f_shuffled, rtol=1e-9):
            failures.append(f"order-dependent: {name}")
        has_values = X_test[a].notna().any()
        if b is not None:  # group operators: only test rows of training groups with a value
            has_values = X_test[b].astype(object).isin(set(X_train[b].astype(object)[f_train.notna()])).any()
        if f_train.notna().any() and f_test.isna().all() and has_values:
            failures.append(f"all-NaN on test rows: {name}")
        try:
            differs = not same_feature(legacy_train_feature(op, *operands(X_train)), f_train, rtol=1e-9)
        except Exception:
            differs = True
        total[op] += 1
        changed[op] += differs
    return failures, changed, total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", type=int, nargs="*", default=[146818, 168350, 359937, 359954])
    parser.add_argument("--max_columns", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=3000)
    args = parser.parse_args()
    all_failures = []
    for dataset_id in args.datasets:
        failures, changed, total = check_dataset(dataset_id, args.max_columns, args.max_rows)
        all_failures += [f"{dataset_id}: {f}" for f in failures]
        summary = ", ".join(f"{op} {changed[op]}/{total[op]}" for op in total if changed[op])
        print(f"Dataset {dataset_id}: {len(failures)} failures; training feature differs from the first "
              f"implementation: {summary or 'none'}")
    for failure in all_failures[:50]:
        print("  FAIL", failure)
    sys.exit(1 if all_failures else 0)


if __name__ == "__main__":
    main()
