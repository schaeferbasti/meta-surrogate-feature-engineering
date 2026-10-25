"""Average the evaluation folds: evaluation/<model>/<method>_Result_<id>_<fold>.parquet -> evaluation/Result_<id>.parquet
(mean and std over the folds per method and model).

    python -m metafe.evaluation.merge_folds --experiment lightgbm
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

from metafe import config
from metafe.data import write_parquet


def merge_folds(experiment):
    frames = defaultdict(list)
    for path in glob.glob(os.path.join(experiment.evaluation_dir, "*", "*_Result_*_*.parquet")):
        match = re.fullmatch(r".+_Result_(\d+)_\d+\.parquet", os.path.basename(path))
        if match:
            frame = pd.read_parquet(path)
            frame["dataset"] = frame["dataset"].astype(int)  # older result files store it as a string
            frames[int(match.group(1))].append(frame)
    for dataset_id, dataset_frames in sorted(frames.items()):
        merged = pd.concat(dataset_frames, ignore_index=True).groupby(
            ["origin", "task_type", "dataset", "model"], as_index=False).agg(
            score_val_mean=("score_val", "mean"), score_val_std=("score_val", "std"),
            score_test_mean=("score_test", "mean"), score_test_std=("score_test", "std"),
            n_folds=("score_test", "count"))
        write_parquet(merged, os.path.join(experiment.evaluation_dir, f"Result_{dataset_id}.parquet"))
        print(f"Dataset {dataset_id}: {len(merged)} (method, model) pairs, "
              f"folds {merged['n_folds'].min()}-{merged['n_folds'].max()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_experiment_argument(parser)
    merge_folds(config.EXPERIMENTS[parser.parse_args().experiment])


if __name__ == "__main__":
    main()
