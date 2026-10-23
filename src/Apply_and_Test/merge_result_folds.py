import glob
import pandas as pd
from collections import defaultdict

# Step 1: Find all files
result_fold_files = glob.glob("test_results/Result_*_*.parquet")

# Step 2: Group by dataset_id
datasets = defaultdict(list)

for file in result_fold_files:
    filename = file.split("/")[-1]
    parts = filename.replace(".parquet", "").split("_")
    dataset_id = int(parts[1])
    df = pd.read_parquet(file)

    # Assign column names (if needed)
    df.columns = ["origin", "task_type", "dataset", "model", "score_val", "score_test"]

    datasets[dataset_id].append(df)

# Step 3: Average over folds for each dataset
for dataset_id, dfs in datasets.items():
    combined_df = pd.concat(dfs, ignore_index=True)

    # Group by all identifier columns and average the scores
    grouped = combined_df.groupby(["origin", "task_type", "dataset", "model"], as_index=False).agg(
        score_val_mean=("score_val", "mean"),
        score_val_std=("score_val", "std"),
        score_test_mean=("score_test", "mean"),
        score_test_std=("score_test", "std")
    )

    # Save result
    grouped.to_parquet(f"test_results/Result_{dataset_id}.parquet", index=False)

    print(f"Averaged result saved for dataset {dataset_id}")
