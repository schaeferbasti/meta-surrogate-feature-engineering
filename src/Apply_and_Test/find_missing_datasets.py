import glob
import pandas as pd

# Load all result files
result_files = glob.glob("test_results/Result_*.parquet")
not_result_files = glob.glob("test_results/Result_*_*.parquet")
result_files = [f for f in result_files if f not in not_result_files]

found_datasets = set()
for f in result_files:
    dataset_id = int(f.split("Result_")[1].split(".parquet")[0])
    found_datasets.add(dataset_id)



all_datasets = {2073, 146818, 146820, 167120, 167210, 168350, 168757, 168784, 189354, 190146, 233211, 359930, 359931,
                359932, 359933, 359935, 359936, 359937, 359938, 359944, 359949, 359950, 359952, 359954, 359955, 359956,
                359958, 359959, 359960, 359962, 359963, 359965, 359968, 359971, 359972, 359974, 359975, 359979, 359981,
                359982, 359983, 359987, 359992, 359993}

print(f"Expected datasets: {sorted(all_datasets)}")
print(f"Total expected: {len(all_datasets)}")
print(f"Found datasets: {sorted(found_datasets)}")
print(f"Total found: {len(found_datasets)}")

missing_all = all_datasets - found_datasets
print(f"\nMissing cluster datasets: {sorted(missing_all)}")

# Check which datasets have results for each method
all_results = []
for f in result_files:
    df = pd.read_parquet(f)
    all_results.append(df)

combined_results = pd.concat(all_results, ignore_index=True)
methods = combined_results['origin'].unique()

for method in methods:
    if method == "3600_pandas" or method == "pandas_recursion":
        method_datasets = set(combined_results[combined_results['origin'] == method]['dataset'].unique().astype(int))
        missing = all_datasets - method_datasets
        print(f"\n{method}: {len(method_datasets)}/{len(all_datasets)} datasets")
        print(f"  Missing datasets: {sorted(missing)}")
