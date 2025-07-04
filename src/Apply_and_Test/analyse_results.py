import glob
import pandas as pd
import matplotlib.pyplot as plt

result_files = glob.glob("test_results/Result_*.parquet")
all_results = []

for result_file in result_files:
    df = pd.read_parquet(result_file)
    dataset_id = int(result_file.split("Result_")[1].split(".parquet")[0])
    df["origin"] = df["origin"].apply(lambda x: "Best Random" if str(x).startswith("Random") else x)
    df["dataset_id"] = dataset_id
    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)

# Convert score to error (you can adjust this as needed)
df_all["error"] = 1 - df_all["score"]

# Pivot to have datasets on x, methods on lines
df_pivot = df_all.pivot(index="dataset_id", columns="origin", values="score")
df_pivot = df_pivot.sort_index()  # Sort by dataset ID

# Plot
plt.figure(figsize=(12, 6))
for method in df_pivot.columns:
    plt.plot(df_pivot.index.astype(str), df_pivot[method], marker='o', label=method)

plt.xlabel("Dataset ID")
plt.ylabel("Autogluon Score")
plt.title("Autogluon Score by FE Method per Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_results/Autogluon_Score_by_FE_Method.png")
plt.show()
