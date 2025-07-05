import glob

import openml
import pandas as pd
import matplotlib.pyplot as plt


def insert_line_breaks(name, max_len=20):
    if len(name) > max_len:
        # Split into chunks of `max_len`, preserving words if possible
        parts = [name[i:i+max_len] for i in range(0, len(name), max_len)]
        return '\n'.join(parts)
    else:
        return name


def analyse_results():
    result_files = glob.glob("test_results/Result_*.parquet")
    all_results = []

    for result_file in result_files:
        df = pd.read_parquet(result_file)
        dataset_id = int(result_file.split("Result_")[1].split(".parquet")[0])
        df["origin"] = df["origin"].apply(lambda x: "Best Random" if str(x).startswith("Random") else x)
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all = df_all.drop_duplicates()
    # Convert score to error (you can adjust this as needed)
    df_all["error"] = - df_all["score"]

    # Pivot to have datasets on x, methods on lines
    df_pivot = df_all.pivot(index="dataset", columns="origin", values="error")
    df_pivot = df_pivot.sort_index()  # Sort by dataset ID


    datasets = df_pivot.index.astype(str)
    dataset_list = []
    for dataset in datasets.tolist():
        task = openml.tasks.get_task(
            int(dataset),
            download_splits=True,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        dataset = task.get_dataset().name
        dataset_list.append(dataset)
    dataset_list_wrapped = [insert_line_breaks(name, max_len=15) for name in dataset_list]

    # Plot
    plt.figure(figsize=(12, 6))
    for method in df_pivot.columns:
        plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)

    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Score")
    plt.yscale("log") # '', '', 'function', 'functionlog
    plt.title("Autogluon Score by FE Method per Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_results/Autogluon_Score_by_FE_Method.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    for method in df_pivot.columns:
        if method == "Best Random":
            plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
        elif method.startswith("pandas"):
            plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
        elif method.startswith("d2v"):
            plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
        elif method.startswith("mfe"):
            plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)

    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Score")
    plt.title("Autogluon Score by FE Method per Dataset (Best random vs. our method)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_results/Random_vs_Me_graph.png")
    plt.show()

    minValueIndex = df_pivot.idxmin(axis=1).value_counts()

    # Plot
    plt.figure(figsize=(10, 6))
    minValueIndex.plot(kind='bar', color='skyblue', label='Count of best result on number of datasets')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Best result on number of datasets")
    plt.title("Which method is the best?")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("test_results/Count_Best_bar.png")
    plt.show()


    df_pivot.drop(columns=["OpenFE"], inplace=True)
    minValueIndex = df_pivot.idxmin(axis=1).value_counts()

    # Plot
    plt.figure(figsize=(10, 6))
    minValueIndex.plot(kind='bar', color='skyblue', label='Count of best result on number of datasets')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Best result on number of datasets")
    plt.title("Which method is the best?")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("test_results/Count_Best_without_OpenFE_bar.png")
    plt.show()


if __name__ == "__main__":
    analyse_results()
