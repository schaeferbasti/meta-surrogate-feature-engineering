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
    df_all["error_val"] = - df_all["score_val"]
    df_all["error_test"] = - df_all["score_test"]

    # Pivot to have datasets on x, methods on lines
    df_pivot_val = df_all.pivot(index="dataset", columns="origin", values="error_val")
    df_pivot_val = df_pivot_val.sort_index()  # Sort by dataset ID
    df_pivot_test = df_all.pivot(index="dataset", columns="origin", values="error_test")
    df_pivot_test = df_pivot_test.sort_index()  # Sort by dataset ID

    datasets = df_pivot_val.index.astype(str)
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
    for method in df_pivot_val.columns:
        plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)
    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Val Score (flipped)")
    # plt.yscale("log") # '', '', 'function', 'functionlog
    plt.title("Autogluon Val Score by FE Method per Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_results/Autogluon_Val_Score_by_FE_Method.png")
    plt.show()

    # Plot
    plt.figure(figsize=(12, 6))
    for method in df_pivot_val.columns:
        plt.plot(dataset_list_wrapped, df_pivot_test[method], marker='o', label=method)
    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Test Score (flipped)")
    # plt.yscale("log")  # '', '', 'function', 'functionlog
    plt.title("Autogluon Test Score by FE Method per Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_results/Autogluon_Test_Score_by_FE_Method.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    for method in df_pivot_val.columns:
        if method == "Best Random":
            plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)
        elif method.startswith("pandas"):
            plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)
        elif method.startswith("d2v"):
            plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)
        elif method.startswith("MFE"):
            plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)

    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Val Score (flipped)")
    plt.title("Autogluon Val Score by FE Method per Dataset (Best random vs. our method)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_results/Random_vs_Me_Val_graph.png")
    plt.show()

    plt.figure(figsize=(12, 6))
    for method in df_pivot_val.columns:
        if method == "Best Random":
            plt.plot(dataset_list_wrapped, df_pivot_test[method], marker='o', label=method)
        elif method.startswith("pandas"):
            plt.plot(dataset_list_wrapped, df_pivot_test[method], marker='o', label=method)
        elif method.startswith("d2v"):
            plt.plot(dataset_list_wrapped, df_pivot_test[method], marker='o', label=method)
        elif method.startswith("MFE"):
            plt.plot(dataset_list_wrapped, df_pivot_test[method], marker='o', label=method)

    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Test Score (flipped)")
    plt.title("Autogluon Test Score by FE Method per Dataset (Best random vs. our method)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_results/Random_vs_Me_Test_graph.png")
    plt.show()

    minValueIndex_val = df_pivot_val.idxmin(axis=1).value_counts()
    minValueIndex_test = df_pivot_val.idxmin(axis=1).value_counts()

    # Plot
    plt.figure(figsize=(10, 6))
    minValueIndex_val.plot(kind='bar', color='skyblue', label='Count of best val result on number of datasets')
    minValueIndex_test.plot(kind='bar',  width=0.3, color='darkblue', label='Count of best test result on number of datasets')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Best result on number of datasets")
    plt.title("Which method is the best?")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("test_results/Count_Best_bar.png")
    plt.show()


    df_pivot_val.drop(columns=["OpenFE"], inplace=True)
    df_pivot_test.drop(columns=["OpenFE"], inplace=True)
    minValueIndex_val = df_pivot_val.idxmin(axis=1).value_counts()
    minValueIndex_test = df_pivot_test.idxmin(axis=1).value_counts()

    # Plot
    plt.figure(figsize=(10, 6))
    minValueIndex_val.plot(kind='bar', color='skyblue', label='Count of best val result on number of datasets')
    minValueIndex_test.plot(kind='bar', width=0.3, color='darkblue', label='Count of best test result on number of datasets')
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
