import os

import openml
import pandas as pd
from matplotlib import pyplot as plt

from src.Apply_and_Test.analyse_results import insert_line_breaks


def main():
    times = {}

    log_files = os.listdir()

    for log_file in log_files:
        if log_file.endswith('.out'):
            f = open(log_file, "r")
            lines = f.readlines()
            method = log_file.split("_")[0]

            for line in lines:
                if line.__contains__("on") and line.__contains__("datasets"):
                    continue
                elif line.startswith("Time for"):
                    time = float(line.split(":")[-1])
                    dataset = line.split(":")[0].split("Dataset ")[-1]
                    method_and_dataset = str(method) + " - " + str(dataset)
                    times.update({method_and_dataset: time})
    print(times)

    df = pd.DataFrame(list(times.items()), columns=['method_dataset', 'value'])
    df[['method', 'dataset']] = df['method_dataset'].str.split(' - ', expand=True)
    df['dataset'] = df['dataset'].astype(int)

    df_pivot = df.pivot(index="dataset", columns="method", values="value")
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
    plt.yscale('log')  # Optional: only if values vary a lot
    plt.ylabel("Time in seconds")
    plt.title("Time for generating metafeatures by FE Method per Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Time_for_FE_per_Dataset.png")
    plt.show()

    time_per_method = df.groupby("method")["value"].sum().sort_values(ascending=False)
    average_time_per_method = df.groupby("method")["value"].mean().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time per Method')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time per Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time per Method")
    plt.xticks(rotation=45, ha="right")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Time_for_FE_per_Method.png")
    plt.show()


if __name__ == "__main__":
    main()
