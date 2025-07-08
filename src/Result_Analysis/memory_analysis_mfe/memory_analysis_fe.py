import os

import openml
from matplotlib import pyplot as plt
import pandas as pd

from src.Apply_and_Test.analyse_results import insert_line_breaks


def main():
    memory_all = {}
    memory_mfe = {}

    log_files = os.listdir("../time_analysis_mfe")
    log_files.sort()
    for log_file in log_files:
        if log_file.endswith('.out'):
            f = open("../time_analysis_mfe/" + log_file, "r")
            lines = f.readlines()
            method = log_file.split("_")[0]
            if method == "MFE":
                category = log_file.split("_")[3].split(".")[0]
                if int(category) == 0:
                    category = "general"
                elif int(category) == 1:
                    category = "statistical"
                elif int(category) == 2:
                    category = "info-theory"
                elif int(category) == 3:
                    category = "landmarking"
                elif int(category) == 4:
                    category = "complexity"
                elif int(category) == 5:
                    category = "clustering"
                elif int(category) == 6:
                    category = "concept"
                elif int(category) == 7:
                    category = "itemset"
                method_category = method + "_" + category
                for line in lines:
                    if line.startswith("Allocated memory per node:"):
                        memory = float(line.split(":")[-1].split(" MB")[0])
                        method_and_category_and_dataset = str(method_category)
                        memory_mfe.update({method_and_category_and_dataset: memory})

            for line in lines:
                if line.startswith("Allocated memory per node:"):
                    memory = float(line.split(":")[-1].split(" MB")[0])
                    method_and_dataset = str(method)
                    memory_all.update({method_and_dataset: memory})

    print(memory_all)
    print(memory_mfe)

    df = pd.DataFrame(list(memory_all.items()), columns=['method', 'value'])
    # df[['method', 'dataset']] = df['method_dataset'].str.split(' - ', expand=True)
    # df['dataset'] = df['dataset'].astype(int)

    memory_per_method = df.groupby("method")["value"].sum().sort_values(ascending=False)
    # Plot
    plt.figure(figsize=(10, 6))
    memory_per_method.plot(kind='bar', color='skyblue', label='Total Memory per Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Memory in MB")
    plt.title("Memory per Method")
    plt.xticks(rotation=45, ha="right")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Memory_for_FE_per_Method.png")
    plt.show()

    df_mfe = pd.DataFrame(list(memory_mfe.items()), columns=['method_category', 'value'])
    df_mfe[['method', 'category']] = df_mfe['method_category'].str.split('_', expand=True)

    memory_per_category = (df_mfe.groupby("category")["value"].sum().sort_values(ascending=False)) / 1000
    # Plot
    plt.figure(figsize=(10, 6))
    memory_per_category.plot(kind='bar', color='skyblue', label='Total Memory per Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Memory in GB")
    plt.title("Memory per MFE Category")
    plt.xticks(rotation=45, ha="right")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Memory_for_MFE_per_Category.png")
    plt.show()


if __name__ == "__main__":
    main()
