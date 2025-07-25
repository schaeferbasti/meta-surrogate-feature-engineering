import os

import openml
import pandas as pd
from matplotlib import pyplot as plt

from src.Result_Analysis.test_analysis.test_analysis import insert_line_breaks


def main():
    times = {}
    times_mfe = {}

    log_files = os.listdir()
    log_files.sort()
    for log_file in log_files:
        if log_file.endswith('.out'):
            f = open(log_file, "r")
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
                    if line.__contains__("on") and line.__contains__("datasets"):
                        continue
                    elif line.startswith("Time for"):
                        time = float(line.split(":")[-1])
                        dataset = line.split(":")[0].split("Dataset ")[-1]
                        method_and_category_and_dataset = str(method_category) + " - " + str(dataset)
                        times_mfe.update({method_and_category_and_dataset: time})

            for line in lines:
                if line.__contains__("on") and line.__contains__("datasets"):
                    continue
                elif line.startswith("Time for"):
                    time = float(line.split(":")[-1])
                    dataset = line.split(":")[0].split("Dataset ")[-1]
                    method_and_dataset = str(method) + " - " + str(dataset)
                    times.update({method_and_dataset: time})

    df = pd.DataFrame(list(times.items()), columns=['method_dataset', 'value'])
    df[['method', 'dataset']] = df['method_dataset'].str.split(' - ', expand=True)
    df['dataset'] = df['dataset'].astype(int)

    df_mfe = pd.DataFrame(list(times_mfe.items()), columns=['method_category_dataset', 'value'])
    df_mfe[['method_category', 'dataset']] = df_mfe['method_category_dataset'].str.split(' - ', expand=True)
    df_mfe['dataset'] = df_mfe['dataset'].astype(int)

    df_pivot = df.pivot(index="dataset", columns="method", values="value")
    df_pivot_x = df_pivot.drop(columns=["MFE"], axis=1)
    df_pivot_mfe = df_mfe.pivot(index="dataset", columns="method_category", values="value")
    df_pivot = pd.concat([df_pivot_x, df_pivot_mfe], axis=1)
    df_pivot = df_pivot.sort_index()  # Sort by dataset ID

    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"        \begin{tabular*}{\textwidth}{@{\extracolsep{0.4em}} c|cccccc @{}}")
    latex_lines.append(r"        \toprule")
    latex_lines.append(r"        Dataset & Pandas & TabPFN & d2v & MFE general & MFE info-theory & MFE statistical \\")
    latex_lines.append(r"        \midrule")

    # Add table rows
    for dataset_id, row in formatted_df.iterrows():
        row_str = f"        {dataset_id} & " + " & ".join(row.values) + r" \\"
        latex_lines.append(row_str)

    # Finish LaTeX code
    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(r"    \caption{Time in seconds elapsed for the calculation of \metafeatures{} of the tested extractors per dataset}")
    latex_lines.append(r"    \label{tab:overview-time-mfe}")
    latex_lines.append(r"\end{table}")

    # Join all lines into one LaTeX string
    latex_code = "\n".join(latex_lines)

    # Print LaTeX table
    print(latex_code)

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
    plt.figure(figsize=(12, 7))
    for method in df_pivot.columns:
        plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
    plt.xlabel("Dataset ID")
    plt.xticks(rotation=90)  # or 90
    plt.yscale('log')  # Optional: only if values vary a lot
    plt.ylabel("Time in seconds")
    plt.title("Time for generating metafeatures by FE Method per Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Time_for_FE_per_Dataset.png", dpi=300)
    plt.show()

    df = df[df["method"] != "TabPFN"]
    time_per_method = df.groupby("method")["value"].sum().sort_values(ascending=False)
    average_time_per_method = df.groupby("method")["value"].mean().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(12, 7))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time per Method')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time per Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time per Method")
    plt.xticks(rotation=90, ha="right")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Time_for_FE_per_Method.png", dpi=300)
    plt.show()


    df_mfe = pd.DataFrame(list(times_mfe.items()), columns=['method_category_dataset', 'value'])
    df_mfe[['method_category', 'dataset']] = df_mfe['method_category_dataset'].str.split(' - ', expand=True)
    df_mfe[['method', 'category']] = df_mfe['method_category'].str.split('_', expand=True)
    df_mfe['dataset'] = df_mfe['dataset'].astype(int)

    time_per_category = df_mfe.groupby("category")["value"].sum().sort_values(ascending=False)
    time_issue_category_names = {"itemset", "landmarking"}
    memory_issue_category_names = {"complexity", "clustering", "concept"}

    time_issue_categories = pd.Series([1, 1], index=list(time_issue_category_names))
    memory_issue_categories = pd.Series([1, 1, 1], index=list(memory_issue_category_names))

    time_per_category = time_per_category._append(time_issue_categories)
    time_per_category = time_per_category._append(memory_issue_categories)
    colors = ['red' if cat in time_issue_categories + memory_issue_categories else 'skyblue' for cat in time_per_category.index]

    # Step 4: Plot total time per category (bar colors depend on memory issue)
    plt.figure(figsize=(12, 7))
    time_per_category.plot(kind='bar', color=colors, label='Total Time per Method')
    for idx, (category, value) in enumerate(time_per_category.items()):
        if category in time_issue_category_names:
            plt.text(idx, value * 1.1, 'Time Limit', color='red', ha='center', va='bottom',)
    for idx, (category, value) in enumerate(time_per_category.items()):
        if category in memory_issue_category_names:
            plt.text(idx, value * 1.1, 'Memory Limit', color='red', ha='center', va='bottom',)
    # Step 5: Add average time per category as overlay
    average_time_per_category = df_mfe.groupby("category")["value"].mean().reindex(time_per_category.index)
    average_time_per_category.plot(kind='bar', width=0.3, color='orange', label='Average Time per Category')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time per MFE Category")
    plt.xticks(rotation=90, ha="right")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Time_for_MFE_per_Category.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
