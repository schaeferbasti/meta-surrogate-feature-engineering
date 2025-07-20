import glob

import numpy as np
import openml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def insert_line_breaks(name, max_len=20):
    if len(name) > max_len:
        # Split into chunks of `max_len`, preserving words if possible
        parts = [name[i:i+max_len] for i in range(0, len(name), max_len)]
        return '\n'.join(parts)
    else:
        return name


def make_model_name_nice(df_pivot):
    model_names_nice = []
    model_names = df_pivot.columns
    for model_name in model_names:
        model_name = model_name.replace('pandas_', 'Pandas, ')
        model_name = model_name.replace('d2v_', 'Dataset2Vec, ')
        model_name = model_name.replace('tabpfn_', 'TabPFN, ')
        model_name = model_name.replace('MFE_general_', 'MFE (general), ')
        model_name = model_name.replace('MFE_statistical', 'MFE (statistical), ')
        model_name = model_name.replace('MFE_info-theory', 'MFE (info-theory), ')
        model_name = model_name.replace("MFE_{'general', 'info-theory'}", 'MFE (general, info-theory), ')
        model_name = model_name.replace("MFE_{'statistical', 'info-theory'}", 'MFE (statistical, info-theory), ')
        model_name = model_name.replace("MFE_{'info-theory', 'general'}", 'MFE (info-theory, general), ')
        model_name = model_name.replace("MFE_{'info-theory', 'statistical'}", 'MFE (info-theory, statistical), ')
        model_name = model_name.replace("MFE_{'general', 'statistical'}", 'MFE (general, statistical), ')
        model_name = model_name.replace("MFE_{'statistical', 'general'}", 'MFE (statistical, general), ')
        model_name = model_name.replace("MFE_{'general', 'statistical', 'info-theory'}", 'MFE (general, statistical, info-theory), ')
        model_name = model_name.replace('best', 'one-shot SM')
        model_name = model_name.replace('_one-shot', 'one-shot')
        model_name = model_name.replace('recursion', 'recursive SM')
        model_name = model_name.replace('_recursive', 'recursive')
        model_names_nice.append(model_name)
    df_pivot.columns = model_names_nice
    return df_pivot


def make_latex_table(df_pivot, without_openfe):
    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"    \tiny")
    if without_openfe:
        latex_lines.append(r"        \begin{tabular*}{\textwidth}{@{\extracolsep{0.4em}} c|cccccccccccccccccccccccc @{}}")
        latex_lines.append(r"        \toprule")
        latex_lines.append(r"        Dataset & \makecell{Best\\Random} & \makecell{MFE\\(general),\\one-shot SM} & \makecell{MFE\\(general),\\recursive SM} & \makecell{MFE\\(info-theory),\\one-shot SM} & \makecell{MFE\\(info-theory),\\recursive SM}  & \makecell{MFE\\(statistical),\\one-shot SM}  & \makecell{MFE\\(statistical),\\recursive SM}  & \makecell{MFE\\(general, info-theory),\\one-shot SM}  & \makecell{MFE\\(general, info-theory),\\recursive SM}  & \makecell{MFE\\(general, statistical),\\one-shot SM}  & \makecell{MFE\\(general, statistical),\\recursive SM}  & \makecell{MFE\\(info-theory, general),\\one-shot SM}  & \makecell{MFE\\(info-theory, general),\\recursive SM}  & \makecell{MFE\\(info-theory, statistical),\\one-shot SM}  & \makecell{MFE\\(info-theory, statistical),\\recursive SM}  & \makecell{MFE\\(statistical, general),\\one-shot SM}  & \makecell{MFE\\(statistical, general),\\recursive SM} & \makecell{MFE\\(statistical,info-theory),\\one-shot SM} & \makecell{MFE\\(statistical, info-theory),\\recursive SM} & \makecell{Original} & \makecell{Dataset2Vec,\\one-shot SM} & \makecell{Dataset2Vec,\\recursive SM} & \makecell{Pandas,\\one-shot SM} & \makecell{Pandas,\\recursive SM} \\")

    else:
        latex_lines.append(r"        \begin{tabular*}{\textwidth}{@{\extracolsep{0.4em}} c|ccccccccccccccccccccccccc @{}}")
        latex_lines.append(r"        \toprule")
        latex_lines.append(r"        Dataset & \makecell{Best\\Random} & \makecell{MFE\\(general),\\one-shot SM} & \makecell{MFE\\(general),\\recursive SM} & \makecell{MFE\\(info-theory),\\one-shot SM} & \makecell{MFE\\(info-theory),\\recursive SM}  & \makecell{MFE\\(statistical),\\one-shot SM}  & \makecell{MFE\\(statistical),\\recursive SM}  & \makecell{MFE\\(general, info-theory),\\one-shot SM}  & \makecell{MFE\\(general, info-theory),\\recursive SM}  & \makecell{MFE\\(general, statistical),\\one-shot SM}  & \makecell{MFE\\(general, statistical),\\recursive SM}  & \makecell{MFE\\(info-theory, general),\\one-shot SM}  & \makecell{MFE\\(info-theory, general),\\recursive SM}  & \makecell{MFE\\(info-theory, statistical),\\one-shot SM}  & \makecell{MFE\\(info-theory, statistical),\\recursive SM}  & \makecell{MFE\\(statistical, general),\\one-shot SM}  & \makecell{MFE\\(statistical, general),\\recursive SM} & \makecell{MFE\\(statistical,info-theory),\\one-shot SM} & \makecell{MFE\\(statistical, info-theory),\\recursive SM} & \makecell{Original} & \makecell{Dataset2Vec,\\one-shot SM} & \makecell{Dataset2Vec,\\recursive SM} & \makecell{Pandas,\\one-shot SM} & \makecell{Pandas,\\recursive SM} & \makecell{OpenFE} \\")

    latex_lines.append(r"        \midrule")

    # Add table rows
    for dataset_id, row in formatted_df.iterrows():
        row_str = f"        {dataset_id} & " + " & ".join(row.values) + r" \\ \midrule"
        latex_lines.append(row_str)

    # Finish LaTeX code
    latex_lines.append(r"    \end{tabular*}")
    if without_openfe:
        latex_lines.append(r"    \caption{Test error of the model on the feature-engineered datasets of the \sm{} approaches using \metafeatures{} of the tested extractors, on the best randomly feature-engineered datasets and on the original datasets}")
        latex_lines.append(r"    \label{tab:test_without_openfe}")
    else:
        latex_lines.append(r"    \caption{Test error of the model on the feature-engineered datasets of the \sm{} approaches using \metafeatures{} of the tested extractors, on the best randomly feature-engineered datasets, on the original datasets, and on the datasets feature-engineered with \gls{OpenFE}}")
        latex_lines.append(r"    \label{tab:test}")
    latex_lines.append(r"\end{table}")

    latex_code = "\n".join(latex_lines)

    print(latex_code)


def make_latex_tables_split(df_pivot, without_openfe, columns_per_table=6):
    import math

    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    method_columns = df_pivot.columns.tolist()
    total_tables = 4

    for table_idx in range(total_tables):
        start_col = table_idx * columns_per_table
        # Fix: Add all remaining columns to the last table
        if table_idx == total_tables - 1:
            end_col = len(method_columns)
        else:
            end_col = start_col + columns_per_table

        current_columns = method_columns[start_col:end_col]

        latex_lines = []
        latex_lines.append(r"\begin{table}[h!]")
        latex_lines.append(r"    \footnotesize")

        column_format = "c|" + "c" * len(current_columns)
        if without_openfe:
            latex_lines.append(fr"    \begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{0.2em}}}} {column_format} @{{}}}}")
        else:
            if table_idx == total_tables - 1:
                latex_lines.append(fr"    \begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{-0.5em}}}} {column_format} @{{}}}}")
            else:
                latex_lines.append(fr"    \begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{0.2em}}}} {column_format} @{{}}}}")
        latex_lines.append(r"        \toprule")

        header_cells = ["Dataset"]
        for col in current_columns:
            escaped_col = col.replace(", ", ",\\\\").replace(" ", "\\\\")  # Optional: better breaking
            header_cells.append(f"\\makecell{{{escaped_col}}}")
        latex_lines.append("        " + " & ".join(header_cells) + r" \\")
        latex_lines.append(r"        \midrule")

        for dataset_id, row in formatted_df.iterrows():
            values = [row[col] for col in current_columns]
            row_str = f"        {dataset_id} & " + " & ".join(values) + r" \\"
            latex_lines.append(row_str)

        latex_lines.append(r"        \bottomrule")
        latex_lines.append(r"    \end{tabular*}")

        base_caption = "Test error of the model on the feature-engineered datasets..."
        label_prefix = "tab:test_without_openfe" if without_openfe else "tab:test_with_openfe"
        latex_lines.append(fr"    \caption{{{base_caption} (Part {table_idx + 1})}}")
        latex_lines.append(fr"    \label{{{label_prefix}_part{table_idx + 1}}}")
        latex_lines.append(r"\end{table}")
        latex_lines.append("")

        print("\n".join(latex_lines))



def get_data(result_files):
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
    df_pivot_val = make_model_name_nice(df_pivot_val)
    df_pivot_test = df_all.pivot(index="dataset", columns="origin", values="error_test")
    df_pivot_test = df_pivot_test.sort_index()  # Sort by dataset ID
    df_pivot_test = make_model_name_nice(df_pivot_test)
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
    #dataset_list_wrapped = [insert_line_breaks(name, max_len=15) for name in dataset_list]
    dataset_list_wrapped = datasets.tolist()
    return dataset_list_wrapped, df_pivot_test, df_pivot_val


def plot_score_graph(dataset_list_wrapped, df_pivot, name):
    if "only_pandas" in name or "openfe_pandas" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
        large_plot = False
        without_openfe = False
    elif "without_OpenFE" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
        large_plot = True
        without_openfe = True
        df_pivot = df_pivot.drop(columns=["OpenFE"])
        if score_type == "test":
            make_latex_tables_split(df_pivot, without_openfe)
    else:
        if name == "Val":
            score_type = "validation"
        else:
            score_type = "test"
        large_plot = True
        without_openfe = False
        column_to_move = df_pivot.pop("OpenFE")
        df_pivot.insert(len(df_pivot.columns), "OpenFE", column_to_move)
        if score_type == "test":
            make_latex_tables_split(df_pivot, without_openfe)
    if without_openfe:
        colors = cm.get_cmap('nipy_spectral')
        color_list = [colors(i) for i in np.linspace(0, 0.95, len(df_pivot.columns))]
    else:
        colors = cm.get_cmap('nipy_spectral', len(df_pivot.columns))

    dataset_list_wrapped = df_pivot.index.tolist()
    if large_plot:
        plt.figure(figsize=(12, 10))
        if without_openfe:
            for idx, method in enumerate(df_pivot.columns):
                plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method, color=color_list[idx])
        else:
            for idx, method in enumerate(df_pivot.columns):
                plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method, color=colors(idx))
    else:
        plt.figure(figsize=(12, 7))
        for method in df_pivot.columns:
            plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
    plt.xlabel("Dataset")
    plt.xticks(rotation=90)  # or 45
    plt.ylabel(score_type.title() + " error")
    plt.title(score_type.title() + " error of the model on the feature-engineered datasets, the original and the randomly feature-engineered datasets")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../Result_Analysis/test_analysis/Graph_" + name + ".png")
    plt.show()


def plot_count_best(df_pivot_val, df_pivot_test, name):
    minValueIndex_val = df_pivot_val.idxmin(axis=1).value_counts()
    minValueIndex_test = df_pivot_test.idxmin(axis=1).value_counts()
    # Plot
    plt.figure(figsize=(12, 7))
    minValueIndex_val.plot(kind='bar', color='skyblue', label='Number of datasets with the lowest validation error')
    minValueIndex_test.plot(kind='bar', width=0.3, color='darkblue', label='Number of datasets with the lowest test error')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Number of datasets")
    plt.title("Count of the lowest validation and test error of the model")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.savefig("../Result_Analysis/test_analysis/Count_Best_" + name + "bar.png")
    plt.show()


def plot_avg_percentage_impr(baseline_col, df_pivot, name, only_pandas=False):
    if "only_pandas" in name or "openfe_pandas" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
    elif "without_OpenFE" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
    else:
        if name == "Val":
            score_type = "validation"
        else:
            score_type = "test"
    improvement = pd.DataFrame()
    for method in df_pivot.columns:
        if method == baseline_col:
            continue
        calc_loss_improvement = ((df_pivot[baseline_col] - df_pivot[method]) / df_pivot[baseline_col]) * 100
        # calc = ((df_pivot[baseline_col] - df_pivot[method]) / df_pivot[method]) * 100
        # f1 = ((df_pivot[method] - df_pivot[baseline_col]) / df_pivot[baseline_col]) * 100
        # increase_in_error = ((df_pivot[method] - df_pivot[baseline_col]) / df_pivot[baseline_col]) * 100
        improvement[method] = calc_loss_improvement
    avg_improvement = improvement.mean().sort_values(ascending=False)
    # for i, val in enumerate(avg_improvement_test):
    #    plt.text(i, val + (1 if val >= 0 else -1), f"{val:.2f}%", ha='center', va='bottom' if val >= 0 else 'top')
    plt.figure(figsize=(12, 7))
    # avg_improvement_test.plot(kind="bar", color="skyblue")
    bars = avg_improvement.plot(kind="bar", color="skyblue")
    if only_pandas:
        for i, val in enumerate(avg_improvement):
            y = 2  # adjust offset for spacing
            plt.text(i, y, f"{val:.2f}%", ha='center', va='top' if val >= 0 else 'bottom', color='black')
    else:
        for i, val in enumerate(avg_improvement):
            y = -1 if val >= 0 else 0  # adjust offset for spacing
            plt.text(i, y, f"{val:.2f}%", ha='center', va='top' if val >= 0 else 'bottom', color='black')
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Average percentage error reduction of the " + score_type + " error of the model\nin relation to the " + score_type + " error of the model on the original datasets")
    plt.xlabel("Method")
    plt.ylabel("Percentage error reduction of the " + score_type + " error\nin relation to the " + score_type + " error on the original datasets")
    plt.xticks(rotation=90, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../Result_Analysis/test_analysis/Average_Percentage_Improvement_" + name + ".png")
    plt.show()


def plot_boxplot_percentage_impr(baseline_col, df_pivot, name):
    if "only_pandas" in name or "openfe_pandas" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
    elif "without_OpenFE" in name:
        score_type = name.split("_")[0]
        if score_type == "Val":
            score_type = "validation"
        else:
            score_type = "test"
    else:
        if name == "Val":
            score_type = "validation"
        else:
            score_type = "test"
    improvement_test = pd.DataFrame()
    for method in df_pivot.columns:
        if method == baseline_col:
            continue
        improvement = (df_pivot[baseline_col] - df_pivot[method]) / df_pivot[baseline_col] * 100
        improvement_test[method] = improvement

    # Sort methods by mean improvement (descending)
    method_order = improvement_test.median().sort_values(ascending=False).index.tolist()
    improvement_test = improvement_test[method_order]

    # Plot
    plt.figure(figsize=(12, 7))
    improvement_test.boxplot(column=method_order, grid=True)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Distribution of the percentage error reduction of the " + score_type + " error of the model\nin relation to the " + score_type + " error of the model on the original datasets")
    plt.xlabel("Method")
    plt.ylabel("Percentage error reduction of the " + score_type + " error\nin relation to the " + score_type + " error on the original datasets")
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.savefig(f"../Result_Analysis/test_analysis/Boxplot_Percentage_Improvement_{name}.png")
    plt.show()


def test_analysis():
    baseline_col = "Original"
    result_files = glob.glob("test_results/Result_*.parquet")
    dataset_list_wrapped, df_pivot_test, df_pivot_val = get_data(result_files)
    df_pivot_test.drop(columns="Best Random", inplace=True)
    df_pivot_val.drop(columns="Best Random", inplace=True)

    # Plot
    plot_score_graph(dataset_list_wrapped, df_pivot_val, "Val")
    plot_score_graph(dataset_list_wrapped, df_pivot_test, "Test")

    plot_score_graph(dataset_list_wrapped, df_pivot_val, "Val_without_OpenFE")
    plot_score_graph(dataset_list_wrapped, df_pivot_test, "Test_without_OpenFE")

    plot_count_best(df_pivot_val, df_pivot_test, "")
    plot_avg_percentage_impr(baseline_col, df_pivot_val, "Val")
    plot_avg_percentage_impr(baseline_col, df_pivot_test, "Test")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_val, "Val")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_test, "Test")

    # Drop OpenFE column to compare MFE approaches
    df_pivot_val_without_OpenFE = df_pivot_val
    df_pivot_test_without_OpenFE = df_pivot_test
    df_pivot_val_without_OpenFE.drop(columns=["OpenFE"], inplace=True)
    df_pivot_test_without_OpenFE.drop(columns=["OpenFE"], inplace=True)
    # Plot again
    plot_count_best(df_pivot_val_without_OpenFE, df_pivot_test_without_OpenFE, "without_OpenFE_")
    plot_avg_percentage_impr(baseline_col, df_pivot_val_without_OpenFE, "Val_without_OpenFE")
    plot_avg_percentage_impr(baseline_col, df_pivot_test_without_OpenFE, "Test_without_OpenFE")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_val_without_OpenFE, "Val_without_OpenFE")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_test_without_OpenFE, "Test_without_OpenFE")

    # Drop everything but pandas & original columns to compare SM approaches
    df_pivot_val_pandas = df_pivot_val[["Pandas, one-shot SM", "Pandas, recursive SM", "Original"]]
    df_pivot_test_pandas = df_pivot_test[["Pandas, one-shot SM", "Pandas, recursive SM", "Original"]]
    # Plot
    plot_avg_percentage_impr(baseline_col, df_pivot_val_pandas, "Val_only_pandas", True)
    plot_avg_percentage_impr(baseline_col, df_pivot_test_pandas, "Test_only_pandas", True)
    plot_boxplot_percentage_impr(baseline_col, df_pivot_val_pandas, "Val_only_pandas")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_test_pandas, "Test_only_pandas")

    # Drop everything but pandas columns to compare SM approaches
    df_pivot_val_pandas = df_pivot_val_pandas[["Pandas, one-shot SM", "Pandas, recursive SM"]]
    df_pivot_test_pandas = df_pivot_test_pandas[["Pandas, one-shot SM", "Pandas, recursive SM"]]
    # Plot again
    plot_count_best(df_pivot_val_pandas, df_pivot_test_pandas, "only_pandas_")
    plot_score_graph(dataset_list_wrapped, df_pivot_val_pandas, "Val_only_pandas")
    plot_score_graph(dataset_list_wrapped, df_pivot_test_pandas, "Test_only_pandas")

    dataset_list_wrapped, df_pivot_test, df_pivot_val = get_data(result_files)

    df_pivot_val_openfe = df_pivot_val[["OpenFE", "Pandas, recursive SM", "Original"]]
    df_pivot_test_openfe = df_pivot_test[["OpenFE", "Pandas, recursive SM", "Original"]]
    # Plot
    plot_avg_percentage_impr(baseline_col, df_pivot_val_openfe, "Val_openfe_pandas", True)
    plot_avg_percentage_impr(baseline_col, df_pivot_test_openfe, "Test_openfe_pandas", True)
    plot_boxplot_percentage_impr(baseline_col, df_pivot_val_openfe, "Val_openfe_pandas")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_test_openfe, "Test_openfe_pandas")

    # Drop everything but pandas columns to compare SM approaches
    df_pivot_val_openfe = df_pivot_val_openfe[["OpenFE", "Pandas, recursive SM"]]
    df_pivot_test_openfe = df_pivot_test_openfe[["OpenFE", "Pandas, recursive SM"]]
    # Plot again
    plot_count_best(df_pivot_val_openfe, df_pivot_test_openfe, "openfe_pandas_")
    plot_score_graph(dataset_list_wrapped, df_pivot_val_openfe, "Val_openfe_pandas")
    plot_score_graph(dataset_list_wrapped, df_pivot_test_openfe, "Test_openfe_pandas")


if __name__ == "__main__":
    test_analysis()
