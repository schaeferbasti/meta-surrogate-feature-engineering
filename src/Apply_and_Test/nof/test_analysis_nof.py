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
    return dataset_list_wrapped, df_pivot_test, df_pivot_val


def plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_val, name):
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
    plt.savefig("analysis/Autogluon_" + name + "_Score_by_FE_Method.png")
    plt.show()


def plot_random_vs_me_graph(dataset_list_wrapped, df_pivot_val, name):
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
        elif method.startswith("Original"):
            plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)
    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon " + name + " Score (flipped)")
    plt.title("Autogluon " + name + " Score by FE Method per Dataset (Best random vs. our method)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("analysis/Random_vs_Me_" + name + "_graph.png")
    plt.show()


def plot_count_best(df_pivot_val, df_pivot_test, name):
    minValueIndex_val = df_pivot_val.idxmin(axis=1).value_counts()
    minValueIndex_test = df_pivot_test.idxmin(axis=1).value_counts()
    # Plot
    plt.figure(figsize=(10, 6))
    minValueIndex_val.plot(kind='bar', color='skyblue', label='Count of best val result on number of datasets')
    minValueIndex_test.plot(kind='bar', width=0.3, color='darkblue',
                            label='Count of best test result on number of datasets')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Best result on number of datasets")
    plt.title("Which method is the best?")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("analysis/Count_Best_" + name + "bar.png")
    plt.show()


def plot_avg_percentage_impr(baseline_col, df_pivot, name, only_pandas=False):
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
    plt.figure(figsize=(10, 6))
    # avg_improvement_test.plot(kind="bar", color="skyblue")
    bars = avg_improvement.plot(kind="bar", color="skyblue")
    if only_pandas:
        for i, val in enumerate(avg_improvement):
            y = 2  # adjust offset for spacing
            plt.text(i, y, f"{val:.2f}%", ha='center', va='top' if val >= 0 else 'bottom', color='black')
    else:
        for i, val in enumerate(avg_improvement):
            y = -0.1 if val >= 0 else 0  # adjust offset for spacing
            plt.text(i, y, f"{val:.2f}%", ha='center', va='top' if val >= 0 else 'bottom', color='black')
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Average Percentage Improvement " + name + " over original Dataset")
    plt.xlabel("Method")
    plt.ylabel("Average Improvement (%)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("analysis/Average_Percentage_Improvement_" + name + ".png")
    plt.show()


def plot_boxplot_percentage_impr(baseline_col, df_pivot, name):
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
    plt.figure(figsize=(10, 6))
    improvement_test.boxplot(column=method_order, grid=True)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Distribution of % Improvement " + name + " over original Dataset")
    plt.xlabel("Method")
    plt.ylabel("Improvement (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"analysis/Boxplot_Percentage_Improvement_{name}.png")
    plt.show()


def test_analysis():
    baseline_col = "Original"
    result_files = glob.glob("test_results/Result_*.parquet")
    dataset_list_wrapped, df_pivot_test, df_pivot_val = get_data(result_files)

    # Plot
    plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_val, "Val")
    plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_test, "Test")

    plot_random_vs_me_graph(dataset_list_wrapped, df_pivot_val, "Val")
    plot_random_vs_me_graph(dataset_list_wrapped, df_pivot_test, "Test")

    plot_count_best(df_pivot_val, df_pivot_test, "")
    plot_avg_percentage_impr(baseline_col, df_pivot_val, "Val")
    plot_avg_percentage_impr(baseline_col, df_pivot_test, "Test")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_val, "Val")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_test, "Test")

    # Drop everything but pandas & original columns to compare SM approaches
    df_pivot_val = df_pivot_val[["pandas_NOF_001_best", "pandas_NOF_010_best", "pandas_NOF_025_best", "pandas_NOF_050_best", "pandas_NOF_075_best", "pandas_NOF_100_best", "pandas_NOF_125_best", "pandas_NOF_150_best", "pandas_NOF_200_best", "Original"]]
    df_pivot_test = df_pivot_test[["pandas_NOF_001_best", "pandas_NOF_010_best", "pandas_NOF_025_best", "pandas_NOF_050_best", "pandas_NOF_075_best", "pandas_NOF_100_best", "pandas_NOF_125_best", "pandas_NOF_150_best", "pandas_NOF_200_best", "Original"]]
    # Plot
    plot_avg_percentage_impr(baseline_col, df_pivot_val, "Val_only_pandas")
    plot_avg_percentage_impr(baseline_col, df_pivot_test, "Test_only_pandas")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_val, "Val_only_pandas")
    plot_boxplot_percentage_impr(baseline_col, df_pivot_test, "Test_only_pandas")

    # Drop everything but pandas columns to compare SM approaches
    df_pivot_val = df_pivot_val[["pandas_NOF_001_best", "pandas_NOF_010_best", "pandas_NOF_025_best", "pandas_NOF_050_best", "pandas_NOF_075_best", "pandas_NOF_100_best", "pandas_NOF_125_best", "pandas_NOF_150_best", "pandas_NOF_200_best"]]
    df_pivot_test = df_pivot_test[["pandas_NOF_001_best", "pandas_NOF_010_best", "pandas_NOF_025_best", "pandas_NOF_050_best", "pandas_NOF_075_best", "pandas_NOF_100_best", "pandas_NOF_125_best", "pandas_NOF_150_best", "pandas_NOF_200_best"]]
    # Plot again
    plot_count_best(df_pivot_val, df_pivot_test, "only_pandas_")
    plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_val, "Val_only_pandas")
    plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_test, "Test_only_pandas")


if __name__ == "__main__":
    test_analysis()
