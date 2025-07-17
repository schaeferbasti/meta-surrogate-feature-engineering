import glob
import re

import openml
import pandas as pd
import matplotlib.pyplot as plt


def insert_line_breaks(name, max_len=20):
    if len(name) > max_len:
        parts = [name[i:i+max_len] for i in range(0, len(name), max_len)]
        return '\n'.join(parts)
    else:
        return name

def extract_mwi(name):
    match = re.search(r'MWI_([0-9]*\.?[0-9]+)', name)
    if match:
        return float(match.group(1))
    elif name == "Original":
        return None  # Or the original number of features
    else:
        return None  # Default fallback


def get_data(result_files):
    all_results = []
    for result_file in result_files:
        df = pd.read_parquet(result_file)
        dataset_id = int(result_file.split("Result_")[1].split(".parquet")[0])
        df["origin"] = df["origin"].apply(lambda x: "Best Random" if str(x).startswith("Random") else x)
        all_results.append(df)
    df_all = pd.concat(all_results, ignore_index=True)
    df_all = df_all.drop_duplicates()
    df_all["error_val"] = - df_all["score_val"]
    df_all["error_test"] = - df_all["score_test"]
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
    return dataset_list_wrapped, df_pivot_test, df_pivot_val, df_all


def plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_val, name):
    plt.figure(figsize=(12, 6))
    for method in df_pivot_val.columns:
        plt.plot(dataset_list_wrapped, df_pivot_val[method], marker='o', label=method)
    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.ylabel("Autogluon Val Score (flipped)")
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
        improvement[method] = calc_loss_improvement
    avg_improvement = improvement.mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
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


def plot_correlation_mwi(df, dataset_n_features, name):
    df["dataset"] = df["dataset"].astype(int)
    df["n_features_original"] = df["dataset"].map(dataset_n_features)
    df["MWI"] = df["origin"].apply(extract_mwi)

    # Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=df["MWI"],
        y=df["n_features_original"],
        c=df["score_test"],  # Color = number of original features
        cmap='viridis',
        s=80,
        edgecolor='black',
        alpha=0.8
    )
    plt.colorbar(scatter, label="Test Score")
    plt.title("Performance vs. MWI (colored by # original features)")
    plt.xlabel("Number of Output Features (MWI)")
    plt.ylabel("Original Number of Features")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"analysis/Correlation_MWI_{name}.png")
    plt.show()

    df_clean = df.dropna(subset=["dataset", "score_test", "MWI", "n_features_original"])
    idx_best = df_clean.groupby("dataset")["score_test"].idxmax()
    df_clean_2 = df_clean.drop(idx_best, axis=0)
    idx_2_best = df_clean_2.groupby("dataset")["score_test"].idxmax()
    df_clean_3 = df_clean_2.drop(idx_2_best, axis=0)
    idx_3_best = df_clean_3.groupby("dataset")["score_test"].idxmax()
    best_rows = df_clean.loc[idx_best].reset_index(drop=True)
    best_rows_2 = df_clean_2.loc[idx_2_best].reset_index(drop=True)
    best_rows_3 = df_clean_3.loc[idx_3_best].reset_index(drop=True)

    # Step 3: Plot the relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(
        best_rows["MWI"],
        best_rows["n_features_original"],
        color="royalblue",
        edgecolor="black",
        s=100,
        alpha=0.8
    )
    plt.scatter(
        best_rows_2["MWI"],
        best_rows_2["n_features_original"],
        color="royalblue",
        edgecolor="black",
        s=100,
        alpha=0.8
    )
    """
    plt.scatter(
        best_rows_3["MWI"],
        best_rows_3["n_features_original"],
        color="royalblue",
        edgecolor="black",
        s=100,
        alpha=0.8
    )
    """
    plt.xlabel("Number of Output Features (MWI)")
    plt.ylabel("Number of Original Features")
    plt.title("Best Score Rows: MWI vs. Original Feature Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("analysis/MWI_vs_OriginalFeatures_best.png")
    plt.show()

    best_scores_by_mwi = best_rows.groupby("MWI", as_index=False)["score_test"].max()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        best_scores_by_mwi["MWI"],
        best_scores_by_mwi["score_test"],
        marker='o',
        color='royalblue',
        linestyle='-',
        linewidth=2
    )

    plt.xlabel("Number of Output Features (MWI)")
    plt.ylabel("Best Test Score")
    plt.title("Best Test Score per MWI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("analysis/BestScore_per_MWI.png")
    plt.show()

    mwi_counts = best_rows["MWI"].value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    plt.bar(mwi_counts.index, mwi_counts.values, color='skyblue', edgecolor='black')

    plt.xlabel("Number of Output Features (MWI)")
    plt.ylabel("Count of Best Scores")
    plt.title("Frequency of MWI in Best-Scoring Pipelines")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("analysis/Frequency_of_Best_MWI.png")
    plt.show()


def test_analysis():
    dataset_n_features = {
        359963: 20,
        359968: 21,
        359971: 31,
        359972: 21,
        359974: 12,
        359975: 37,
        359979: 10,
        359981: 7,
        359982: 17,
        359983: 15,
        359987: 10,
        359992: 12,
        359993: 20,
    }
    baseline_col = "Original"
    result_files = glob.glob("test_results/Result_*.parquet")
    dataset_list_wrapped, df_pivot_test, df_pivot_val, df_all = get_data(result_files)

    df_all.groupby('origin')['score_test'].idxmax().apply(lambda i: df_all.loc[i, ['origin', 'score_test']])
    """
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
    """

    # Drop everything but pandas & original columns to compare SM approaches
    # df_pivot_val_pandas = df_pivot_val[["pandas_MWI_001_best", "pandas_MWI_010_best", "pandas_MWI_025_best", "pandas_MWI_050_best", "pandas_MWI_075_best", "pandas_MWI_100_best", "pandas_MWI_125_best", "pandas_MWI_150_best", "pandas_MWI_200_best", "Original"]]
    df_pivot_test_pandas = df_pivot_test[["pandas_MWI_0.1_recursion", "pandas_MWI_0.2_recursion", "pandas_MWI_0.3_recursion", "pandas_MWI_0.05_recursion", "pandas_MWI_0.15_recursion", "pandas_MWI_0.5_recursion", "pandas_MWI_1.0_recursion", "Original"]]  # "pandas_MWI_0_recursion"
    # plot_avg_percentage_impr(baseline_col, df_pivot_val_pandas, "Val_only_pandas")
    plot_avg_percentage_impr(baseline_col, df_pivot_test_pandas, "Test_only_pandas")
    df_pivot_test_mfe_statistical = df_pivot_test[["MFE_statistical_MWI_0.1_recursion", "MFE_statistical_MWI_0.2_recursion", "MFE_statistical_MWI_0.3_recursion", "MFE_statistical_MWI_0.05_recursion", "MFE_statistical_MWI_0.15_recursion", "Original"]]  # "MFE_statistical_MWI_0_recursion", "MFE_statistical_MWI_0.5_recursion", "MFE_statistical_MWI_1.0_recursion"
    plot_avg_percentage_impr(baseline_col, df_pivot_test_mfe_statistical, "Test_only_mfe_statistical")
    df_pivot_test_mfe_info = df_pivot_test[["MFE_info-theory_MWI_0.1_recursion", "MFE_info-theory_MWI_0.2_recursion", "MFE_info-theory_MWI_0.3_recursion", "MFE_info-theory_MWI_0.05_recursion", "MFE_info-theory_MWI_0.15_recursion", "Original"]]  # "MFE_info-theory_MWI_0_recursion", "MFE_info-theory_MWI_0.5_recursion", "MFE_info-theory_MWI_1.0_recursion",
    plot_avg_percentage_impr(baseline_col, df_pivot_test_mfe_info, "Test_only_mfe_info-theory")
    df_pivot_test_mfe_general = df_pivot_test[["MFE_general_MWI_0.1_recursion", "MFE_general_MWI_0.05_recursion", "MFE_general_MWI_0.15_recursion", "Original"]]  #
    plot_avg_percentage_impr(baseline_col, df_pivot_test_mfe_general, "Test_only_mfe_general")
    df_pivot_test_mfe_general_info = df_pivot_test[["MFE_{\'general\', \'info_MWI_0.0_recursion", "MFE_{\'general\', \'info_MWI_0.05_recursion", "MFE_{\'general\', \'info_MWI_0.15_recursion", "MFE_{\'general\', \'info_MWI_0.2_recursion", "MFE_{\'general\', \'info_MWI_0.3_recursion", "Original"]]  # "MFE_{\'general\', \'info_MWI_0.5_recursion", "MFE_{\'general\', \'info_MWI_1.0_recursion", "MFE_{\'general\', \'info_MWI_0_recursion",
    plot_avg_percentage_impr(baseline_col, df_pivot_test_mfe_general_info, "Test_only_mfe_general_info")
    df_pivot_test_mfe_general_statistical = df_pivot_test[["MFE_{\'general\', \'statistical\'}_MWI_0.0_recursion", "MFE_{\'general\', \'statistical\'}_MWI_0.05_recursion", "MFE_{\'general\', \'statistical\'}_MWI_0.15_recursion", "MFE_{\'general\', \'statistical\'}_MWI_0.2_recursion", "MFE_{\'general\', \'statistical\'}_MWI_0.3_recursion", "Original"]]  # "MFE_{\'general\', \'statistical\'}_MWI_0_recursion", "MFE_{\'general\', \'statistical\'}_MWI_0.5_recursion", "MFE_{\'general\', \'statistical\'}_MWI_1.0_recursion",
    plot_avg_percentage_impr(baseline_col, df_pivot_test_mfe_general_statistical, "Test_only_mfe_general_statistical")
    df_pivot_test_mfe_statistical_info = df_pivot_test[["MFE_{\'statistical\', \'info_MWI_0.0_recursion", "MFE_{\'statistical\', \'info_MWI_0.05_recursion", "MFE_{\'statistical\', \'info_MWI_0.15_recursion", "MFE_{\'statistical\', \'info_MWI_0.2_recursion", "MFE_{\'statistical\', \'info_MWI_0.3_recursion", "Original", "MFE_{\'statistical\', \'info_MWI_0.5_recursion"]]  # "MFE_{\'statistical\', \'info_MWI_0_recursion", "MFE_{\'statistical\', \'info_MWI_1.0_recursion"
    plot_avg_percentage_impr(baseline_col, df_pivot_test_mfe_statistical_info, "Test_only_mfe_general_statistical")



    # Plot
    # plot_correlation_mwi(df_all, dataset_n_features, "")
    # plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_test, "Test")
    """
    plot_autogluon_score_graph(dataset_list_wrapped, df_pivot_val_pandas, "Val_only_pandas")

    plot_random_vs_me_graph(dataset_list_wrapped, df_pivot_val_pandas, "Val_only_pandas")
    plot_random_vs_me_graph(dataset_list_wrapped, df_pivot_test_pandas, "Test_only_pandas")
    """

if __name__ == "__main__":
    test_analysis()
