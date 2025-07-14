import glob

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd

from src.Result_Analysis.test_analysis.test_analysis import insert_line_breaks
from src.utils.get_data import split_data, get_openml_dataset_split


def plot_delta_r2_graph(dataset_list_wrapped, df_pivot, name):
    plt.figure(figsize=(12, 6))
    for method in df_pivot.columns:
        plt.plot(dataset_list_wrapped, df_pivot[method], marker='o', label=method)
    plt.xlabel("Dataset ID")
    plt.xticks(rotation=45)  # or 90
    plt.yscale("log")
    plt.ylabel("DeltaR2 Score")
    plt.title("DeltaR2 Score " + name + " by FE Method per Dataset")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if name == "":
        plt.savefig("Delta_R2_Score_by_FE_Method.png")
    else:
        plt.savefig("Delta_R2_Score_by_FE_Method_" + name.replace(" ", "_") + ".png")
    plt.show()


"""
def semipartial_corr_formula(X_base, X_new, Y):
    # Compute semipartial correlation for each feature in X_new with Y, controlling for X_base.
    X_base = np.asarray(X_base)
    Y = np.asarray(Y).ravel()
    results = {}
    for col in X_new.columns:
        x_new = X_new[col].to_numpy().ravel()
        # Correlation between Y and new feature
        r_yx1 = np.corrcoef(Y, x_new)[0, 1]
        # Correlation between Y and base features (multiple)
        r_yx2 = np.mean([np.corrcoef(Y, X_base[:, i])[0, 1] for i in range(X_base.shape[1])])
        # Correlation between new feature and base features (average)
        r_x1x2 = np.mean([np.corrcoef(x_new, X_base[:, i])[0, 1] for i in range(X_base.shape[1])])
        # Semipartial correlation
        numerator = r_yx1 - r_yx2 * r_x1x2
        denominator = np.sqrt(1 - r_x1x2 ** 2)
        r_semipartial = numerator / denominator if denominator != 0 else np.nan
        results[col] = r_semipartial
    return results
"""


def identify_suppressors(X_base, X_train_fe, Y, corr_matrix_pd, delta_r2, threshold_low=0.1, threshold_high=0.3):
    suppressors = []

    for col in X_train_fe.columns:
        r_yx = corr_matrix_pd.loc[col, "target"]  # correlation with Y
        r_xx = corr_matrix_pd.loc[col, X_base.columns].abs().max()  # strongest correlation with any X_base

        if abs(r_yx) < threshold_low and r_xx > threshold_high:
            suppressors.append({
                'feature': col,
                'r_with_Y': r_yx,
                'max_r_with_X_base': r_xx,
                'delta_r2': delta_r2
            })

    return pd.DataFrame(suppressors)


def calculate_incremental_validity(Y, X_base, X_train_fe):
    Y_df = pd.DataFrame(Y, columns=["target"])
    all_vars = pd.concat([X_train_fe, Y_df], axis=1)
    try:
        corr_matrix = all_vars.corr().values
    except ValueError:
        Y = pd.Series(pd.factorize(Y)[0], index=X_base.index)
        Y_df = pd.DataFrame(Y, columns=["target"])
        all_vars = pd.concat([X_train_fe, Y_df], axis=1)
        try:
            corr_matrix = all_vars.corr().values
        except ValueError:
            X_base = X_base.apply(lambda x: pd.factorize(x)[0])
            X_train_fe = X_train_fe.apply(lambda x: pd.factorize(x)[0])
            all_vars = pd.concat([X_train_fe, Y_df], axis=1)
            corr_matrix = all_vars.corr().values
    corr_matrix = np.nan_to_num(corr_matrix)
    corr_matrix_pd = pd.DataFrame(corr_matrix, columns=all_vars.columns, index=all_vars.columns)

    k = X_base.shape[1]
    r_yx_base = corr_matrix[-1, :k]  # Correlations Y with X_base
    R_xx_base = corr_matrix[:k, :k]  # Correlations among X_base

    r2_base = r_yx_base @ np.linalg.pinv(R_xx_base) @ r_yx_base.T

    r_yx_full = corr_matrix[-1, :-1]  # Y with all predictors
    R_xx_full = corr_matrix[:-1, :-1]  # All predictors with each other

    r2_full = r_yx_full @ np.linalg.pinv(R_xx_full) @ r_yx_full.T

    delta_r2 = r2_full - r2_base
    return corr_matrix_pd, r2_base, r2_full, delta_r2


def plot_incremental_validity(delta_r2, r2_base, r2_full):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw circle for base features
    base = plt.Circle((0.4, 0.5), 0.3, color='skyblue', alpha=0.5, label='Base Features')
    # Draw circle for new features (shifted to the right)
    new = plt.Circle((0.6, 0.5), 0.2, color='salmon', alpha=0.5, label='New Features')
    ax.add_patch(base)
    ax.add_patch(new)
    # Annotate values
    plt.text(0.3, 0.5, f"{r2_base:.2f}", fontsize=12, ha='center', va='center')
    plt.text(0.7, 0.5, f"{delta_r2:.2f}", fontsize=12, ha='center', va='center')
    plt.text(0.5, 0.5, f"{r2_full:.2f}", fontsize=12, ha='center', va='center', fontweight='bold')
    # Final formatting
    plt.title("Explained Variance in Target (Venn-style Diagram)")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()


def statistical_analysis():
    path = "../../Apply_and_Test/test_data/"
    data_files = glob.glob(path + "FE_*.parquet")
    data_files.sort()
    columns = ["Dataset", "Method", "DeltaR2", "Count_Suppressor_Candidates", "Number_of_Features", "DeltaR2_Normalized"]
    delta_r2_df = pd.DataFrame(columns=columns)
    delta_r2_non_suppressor_df = pd.DataFrame(columns=columns)
    try:
        delta_r2_df = pd.read_parquet("delta_r2.parquet")
    except FileNotFoundError:
        for data_file in data_files:
            print("Processing ", data_file)
            data = pd.read_parquet(path + data_file)
            dataset = int(data_file.split("FE_")[1].split("_")[0])
            method = data_file.split(".parquet")[0].split(str(dataset) + "_")[1]
            X_train_fe, y_train_fe, X_test_fe, y_test_fe = split_data(data, "target")
            X_train, y_train, X_test, y_test = get_openml_dataset_split(dataset)
            features = X_train_fe.drop(X_train.columns, axis=1)
            number_of_features = features.shape[1]
            if number_of_features > 0:
                X_train = X_train_fe.drop(features.columns, axis=1)
                corr_matrix_pd, r2_base, r2_full, delta_r2 = calculate_incremental_validity(y_train_fe, X_train, X_train_fe)
                suppressor_df = identify_suppressors(X_train, X_train_fe, y_train_fe, corr_matrix_pd, delta_r2)
                delta_r2_normalized = delta_r2 / number_of_features
                count_suppressor_candidates = len(suppressor_df)
                try:
                    X_train_fe_non_suppressors = X_train_fe.drop(columns=suppressor_df['feature'])
                except KeyError:
                    X_train_fe_non_suppressors = X_train_fe
                corr_matrix_pd_non_suppressor, r2_base_non_suppressor, r2_full_non_suppressor, delta_r2_non_suppressor = calculate_incremental_validity(y_train_fe, X_train, X_train_fe_non_suppressors)
                delta_r2_non_suppressor_normalized = delta_r2_non_suppressor / number_of_features
            else:
                delta_r2 = 0
                delta_r2_normalized = 0
                delta_r2_non_suppressor = 0
                delta_r2_non_suppressor_normalized = 0
                count_suppressor_candidates = 0
            print(number_of_features)
            print(delta_r2)
            print(delta_r2_normalized)
            print(delta_r2_non_suppressor)
            print(delta_r2_non_suppressor_normalized)

            # Suppressor & normalized
            new_row = pd.DataFrame(columns=columns)
            new_row.loc[len(new_row)] = [dataset, method, delta_r2, count_suppressor_candidates, number_of_features, delta_r2_normalized]
            delta_r2_df = pd.concat([delta_r2_df, new_row])

            # No suppressor & normalized
            new_row_non_suppressor = pd.DataFrame(columns=columns)
            new_row_non_suppressor.loc[len(new_row_non_suppressor)] = [dataset, method, delta_r2_non_suppressor, count_suppressor_candidates, number_of_features, delta_r2_non_suppressor_normalized]
            delta_r2_non_suppressor_df = pd.concat([delta_r2_non_suppressor_df, new_row_non_suppressor])

            delta_r2_df.dropna(inplace=True)
            delta_r2_df.to_parquet("delta_r2.parquet")
            delta_r2_non_suppressor_df.dropna(inplace=True)
            delta_r2_non_suppressor_df.to_parquet("delta_r2_non_suppressor.parquet")

    df_pivot = delta_r2_df.pivot(index="Dataset", columns="Method", values="DeltaR2")
    random_cols = [col for col in df_pivot.columns if col.startswith("Random_fold_")]
    df_pivot["Best_Random"] = df_pivot[random_cols].max(axis=1)
    df_pivot = df_pivot.drop(random_cols, axis=1)
    df_pivot_normalized = delta_r2_df.pivot(index="Dataset", columns="Method", values="DeltaR2_Normalized")
    df_pivot_normalized["Best_Random"] = df_pivot_normalized[random_cols].max(axis=1)
    df_pivot_normalized = df_pivot_normalized.drop(random_cols, axis=1)

    df_pivot = df_pivot.sort_index()
    df_pivot_normalized = df_pivot_normalized.sort_index()

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

    plot_delta_r2_graph(dataset_list_wrapped, df_pivot, "")
    plot_delta_r2_graph(dataset_list_wrapped, df_pivot_normalized, "Normalized")

    random_cols = [col for col in df_pivot.columns if col.startswith("MFE_")]
    df_pivot["Best_MFE"] = df_pivot[random_cols].max(axis=1)
    df_pivot = df_pivot.drop(random_cols, axis=1)
    random_cols = [col for col in df_pivot.columns if col.startswith("pandas_")]
    df_pivot["Best_Pandas"] = df_pivot[random_cols].max(axis=1)
    df_pivot = df_pivot.drop(random_cols, axis=1)
    random_cols = [col for col in df_pivot.columns if col.startswith("d2v_")]
    df_pivot["Best_d2v"] = df_pivot[random_cols].max(axis=1)
    df_pivot = df_pivot.drop(random_cols, axis=1)
    random_cols = [col for col in df_pivot_normalized.columns if col.startswith("MFE_")]
    df_pivot_normalized["Best_MFE"] = df_pivot_normalized[random_cols].max(axis=1)
    df_pivot_normalized = df_pivot_normalized.drop(random_cols, axis=1)
    random_cols = [col for col in df_pivot_normalized.columns if col.startswith("pandas_")]
    df_pivot_normalized["Best_Pandas"] = df_pivot_normalized[random_cols].max(axis=1)
    df_pivot_normalized = df_pivot_normalized.drop(random_cols, axis=1)
    random_cols = [col for col in df_pivot_normalized.columns if col.startswith("d2v_")]
    df_pivot_normalized["Best_d2v"] = df_pivot_normalized[random_cols].max(axis=1)
    df_pivot_normalized = df_pivot_normalized.drop(random_cols, axis=1)


    plot_delta_r2_graph(dataset_list_wrapped, df_pivot, "best")
    plot_delta_r2_graph(dataset_list_wrapped, df_pivot_normalized, "best Normalized")

    df_pivot_pandas = df_pivot[["pandas_CatBoost_best", "pandas_CatBoost_recursion", "OpenFE", "Best_Random"]]
    df_pivot_normalized_pandas = df_pivot_normalized[["pandas_CatBoost_best", "pandas_CatBoost_recursion", "OpenFE", "Best_Random"]]

    plot_delta_r2_graph(dataset_list_wrapped, df_pivot_pandas, "Pandas and OpenFE")
    plot_delta_r2_graph(dataset_list_wrapped, df_pivot_normalized_pandas, "Normalized Pandas and OpenFE")



if __name__ == '__main__':
    statistical_analysis()
