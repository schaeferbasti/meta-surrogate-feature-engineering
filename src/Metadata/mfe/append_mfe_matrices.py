import os
import pandas as pd
from src.utils.get_matrix import get_additional_mfe_columns_group


def main():
    # Define column groups
    general_columns = get_additional_mfe_columns_group("general")
    statistical_columns = get_additional_mfe_columns_group("statistical")
    info_theory_columns = get_additional_mfe_columns_group("info-theory")

    merge_keys = ["dataset - id", "feature - name", "operator", "model", "improvement"]

    # Accumulate data for each category
    general_dfs = []
    statistical_dfs = []
    info_theory_dfs = []

    log_files = sorted(os.listdir())

    for log_file in log_files:
        if log_file.endswith('.parquet'):
            category = log_file.split("_")[1]
            matrix = pd.read_parquet(log_file)

            if category == "general" and len(matrix.columns) == len(general_columns) + 5:
                general_dfs.append(matrix)
            elif category == "statistical" and len(matrix.columns) == len(statistical_columns) + 5:
                statistical_dfs.append(matrix)
            elif category == "info-theory" and len(matrix.columns) == len(info_theory_columns) + 5:
                info_theory_dfs.append(matrix)

    # Concatenate by category
    result_matrix_only_general = pd.concat(general_dfs, ignore_index=True)
    result_matrix_only_statistical = pd.concat(statistical_dfs, ignore_index=True)
    result_matrix_only_info_theory = pd.concat(info_theory_dfs, ignore_index=True)

    # Merge full combinations only once
    def safe_merge(left, right):
        return pd.merge(left, right, on=merge_keys, how="inner")

    result_matrix_all = safe_merge(
        safe_merge(result_matrix_only_general, result_matrix_only_statistical),
        result_matrix_only_info_theory
    )

    result_matrix_without_general = safe_merge(result_matrix_only_statistical, result_matrix_only_info_theory)
    result_matrix_without_statistical = safe_merge(result_matrix_only_general, result_matrix_only_info_theory)
    result_matrix_without_info_theory = safe_merge(result_matrix_only_general, result_matrix_only_statistical)

    # Save outputs
    result_matrix_only_general.to_parquet("MFE_General_Matrix_Complete.parquet")
    result_matrix_only_statistical.to_parquet("MFE_Statistical_Matrix_Complete.parquet")
    result_matrix_only_info_theory.to_parquet("MFE_Info_Theory_Matrix_Complete.parquet")
    result_matrix_without_general.to_parquet("MFE_Without_General_Matrix_Complete.parquet")
    result_matrix_without_statistical.to_parquet("MFE_Without_Statistical_Matrix_Complete.parquet")
    result_matrix_without_info_theory.to_parquet("MFE_Without_Info_Theory_Matrix_Complete.parquet")
    result_matrix_all.to_parquet("MFE_All_Matrix_Complete.parquet")


if __name__ == "__main__":
    main()