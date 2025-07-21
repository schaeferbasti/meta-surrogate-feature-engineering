import pandas as pd
from matplotlib import pyplot as plt


def plot_count(df, name):
    # Plot
    plt.figure(figsize=(12, 7))
    df.plot(kind='bar', width=0.3, color='darkblue', label='Count of datasets')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Number of datasets")
    plt.title("Count of datasets with metafeatures per method")
    plt.xticks(rotation=90, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Count_datasets_per_method_" + name + "bar.png")
    plt.show()


def main():
    files = ["../../Metadata/core/Core_Matrix_Complete.parquet", "../../Metadata/d2v/D2V_Matrix_Complete.parquet", "../../Metadata/pandas/Pandas_Matrix_Complete.parquet", "../../Metadata/mfe/MFE_General_Matrix_Complete.parquet"]
    df = pd.DataFrame(columns=["Method", "Count"])
    for file in files:
        dataset = pd.read_parquet(file)
        datasets = dataset["dataset - id"].unique()
        print(file)
        print(datasets)
        print(len(datasets))
        method = file.split("/")[-1].split("_Matrix")[0]
        if method == "Core":
            method = "Total"
        elif method == "MFE_General":
            method = "MFE"
        new_row = pd.DataFrame([{"Method": method, "Count": len(datasets)}])
        df = pd.concat([df, new_row], ignore_index=True)
    new_row = pd.DataFrame([{"Method": "OpenFE", "Count": 36}])
    df_openfe = pd.concat([df, new_row], ignore_index=True)
    df.set_index("Method", inplace=True)
    df_openfe.set_index("Method", inplace=True)
    df_openfe = df_openfe.drop(df_openfe.index[[1, 3]])
    plot_count(df, "")
    plot_count(df_openfe, "openfe_")


if __name__ == "__main__":
    main()
