import pandas as pd

files = ["../Metadata/core/Core_Matrix_Complete.parquet", "../Metadata/d2v/D2V_Matrix_Complete.parquet", "../Metadata/pandas/Pandas_Matrix_Complete.parquet", "../Metadata/mfe/MFE_General_Matrix_Complete.parquet"]

for file in files:
    dataset = pd.read_parquet(file)
    datasets = dataset["dataset - id"].unique()
    print(file)
    print(datasets)
    print(len(datasets))
