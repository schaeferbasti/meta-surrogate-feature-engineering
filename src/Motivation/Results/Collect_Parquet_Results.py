from glob import glob
import pandas as pd

files = glob('results_*.parquet')
columns=['Dataset', 'LGBM', 'OpenFE + LGBM', 'Autogluon LGBM', 'OpenFE + Autogluon LGBM', 'Tuned Autogluon LGBM', 'OpenFE + Tuned Autogluon LGBM']
df = pd.DataFrame(columns=columns)
for file in files:
    df_form_file = pd.read_parquet(file, engine='pyarrow', columns=columns)
    df = pd.concat([df, df_form_file], axis=0, ignore_index=True)
print(df)
df.to_parquet('results.parquet')
