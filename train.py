import pandas as pd

import numpy as np



# 1. Define operational settings and sensor column names

setting_names = [f'setting_{i}' for i in range(1, 4)]

sensor_names = [f'sensor_{i}' for i in range(1, 22)]

col_names = ['engine_id', 'cycle'] + setting_names + sensor_names



# 2. Load the training dataset

df_train = pd.read_csv(

    "data/train_FD001.txt",

    sep=r"\s+",

    header=None,

    names=col_names

)



print(f"Dataset Shape: {df_train.shape}")



# 3. Calculate Remaining Useful Life (RUL)

max_cycle = df_train.groupby('engine_id')['cycle'].max().reset_index()

max_cycle.columns = ['engine_id', 'max_cycle']



df_train = pd.merge(df_train, max_cycle, on='engine_id')



# Calculate RUL

df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']



# Create Target Column

# 0 = Healthy

# 1 = Maintenance Required

df_train['maintenance_required'] = (

    df_train['RUL'] <= 30

).astype(int)



# Remove helper column

df_train = df_train.drop(columns=['max_cycle'])



# Display first rows

print("\nFirst 5 rows:")

print(

    df_train[

        ['engine_id',

         'cycle',

         'sensor_2',

         'sensor_3',

         'RUL',

         'maintenance_required']

    ].head()

)



# Check target distribution

print("\nTarget Distribution:")

print(df_train['maintenance_required'].value_counts())