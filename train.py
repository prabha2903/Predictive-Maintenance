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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features
features = setting_names + sensor_names

X = df_train[features]
y = df_train['maintenance_required']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))