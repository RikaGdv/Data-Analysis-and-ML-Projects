import pandas as pd
import numpy as np

df = pd.read_csv("../data/heart_failure_dataset.csv")

print("\n" + "="*40)
print("DATA OVERVIEW")
print("="*40)

print("\nShape:", df.shape)

print("\nFirst rows:")
print(df.head().to_string(index=False))

print("\nInfo:")
df.info()

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nMissing values: {missing if not missing.empty else 'No missing values'}")

print("\nStatistics (numeric):")
print(df.describe().to_string())

print("\nStatistics (categorical):")
print(df.describe(include=["object", "string"]).to_string())

print("\nUnique values:")
for col in df.columns:
    print(col, ":", df[col].nunique())

dup_count = df.duplicated().sum()
print(f"\nDuplicate values: {dup_count}")

# ==========================
# Data Cleaning
# ==========================

print("\n" + "="*40)
print("DATA CLEANING")
print("="*40)

# RestingBP and Cholesterol cannot be zero in patients. These values likely represent missing measurements encoded as 0.
print("\nInvalid values BEFORE cleaning:")
print("RestingBP = 0:", (df["RestingBP"] == 0).sum())
print("Cholesterol = 0:", (df["Cholesterol"] == 0).sum())

# Oldpeak negative values are clinically valid and represent ST elevation.
# These will be retained without modification.
print("\nOldpeak < 0:", (df["Oldpeak"] < 0).sum())

df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)

df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())
df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())

df["RestingBP"] = df["RestingBP"].astype(float)
df["Cholesterol"] = df["Cholesterol"].astype(float)

print("\nInvalid values AFTER cleaning:")
print("RestingBP = 0:", (df["RestingBP"] == 0).sum())
print("Cholesterol = 0:", (df["Cholesterol"] == 0).sum())

print(f"\nMissing values AFTER cleaning: {df.isna().sum().sum()}")

print("\nFinal statistics after cleaning:")
print(df.describe().to_string())

# ==========================
# Save Clean Dataset
# ==========================

df.to_csv("../data/heart_failure_dataset_clean.csv", index=False)

print("\nCleaned dataset saved.")