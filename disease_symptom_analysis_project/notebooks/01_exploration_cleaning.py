import pandas as pd

df = pd.read_csv("../data/disease_symptom_and_patient_profile_dataset.csv")

print("\n" + "="*40)
print("DATA OVERVIEW")
print("="*40)

print("\nShape:", df.shape)

print("\nFirst rows:")
print(df.head().to_string(index=False))

print("\nInfo:")
df.info()

print("\nMissing values:")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if not missing.empty else "No missing values")

print("\nStatistics (numeric):")
print(df.describe())

print("\nStatistics (categorical):")
print(df.describe(include=["object", "string"]).to_string())

print("\nUnique values:")
for col in df.columns:
    print(col, ":", df[col].nunique())

dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

# Duplicate rows were retained because identical symptom profiles
# may represent different patients rather than data entry errors.





