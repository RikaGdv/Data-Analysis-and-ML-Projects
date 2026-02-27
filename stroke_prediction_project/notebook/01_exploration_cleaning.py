import pandas as pd

df = pd.read_csv("../data/stroke_dataset.csv")

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
print(f"\nMissing values: {missing if not missing.empty else 'No missing values.'}")

print("\nStatistics (numeric):")
print(df.describe().to_string())

print("\nStatistics (categorical):")
print(df.describe(include=["object", "string"]).to_string())

print("\nUnique values:")
for col in df.columns:
    print(col, ":", df[col].nunique())

print("\nGender exploration:")
print(df['gender'].value_counts())

df = df.drop(columns=["id"])
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

# ==========================
# Data Cleaning
# ==========================

print("\n" + "="*40)
print("DATA CLEANING")
print("="*40)

if dup_count > 0:
    df = df.loc[~df.duplicated()].copy()
    print(f"Removed {dup_count} duplicate rows.")
else:
    print("No duplicate clinical records found.")

# Remove rare 'Other' gender category (only 1 record).
# A single sample is insufficient for the model to learn meaningful patterns,
# so it is removed to improve model reliability.

other_count = (df["gender"] == "Other").sum()

if other_count > 0:
    df = df[df["gender"] != "Other"]
    print(f"\nRemoved {other_count} record with gender='Other'.")
else:
    print("\nNo 'Other' gender records found.")

# Impute missing BMI values with the median
missing_bmi = df["bmi"].isna().sum()

if missing_bmi > 0:
    median_bmi = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(median_bmi)
    print(f"\nFilled {missing_bmi} missing BMI values with median ({median_bmi:.2f}).")
else:
    print("\nNo missing BMI values found.")

print("\nShape after cleaning:", df.shape)

# ==========================
# Save Data
# ==========================

df.to_csv("../data/stroke_dataset_clean.csv", index=False)

print("\nCleaned dataset saved.")