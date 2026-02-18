import pandas as pd

df = pd.read_csv("../data/healthcare_dataset.csv")

# ==========================
# Data Exploration
# ==========================

print("\n" + "="*40)
print("DATA OVERVIEW")
print("="*40)

print("\nShape:", df.shape)
# The dataset contains 55500 rows and 15 columns.

print("\nFirst rows:")
print(df.head().to_string(index=False))
# There is inconsistent capitalization in names.

print("\nInfo:")
df.info()
# The Date columns are strings, they should be converted to datetime for time-based analysis.

print("\nMissing values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing if not missing.empty else "No missing values.")
# There are no missing values.

print("\nStatistics (numeric):")
print(df.describe())
# Billing amount has negative values, we need to investigate this.

print("\nStatistics (categorical):")
print(df.describe(include=["object", "string"]).to_string())

print("\nUnique values:")
for col in df.columns:
    print(col, ":", df[col].nunique())

dup_count = df.duplicated().sum()
print("\nDuplicate rows:", dup_count)

# ==========================
# Data Cleaning
# ==========================

# Fix messy name formatting
df["Name"] = df["Name"].str.title()

# Date Conversion
date_cols = ["Date of Admission", "Discharge Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Check for issues
bad_dates = df[date_cols].isna().sum()
if bad_dates.sum() > 0:
    print(f"\nWarning: Found invalid date formats:\n{bad_dates[bad_dates > 0]}")
else:
    print("\nDate Validation: All dates parsed successfully.")

# Creation of Length of Stay
df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
neg_los = (df["Length of Stay"] < 0).sum()
df = df[df["Length of Stay"] >= 0]
print(f"\nRemoved {neg_los} rows where Discharge was before Admission.")

# Handle Negative Billing
neg_bill_count = (df["Billing Amount"] < 0).sum()
df = df[df["Billing Amount"] >= 0]
print(f"\nRemoved {neg_bill_count} rows with negative billing.")

# Exact duplicate rows removed to avoid bias in analysis
df = df.drop_duplicates()

print(f"\nFinal Shape after cleaning: {df.shape}")

# ==========================
# Save Clean Dataset
# ==========================

df.to_csv("../data/healthcare_dataset_clean.csv", index=False)
print("\nClean dataset saved.")


