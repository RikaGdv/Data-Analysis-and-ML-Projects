import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

# ==========================
# 1) Load data
# ==========================

df = pd.read_csv("../data/disease_symptom_and_patient_profile_dataset.csv")

target_col = "Outcome Variable"

print("\n========================================")
print("MODEL BASELINE: Predicting Outcome Variable")
print("========================================")

print("\nDataset shape:", df.shape)
counts = df[target_col].value_counts()
print("\nTarget Distribution:")
for label, count in counts.items():
    pct = (count / len(df)) * 100
    print(f"{label:<15}: {count} ({pct:.1f}%)")

# Disease has very high cardinality,
# so we exclude it to avoid overfitting and excessive encoding.

drop_cols = ["Disease"]
X = df.drop(columns=[target_col] + drop_cols)

y = df[target_col]

print("\nFeatures used:", list(X.columns))
print("\nData types:")
print(X.dtypes)

# ==========================
# 3) Train/Test split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain set size:", X_train.shape)
print("\nTest set size:", X_test.shape)

# ==========================
# 4) Preprocessing
# ==========================

health_categories = ["Low", "Normal", "High"]

numeric_features = ["Age"]
ordinal_features = ["Blood Pressure", "Cholesterol Level"]
binary_features = ["Fever", "Cough", "Fatigue", "Difficulty Breathing", "Gender"]

print("\nNumeric features:", numeric_features)
print("\nOrdinal features:", ordinal_features)
print("\nBinary features:", binary_features)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("ord", OrdinalEncoder(categories=[health_categories, health_categories]), ordinal_features),
        ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), binary_features),
    ]
)

# ==========================
# 5) Model 1: Logistic Regression
# ==========================

log_reg_model = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("model", LogisticRegression(max_iter=2000)),
    ]
)

print("\n========================================")
print("Training Logistic Regression...")
print("========================================")

log_reg_model.fit(X_train, y_train)

y_pred_lr = log_reg_model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"\nClassification Report (Logistic Regression):\n{classification_report(y_test, y_pred_lr)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")

# ==========================
# 6) Model 2: Random Forest
# ==========================

rf_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

print("\n========================================")
print("Training Random Forest...")
print("========================================")

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print(f"\nRandom Forest accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"\nClassification report (Random Forest):\n{classification_report(y_test, y_pred_rf)}")
print(f"\nConfusion matrix (Random Forest):\n{confusion_matrix(y_test, y_pred_rf)}")

importances = rf_model.named_steps["model"].feature_importances_

feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

importance_df["feature"] = (
    importance_df["feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

print("\nTop 15 most important features:")
print(importance_df.head(15).to_string(index=False))

# ==========================
# 7) Model Comparison & Visualization
# ==========================

print("\n" + "="*40)
print("FINAL COMPARISON")
print("="*40)

results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)]
})
print(results.to_string(index=False))

# --- Feature Importance (Random Forest) ---
feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()
importances = rf_model.named_steps["model"].feature_importances_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
feat_imp.index = (
    feat_imp.index
    .str.replace("num__", "", regex=False)
    .str.replace("ord__", "", regex=False)
    .str.replace("bin__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=feat_imp.values,
    y=feat_imp.index,
    hue=feat_imp.index,
    palette="magma",
    legend=False
)
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../figures/07_feature_importance.png")
plt.show()

# ==========================
# 8) Final Interpretation
# ==========================

print("\n" + "=" * 40)
print("INSIGHTS & CONCLUSIONS")
print("=" * 40)
print(f"""
- Model Performance: Random Forest ({results.iloc[1]['Accuracy']:.2%}) significantly 
  outperforms Logistic Regression ({results.iloc[0]['Accuracy']:.2%}). 
  This suggests that the relationship between symptoms and patient outcomes is 
  complex and non-linear rather than a simple additive effect.

- Key Drivers: Age is the most dominant predictor, followed by systemic health 
  indicators like Cholesterol and Blood Pressure.

- Feature Logic: The use of Ordinal Encoding ensured the model respected the 
  biological gradient (Low < Normal < High) of health metrics.

- Data Context: The small sample size (N=349) and 
  intentional duplicates mean these results should be viewed as a 
  methodological baseline rather than clinical fact.
""")