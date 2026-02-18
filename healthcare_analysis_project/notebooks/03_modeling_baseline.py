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

# ==========================
# 1) Load cleaned data
# ==========================

df = pd.read_csv("../data/healthcare_dataset_clean.csv")

# Choosing which data to predict:
target_col = "Test Results"

print("\n========================================")
print("MODEL BASELINE: Predicting Test Results")
print("========================================")

print("\nDataset shape:", df.shape)
print("\nTarget Distribution:")
for label, count in df[target_col].value_counts().items():
    print(f"{label}: {count}")

# ==========================
# 2) Choosing features and target
# ==========================

drop_cols = ["Name", "Doctor", "Hospital", "Date of Admission", "Discharge Date", "Room Number"]

X = df.drop(columns=[target_col] + drop_cols)
y = df[target_col]

print("\nFeatures used:", list(X.columns))


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
print("\nTest set size :", X_test.shape)


# ==========================
# 4) Preprocessing
# ==========================

numeric_features = ["Age", "Billing Amount", "Length of Stay"]

categorical_features = [c for c in X.columns if c not in numeric_features]

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ==========================
# 5) Model 1: Logistic Regression
# ==========================

log_reg_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000))
])

print("\n========================================")
print("Training Logistic Regression...")
print("========================================")

log_reg_model.fit(X_train, y_train)

y_pred_lr = log_reg_model.predict(X_test)

print("\nLogistic Regression accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))
print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))
print("Confusion matrix (Logistic Regression):")
print(confusion_matrix(y_test, y_pred_lr))


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

print("\nRandom Forest accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
print("\nClassification report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print("Confusion matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))


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

ohe_feature_names = rf_model.named_steps['preprocess'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_feature_names)

importances = rf_model.named_steps['model'].feature_importances_
feat_imp = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    x=feat_imp.values,
    y=feat_imp.index,
    hue=feat_imp.index,
    palette="viridis",
    legend=False
)
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../figures/08_feature_importance.png")
plt.show()

# ==========================
# 7) Final Interpretation
# ==========================
print("\n" + "=" * 40)
print("INSIGHTS & CONCLUSIONS")
print("=" * 40)
print(f"""
- Model Performance: Random Forest ({results.iloc[1]['Accuracy']:.2%}) significantly 
  outperforms Logistic Regression ({results.iloc[0]['Accuracy']:.2%}). 

- Predictive Signal: While accuracy is low, the Random Forest is finding patterns 
  ~10% better than random guessing (33.3%), primarily in continuous variables.

- Key Drivers: According to the Feature Importance plot, '{feat_imp.index[0]}' 
  and '{feat_imp.index[1]}' are the strongest predictors.

- Synthetic Data Context: As seen in the visualizations (Figures 01-07), features 
  like Medical Condition and Age are distributed almost uniformly across all outcomes.

- Note: The low predictive power is likely a limitation of the synthetic 
  dataset's randomness rather than the model architecture itself.
""")