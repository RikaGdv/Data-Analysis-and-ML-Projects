import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ==========================
# 1) Load Data & Split
# ==========================

df = pd.read_csv("../data/stroke_dataset_clean.csv")

X = df.drop("stroke", axis=1)
y = df["stroke"]

numeric_features = ["age", "avg_glucose_level", "bmi"]
categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# ==========================
# 2) Train with Best Parameters
# ==========================

print("\nTraining Optimized Model...")
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        max_depth=3,
        learning_rate=0.01,
        n_estimators=200,
        subsample=1.0,
        colsample_bytree=1.0
    ))
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]

# ==========================
# 3) Threshold Exploration
# ==========================

print("\n" + "=" * 50)
print("TESTING CLINICAL THRESHOLDS")
print("=" * 50)

thresholds_to_test = np.arange(0.1, 0.9, 0.05)
metrics = []

for t in thresholds_to_test:
    y_pred_thresh = (y_prob >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics.append([t, precision, recall, f1, tp, fn, fp, tn])

metrics_df = pd.DataFrame(
    metrics,
    columns=["threshold", "precision", "recall", "f1", "TP", "FN", "FP", "TN"]
)

print(metrics_df[["threshold", "recall", "precision", "f1", "TP", "FP"]].round(3).to_string(index=False))

# ==========================
# 4) The "Clinical Rule" Selection
# ==========================

# We want the highest threshold that still catches >= 85% of strokes.
# If none catch 85%, we just take the one with the highest F1 score as a fallback.
high_recall_df = metrics_df[metrics_df["recall"] >= 0.85]

if not high_recall_df.empty:
    optimal_row = high_recall_df.iloc[-1]
else:
    optimal_row = metrics_df.loc[metrics_df["f1"].idxmax()]

optimal_threshold = optimal_row["threshold"]

print("\n" + "=" * 50)
print(f"OPTIMAL CLINICAL THRESHOLD SELECTED: {optimal_threshold:.2f}")
print("=" * 50)
print(f"Strokes Caught (TP): {int(optimal_row['TP'])} out of 50")
print(f"Missed Strokes (FN): {int(optimal_row['FN'])}")
print(f"False Alarms (FP):   {int(optimal_row['FP'])}")

# ==========================
# 5) Plot Recall and F1 vs Threshold
# ==========================

plt.figure(figsize=(8, 5))
plt.plot(metrics_df["threshold"], metrics_df["recall"], marker='o', label="Recall (Catching Strokes)", color="#E74C3C")
plt.plot(metrics_df["threshold"], metrics_df["f1"], marker='o', label="F1 Score (Balance)", color="#3498DB")
plt.axvline(optimal_threshold, color="gray", linestyle="--", label=f"Chosen Threshold = {optimal_threshold:.2f}")
plt.xlabel("Decision Threshold (Probability %)")
plt.ylabel("Score")
plt.title("Medical Trade-off: Recall vs F1 Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/09_threshold_vs_recall_f1.png", dpi=300)
plt.show()