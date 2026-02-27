import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ==========================
# 1) Load Data & Retrain Optimal Model
# ==========================

print("\n" + "="*50)
print("TRAINING MODEL FOR SHAP ANALYSIS")
print("="*50)

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

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        eval_metric="logloss", random_state=42, max_depth=3,
        learning_rate=0.01, n_estimators=200, subsample=1.0, colsample_bytree=1.0
    ))
])

pipeline.fit(X_train, y_train)

# ==========================
# 2) Extract Data & Clean Names for SHAP
# ==========================

print("\nCalculating SHAP values... (This takes a few seconds)")

X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)

raw_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

clean_feature_names = []
for name in raw_feature_names:
    name = name.replace("num__", "").replace("cat__", "").replace("remainder__", "")
    name = name.replace("_", " ").title()
    name = name.replace("Bmi", "BMI")
    clean_feature_names.append(name)

X_test_shap = pd.DataFrame(X_test_transformed, columns=clean_feature_names)

xgb_model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test_shap)

# ==========================
# 3) Plot 1: Global Summary
# ==========================

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title("SHAP Summary: Top Drivers of Stroke Risk", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("../figures/10_shap_summary.png", dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# 4) Plot 2: Local Waterfall (Explaining ONE Patient)
# ==========================

stroke_indices = np.where(y_test == 1)[0]
patient_idx = stroke_indices[0]

plt.figure(figsize=(8, 5))
shap.plots.waterfall(shap_values[patient_idx], show=False)
plt.title(f"Clinical Explanation for Patient #{patient_idx} (Actual Stroke)", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("../figures/11_shap_waterfall.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nSHAP analysis complete! Check your figures folder.")