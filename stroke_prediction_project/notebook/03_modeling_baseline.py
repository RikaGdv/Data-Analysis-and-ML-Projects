import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 1) Loading dataset
# ============================================================

df = pd.read_csv("../data/stroke_dataset_clean.csv")

print("\nDataset shape:", df.shape)

# ==========================
# 2) Features and Preprocessing
# ==========================

X = df.drop("stroke", axis=1)
y = df["stroke"]


numeric_features = [
    "age",
    "avg_glucose_level",
    "bmi"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"
)

# ============================================================
# 3) Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# ============================================================
# 4) Defining models
# ============================================================

models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(
        probability=True, 
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss", 
        random_state=42
    )
}

# ============================================================
# 5) Training loop
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "="*50)
    print(f"Training {name}")
    print("="*50)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{name} ROC-AUC: {roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    results[name] = roc

# ============================================================
# Summary
# ============================================================

print("\n" + "="*50)
print("MODEL COMPARISON (ROC-AUC)")
print("="*50)

for name, score in results.items():
    print(f"{name}: {score:.4f}")