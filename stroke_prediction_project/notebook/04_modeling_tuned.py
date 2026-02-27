import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt


# ============================================================
# 1) Load dataset
# ============================================================

df = pd.read_csv("../data/stroke_dataset_clean.csv")

print("\nDataset shape:", df.shape)


# ============================================================
# 2) Features and Target
# ============================================================

X = df.drop("stroke", axis=1)
y = df["stroke"]


# ============================================================
# 3) Define feature groups
# ============================================================

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


# ============================================================
# 4) Preprocessing
# ============================================================

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
# 5) Train/Test split
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
# 6) Create pipeline
# ============================================================

pipeline = Pipeline(steps=[

    ("preprocessor", preprocessor),

    ("smote", SMOTE(random_state=42)),

    ("model", XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))

])


# ============================================================
# 7) Define hyperparameter grid
# ============================================================

param_grid = {

    "model__n_estimators": [100, 200],

    "model__max_depth": [3, 5, 7],

    "model__learning_rate": [0.01, 0.1],

    "model__subsample": [0.8, 1.0],

    "model__colsample_bytree": [0.8, 1.0]

}


# ============================================================
# 8) GridSearchCV
# ============================================================

grid_search = GridSearchCV(

    estimator=pipeline,

    param_grid=param_grid,

    cv=5,

    scoring="roc_auc",

    n_jobs=-1,

    verbose=2

)


print("\nStarting GridSearchCV...")

grid_search.fit(X_train, y_train)


# ============================================================
# 9) Best model
# ============================================================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)


# ============================================================
# 10) Evaluation
# ============================================================

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]


roc = roc_auc_score(y_test, y_prob)

print(f"\nTuned XGBoost ROC-AUC: {roc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# 11) ROC Curve
# ============================================================

display = RocCurveDisplay.from_estimator(
    best_model,
    X_test,
    y_test
)

display.ax_.set_xlabel("False Positive Rate")
display.ax_.set_ylabel("True Positive Rate")

plt.title("ROC Curve - Tuned XGBoost")

plt.tight_layout()
plt.savefig("../figures/08_roc_curve_tuned_xgboost.png", dpi=300)

plt.show()