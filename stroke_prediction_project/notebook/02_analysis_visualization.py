import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================
# Setup
# ==========================

plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (8, 5)

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

df = pd.read_csv("../data/stroke_dataset_clean.csv")
os.makedirs("../figures", exist_ok=True)

def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

df["Stroke Status"] = df["stroke"].map({0: "No Stroke", 1: "Stroke"})

color_dict = {"No Stroke": "#2ECC71", "Stroke": "#E74C3C"}
stack_colors = ["#2ECC71", "#E74C3C"]

# ==========================
# 01 Stroke Distribution
# ==========================

section("Stroke Distribution")

plt.figure()

ax = sns.countplot(
    data=df,
    x="Stroke Status",
    hue="Stroke Status",
    palette=color_dict,
    legend=False
)

plt.title("Stroke Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")

for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(
        f"{height}",
        (p.get_x() + p.get_width()/2, height),
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
        xytext=(0, 5),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8)
    )

plt.tight_layout()
plt.savefig("../figures/01_stroke_distribution.png", dpi=300)
plt.show()


# ==========================
# 02 Stroke % by Gender
# ==========================

section("Stroke Rate by Gender")

plt.figure()

ax = sns.barplot(
    data=df,
    x="gender",
    y="stroke",
    errorbar=None,
    estimator=lambda x: np.mean(x) * 100
)

plt.title("Stroke Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Stroke Rate (%)")

for p in ax.patches:
    value = p.get_height()
    ax.annotate(f"{value:.2f}%",
                (p.get_x() + p.get_width()/2, value),
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=11)

plt.tight_layout()
plt.savefig("../figures/02_stroke_rate_by_gender.png", dpi=300)
plt.show()

# ==========================
# 03 Age Distribution by Stroke
# ==========================

section("Age Distribution by Stroke")

plt.figure()

sns.kdeplot(
    data=df,
    x="age",
    hue="Stroke Status",
    fill=True,
    palette=color_dict,
    common_norm=False,
    alpha=0.5,
    linewidth=2
)

plt.title("Age Distribution: Stroke vs. No Stroke")
plt.xlabel("Age")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("../figures/03_age_distribution_kde.png", dpi=300)
plt.show()

# ==========================
# 04 Numeric Vitals (Glucose & BMI)
# ==========================

section("Numeric Vitals: Glucose and BMI")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(
    data=df, x="avg_glucose_level", hue="Stroke Status",
    fill=True, palette=color_dict, common_norm=False,
    alpha=0.5, linewidth=2, ax=axes[0]
)
axes[0].set_title("Glucose Level Distribution")
axes[0].set_xlabel("Average Glucose Level")
axes[0].set_ylabel("Density")

sns.kdeplot(
    data=df, x="bmi", hue="Stroke Status",
    fill=True, palette=color_dict, common_norm=False,
    alpha=0.5, linewidth=2, ax=axes[1]
)
axes[1].set_title("BMI Distribution")
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("../figures/04_vitals_kde.png", dpi=300)
plt.show()

# ==========================
# 05 Clinical Risk Factors
# ==========================
section("Clinical Risk Factors")

df["hypertension_label"] = df["hypertension"].map({0: "No", 1: "Yes"})
df["heart_disease_label"] = df["heart_disease"].map({0: "No", 1: "Yes"})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

yes_no_palette = {"No": "#2ECC71", "Yes": "#E74C3C"}

# 1. Hypertension
ax1 = sns.barplot(
    data=df,
    x="hypertension_label",
    y="stroke",
    hue="hypertension_label",
    order=["No", "Yes"],
    estimator=lambda x: np.mean(x) * 100,
    errorbar=None,
    palette=yes_no_palette,
    legend=False,
    ax=axes[0]
)
axes[0].set_title("Stroke Rate by Hypertension")
axes[0].set_xlabel("Hypertension Diagnosis")
axes[0].set_ylabel("Stroke Rate (%)")

ax2 = sns.barplot(
    data=df,
    x="heart_disease_label",
    y="stroke",
    hue="heart_disease_label",
    order=["No", "Yes"],
    estimator=lambda x: np.mean(x) * 100,
    errorbar=None,
    palette=yes_no_palette,
    legend=False,
    ax=axes[1]
)
axes[1].set_title("Stroke Rate by Heart Disease")
axes[1].set_xlabel("Heart Disease Diagnosis")
axes[1].set_ylabel("")

smoking_colors = ["#3498DB", "#F39C12", "#E74C3C", "#bdc3c7"]

ax3 = sns.barplot(
    data=df,
    x="smoking_status",
    y="stroke",
    hue="smoking_status",
    estimator=lambda x: np.mean(x) * 100,
    errorbar=None,
    palette=smoking_colors,
    legend=False,
    order=["never smoked", "formerly smoked", "smokes", "Unknown"],
    ax=axes[2]
)
axes[2].set_title("Stroke Rate by Smoking")
axes[2].tick_params(axis='x', rotation=15)
axes[2].set_xlabel("Smoking History")
axes[2].set_ylabel("")

for ax in [ax1, ax2, ax3]:
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=3,
            fontweight="bold"
        )

plt.tight_layout()
plt.savefig("../figures/05_clinical_risk_factors.png", dpi=300)
plt.show()

# ==========================
# 06 Correlation Heatmap
# ==========================

section("Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

rename_dict = {
    "age": "Age",
    "avg_glucose_level": "Glucose Level",
    "bmi": "BMI",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "stroke": "Stroke"
}

numeric_df = numeric_df.rename(columns=rename_dict)

corr = numeric_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("../figures/06_correlation_heatmap.png", dpi=300)
plt.show()

# ==========================
# 07 Feature Correlation with Stroke
# ==========================

section("Feature Correlation with Stroke")

corr_with_stroke = numeric_df.corr()["Stroke"].drop("Stroke").sort_values(ascending=False)

plt.figure(figsize=(8, 4))

ax = sns.barplot(
    x=corr_with_stroke.values,
    y=corr_with_stroke.index,
    palette="vlag_r",
    hue=corr_with_stroke.index,
    legend=False
)

plt.title("Correlation of Clinical Features with Stroke Risk", fontsize=14, fontweight="bold")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("")

for container in ax.containers:
    ax.bar_label(container, fmt="%+.3f", padding=5, fontweight="bold", fontsize=11)

plt.axvline(x=0, color='grey', linewidth=1, linestyle='--')

plt.xlim(corr_with_stroke.min() - 0.05, corr_with_stroke.max() + 0.08)

plt.tight_layout()
plt.savefig("../figures/07_feature_correlation_with_stroke.png", dpi=300)
plt.show()