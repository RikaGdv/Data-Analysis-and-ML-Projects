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

df = pd.read_csv("../data/heart_failure_dataset_clean.csv")
os.makedirs("../figures", exist_ok=True)

def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

df["HeartDiseaseLabel"] = df["HeartDisease"].map({0: "No Heart Disease", 1: "Heart Disease"})
color_dict = {"No Heart Disease": "#2ECC71", "Heart Disease": "#E74C3C"}
stack_colors = ["#2ECC71", "#E74C3C"]

# ==========================
# 01 Heart Disease Distribution
# ==========================

section("Heart Disease Distribution")

plt.figure()
ax = sns.countplot(data=df, x="HeartDiseaseLabel", hue="HeartDiseaseLabel",
                   palette=color_dict, legend=False)
plt.title("Heart Disease Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f"{height}", (p.get_x() + p.get_width()/2, height),
                ha="center", va="bottom", fontweight="bold", fontsize=12,
                xytext=(0, 5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
plt.tight_layout()
plt.savefig("../figures/01_heart_disease_distribution.png", dpi=300)
plt.show()

# ==========================
# 02 Heart Disease % by Age Group
# ==========================

section("Heart Disease Percentage by Age Group")

df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 40, 50, 60, 70, 100],
                        labels=["<40", "40-50", "50-60", "60-70", "70+"])
age_pct = pd.crosstab(df["AgeGroup"], df["HeartDiseaseLabel"], normalize="index") * 100
age_pct = age_pct[["No Heart Disease", "Heart Disease"]]

fig, ax = plt.subplots()
age_pct.plot(kind="bar", stacked=True, color=stack_colors, ax=ax)
plt.title("Heart Disease Percentage by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers:
    labels = [f"{v.get_height():.1f}%" if v.get_height() > 0 else "" for v in container]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("../figures/02_heart_disease_by_age_group.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 03 Heart Disease % by Sex
# ==========================

section("Heart Disease Percentage by Sex")

sex_pct = pd.crosstab(df["Sex"], df["HeartDiseaseLabel"], normalize="index") * 100
sex_pct = sex_pct[["No Heart Disease", "Heart Disease"]]

fig, ax = plt.subplots()
sex_pct.plot(kind="bar", stacked=True, color=stack_colors, ax=ax)
plt.title("Heart Disease Percentage by Sex")
plt.xlabel("Sex")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers:
    labels = [f"{v.get_height():.1f}%" for v in container]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("../figures/03_heart_disease_by_sex.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 04 Chest Pain Type - Outcome Percentage
# ==========================

section("Heart Disease Percentage by Chest Pain Type")

cp_pct = pd.crosstab(df["ChestPainType"], df["HeartDiseaseLabel"], normalize="index") * 100
cp_pct = cp_pct[["No Heart Disease", "Heart Disease"]]

fig, ax = plt.subplots()
cp_pct.plot(kind="bar", stacked=True, color=stack_colors, ax=ax)
plt.title("Heart Disease Risk by Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')
for container in ax.containers:
    labels = [f"{v.get_height():.1f}%" if v.get_height() > 0 else "" for v in container]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig("../figures/04_chest_pain_stacked.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 05 Cholesterol Distribution
# ==========================

section("Cholesterol Distribution by Heart Disease")

plt.figure()
sns.kdeplot(data=df, x="Cholesterol", hue="HeartDiseaseLabel", fill=True,
            palette=color_dict, alpha=0.5, linewidth=2, common_norm=False)
# Add median lines for each group
for label, color in color_dict.items():
    median_val = df.loc[df["HeartDiseaseLabel"] == label, "Cholesterol"].median()
    plt.axvline(median_val, color=color, linestyle="--", linewidth=1.5,
                label=f"{label} Median")
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol (mg/dL)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/05_cholesterol_distribution.png", dpi=300)
plt.show()

# ==========================
# 06 Clinical Risk Profile
# ==========================

section("Clinical Risk Profile: High-Risk Markers")

risk_markers = [
    ("Exercise Angina (Yes)", df["ExerciseAngina"] == "Y"),
    ("Asymptomatic Chest Pain (ASY)", df["ChestPainType"] == "ASY"),
    ("ST Slope (Flat)", df["ST_Slope"] == "Flat"),
    ("Fasting Blood Sugar > 120", df["FastingBS"] == 1)
]

risk_data = []
for name, condition in risk_markers:
    positive_rate = df.loc[condition, "HeartDisease"].mean() * 100
    patient_count = condition.sum()
    risk_data.append({"Risk Factor": name, "Heart Disease Rate (%)": positive_rate, "Patient Count": patient_count})

risk_df = pd.DataFrame(risk_data).sort_values("Heart Disease Rate (%)", ascending=False)

plt.figure(figsize=(10, 5))
ax = sns.barplot(data=risk_df, x="Heart Disease Rate (%)", y="Risk Factor",
                 palette="Reds_r", hue="Risk Factor", legend=False)
plt.title("Clinical Risk Factors vs. Heart Disease Rate", pad=15)
plt.xlabel("Percentage Diagnosed with Heart Disease (%)")
plt.ylabel("Clinical Marker")
plt.xlim(0, 100)
for i, p in enumerate(ax.patches):
    width = p.get_width()
    count = risk_df.iloc[i]["Patient Count"]
    ax.text(width - 2, p.get_y() + p.get_height() / 2, f"{width:.1f}%", va="center", ha="right",
            color="white", fontweight="bold", fontsize=11)
    ax.text(width + 2, p.get_y() + p.get_height() / 2, f"(n={count})", va="center", ha="left",
            color="black", fontstyle="italic", fontsize=10)
plt.tight_layout()
plt.savefig("../figures/06_clinical_risk_profile.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 07 Numeric Features Grid
# ==========================

section("Numeric Features Distributions")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
numeric_cols_to_plot = ["Age", "RestingBP", "MaxHR", "Oldpeak"]
titles = ["Age Distribution", "Resting Blood Pressure", "Maximum Heart Rate", "Oldpeak (ST Depression)"]

for i, col in enumerate(numeric_cols_to_plot):
    sns.violinplot(data=df, x="HeartDiseaseLabel", y=col, hue="HeartDiseaseLabel", palette=color_dict, legend=False,
                   ax=axes[i])
    axes[i].set_title(titles[i], fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel(col)

plt.suptitle("Clinical Metrics by Diagnosis", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("../figures/07_numeric_violin_grid.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 08 Exercise Angina vs MaxHR & Age
# ==========================

section("Exercise Angina Impact on MaxHR and Age")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(
    data=df,
    x="ExerciseAngina",
    y="MaxHR",
    hue="ExerciseAngina",
    palette=["#3498DB","#E74C3C"],
    legend=False,
    ax=axes[0]
)
axes[0].set_title("MaxHR by Exercise Angina")
axes[0].set_xlabel("Exercise Angina")
axes[0].set_ylabel("MaxHR")

sns.boxplot(
    data=df,
    x="ExerciseAngina",
    y="Age",
    hue="ExerciseAngina",
    palette=["#3498DB","#E74C3C"],
    legend=False,
    ax=axes[1]
)
axes[1].set_title("Age by Exercise Angina")
axes[1].set_xlabel("Exercise Angina")
axes[1].set_ylabel("Age")

plt.tight_layout()
plt.savefig("../figures/08_exercise_angina_numeric.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 09 Fasting Blood Sugar vs Heart Disease
# ==========================

section("Fasting Blood Sugar vs Heart Disease")

fbs_pct = pd.crosstab(df["FastingBS"], df["HeartDiseaseLabel"], normalize="index") * 100
fbs_pct = fbs_pct[["No Heart Disease", "Heart Disease"]]

fig, ax = plt.subplots()
fbs_pct.plot(kind="bar", stacked=True, color=stack_colors, ax=ax)

plt.title("Heart Disease Risk by Fasting Blood Sugar")
plt.xlabel("Fasting Blood Sugar (>120 mg/dL)")
ax.set_xticklabels(["Normal (<=120)", "High (>120)"], rotation=0)
plt.ylabel("Percentage (%)")
plt.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc='upper left')

for container in ax.containers:
    labels = [f"{v.get_height():.1f}%" if v.get_height() > 0 else "" for v in container]
    ax.bar_label(container, labels=labels, label_type="center", color="white", fontweight="bold")

plt.tight_layout()
plt.savefig("../figures/09_fastingbs_stacked.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 10 Feature Correlation Heatmap
# ==========================

section("Numeric Feature Correlation Heatmap")

numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(9, 7))
sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
            vmin=-1, vmax=1, linewidths=0.5)
plt.title("Numeric Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../figures/10_feature_correlation.png", dpi=300)
plt.show()


