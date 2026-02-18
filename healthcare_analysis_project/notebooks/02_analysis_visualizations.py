import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.rcParams["figure.dpi"] = 80
sns.set_style("whitegrid")
sns.set_context("talk")

df = pd.read_csv("../data/healthcare_dataset_clean.csv")
os.makedirs("../figures", exist_ok=True)

# --------------------------
# Derived features for analysis
# --------------------------

age_order = ["Child", "Young Adult", "Adult", "Middle Age", "Senior"]
billing_order = ["Low", "Medium", "High", "Very High"]

df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=age_order
)

df["Billing Category"] = pd.qcut(
    df["Billing Amount"],
    q=4,
    labels=billing_order
)

# --------------------------
# Helper functions
# --------------------------

def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

def add_percent_labels(ax, min_pct=5):
    for p in ax.patches:
        height = p.get_height()
        if height >= min_pct:
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + height / 2
            ax.text(
                x, y, f"{height:.1f}%",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white"
            )

def stacked_percent_plot(df, x_col, hue_col, title, xlabel, ylabel, save_path, order=None, hue_order=None):
    tab = pd.crosstab(df[x_col], df[hue_col], normalize="index") * 100

    if order is not None:
        tab = tab.loc[order]
    if hue_order is not None:
        tab = tab[hue_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    tab.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)

    # labels inside bars
    add_percent_labels(ax, min_pct=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# --------------------------
# Visualizations
# --------------------------

# ==========================
# 01) Billing Amount by Medical Condition
# ==========================

section("Billing Amount by Medical Condition")
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Medical Condition",
    y="Billing Amount",
    palette="Set2",
    hue="Medical Condition",
    legend=False
)
plt.title("Billing Amount by Medical Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/01_billing_by_condition.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 02) Age Distribution by Medical Condition
# ==========================

section("Age Distribution by Medical Condition")
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Medical Condition",
    y="Age"
)
plt.title("Age Distribution by Medical Condition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/02_age_by_condition.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 03) Length of Stay by Admission Type
# ==========================

section("Length of Stay by Admission Type")
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Admission Type",
    y="Length of Stay"
)
plt.title("Length of Stay by Admission Type")
plt.tight_layout()
plt.savefig("../figures/03_los_by_admission_type.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 04) Test Results by Admission Type (%)
# ==========================

section("Test Results by Admission Type (%)")
stacked_percent_plot(
    df=df,
    x_col="Admission Type",
    hue_col="Test Results",
    title="Test Results Distribution by Admission Type (%)",
    xlabel="Admission Type",
    ylabel="Percentage (%)",
    save_path="../figures/04_test_results_by_admission_type.png",
    hue_order=["Normal", "Inconclusive", "Abnormal"]  # optional, keeps legend consistent
)

# ==========================
# 05) Billing Amount by Insurance Provider
# ==========================

section("Billing Amount by Insurance Provider")
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="Insurance Provider",
    y="Billing Amount"
)
plt.title("Billing Amount by Insurance Provider")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/05_billing_by_insurance.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 06) Test Results by Age Group
# ==========================

section("Test Results by Age Group (%)")
stacked_percent_plot(
    df=df,
    x_col="Age Group",
    hue_col="Test Results",
    title="Test Results by Age Group (%)",
    xlabel="Age Group",
    ylabel="Percentage (%)",
    save_path="../figures/06_test_results_by_age_group.png",
    order=age_order,
    hue_order=["Normal", "Inconclusive", "Abnormal"]
)

# ==========================
# 07) Admission Type by Billing Category
# ==========================

section("Admission Type by Billing Category (%)")
stacked_percent_plot(
    df=df,
    x_col="Billing Category",
    hue_col="Admission Type",
    title="Admission Type by Billing Category (%)",
    xlabel="Billing Category",
    ylabel="Percentage (%)",
    save_path="../figures/07_admission_type_by_billing_category.png",
    order=billing_order,
    hue_order=["Urgent", "Emergency", "Elective"]
)
