import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==========================
# Setup
# ==========================

plt.rcParams["figure.dpi"] = 80
sns.set_style("whitegrid")
sns.set_context("talk")

df = pd.read_csv("../data/disease_symptom_and_patient_profile_dataset.csv")
os.makedirs("../figures", exist_ok=True)

symptoms = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]

def section(title):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

# ==========================
# 01) Outcome Distribution
# ==========================

section("Outcome Distribution")

plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df,
    x="Outcome Variable",
    hue="Outcome Variable",
    palette={"Positive": "#2ECC71", "Negative": "#E74C3C"},
    legend=False
)

plt.title("Patient Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Number of Patients")

for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(
        f"{height}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        xytext=(0, 6),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )

plt.tight_layout()
plt.savefig("../figures/01_outcome_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 02) Top 10 Diseases
# ==========================

section("Top 10 Most Common Diseases")

top10_counts = df["Disease"].value_counts().head(10)

plt.figure(figsize=(9, 9))
plt.pie(
    top10_counts.values,
    labels=top10_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white"}
)

plt.title("Top 10 Most Common Diseases (Share of Patients)")
plt.tight_layout()
plt.savefig("../figures/02_top10_diseases_pie.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 03) Outcome % for Top 10 Diseases
# ==========================

section("Top 10 Diseases - Outcome Percentage")

top10 = df["Disease"].value_counts().head(10).index
df_top10 = df[df["Disease"].isin(top10)]

counts = pd.crosstab(df_top10["Disease"], df_top10["Outcome Variable"])
percentages = counts.div(counts.sum(axis=1), axis=0) * 100

for col in ["Negative", "Positive"]:
    if col not in percentages.columns:
        percentages[col] = 0
percentages = percentages[["Negative", "Positive"]]
percentages = percentages.loc[top10]

fig, ax = plt.subplots(figsize=(10, 6))
percentages.plot(
    kind="bar",
    stacked=True,
    color={"Negative": "#E74C3C", "Positive": "#2ECC71"},
    ax=ax
)

ax.set_title("Outcome Percentage for Top 10 Diseases")
ax.set_xlabel("Disease")
ax.set_ylabel("Percentage (%)")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Outcome")

for p in ax.patches:
    h = p.get_height()
    if h > 3:
        x = p.get_x() + p.get_width() / 2
        y = p.get_y() + h / 2
        ax.text(x, y, f"{h:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

plt.tight_layout()
plt.savefig("../figures/03_top10_disease_outcome_percentage.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 04) Symptom Presence (%) by Top 10 Diseases
# ==========================

section("Symptom Presence (%) by Top 10 Diseases")

df_hm = df[df["Disease"].isin(top10)].copy()

for s in symptoms:
    df_hm[s] = df_hm[s].map({"Yes": 1, "No": 0})

symptom_pct = df_hm.groupby("Disease")[symptoms].mean() * 100
symptom_pct = symptom_pct.loc[top10]  # same order as Top 10

plt.figure(figsize=(10, 6))
ax = sns.heatmap(symptom_pct, annot=True, fmt=".1f", cmap="Blues")

plt.title("Symptom Presence (%) for Top 10 Diseases")
plt.xlabel("Symptom")
plt.ylabel("Disease")

plt.tight_layout()
plt.savefig("../figures/04_symptoms_by_disease_percent.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 05) Symptom Impact
# ==========================

section("Symptom Impact: Positive Rate vs. Frequency")

df_rate = df.copy()
df_rate["Outcome_num"] = df_rate["Outcome Variable"].map({"Positive": 1, "Negative": 0})

combined_data = []
for s in symptoms:
    pos_rate = df_rate.loc[df_rate[s] == "Yes", "Outcome_num"].mean() * 100
    total_count = (df_rate[s] == "Yes").sum()
    combined_data.append({"Symptom": s, "Positive Rate (%)": pos_rate, "Patient Count": total_count})

impact_df = pd.DataFrame(combined_data).sort_values("Positive Rate (%)", ascending=False)

plt.figure(figsize=(11, 6))
ax = sns.barplot(data=impact_df, x="Positive Rate (%)", y="Symptom", palette="viridis", hue="Symptom", legend=False)

plt.title("Symptom Prevalence vs. Diagnostic Correlation", pad=20)
plt.xlabel("Positive Outcome Rate (%)")
plt.ylabel("Symptom")
plt.xlim(0, 100)

for i, p in enumerate(ax.patches):
    width = p.get_width()
    count = impact_df.iloc[i]["Patient Count"]

    ax.text(width - 5, p.get_y() + p.get_height() / 2, f"{width:.1f}%",
            va="center", ha="right", color="white", fontweight="bold", fontsize=12)

    ax.text(width + 2, p.get_y() + p.get_height() / 2, f"(n={count} patients)",
            va="center", ha="left", color="black", fontstyle="italic", fontsize=11)

plt.tight_layout()
plt.savefig("../figures/05_symptom_impact.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================
# 06) Outcome % by Age Group
# ==========================

section("Outcome Percentage by Age Group")

df_age = df.copy()
df_age["Age Group"] = pd.cut(
    df_age["Age"],
    bins=[0, 30, 45, 60, 100],
    labels=["<30", "30-45", "45-60", "60+"]
)

age_outcome = pd.crosstab(
    df_age["Age Group"],
    df_age["Outcome Variable"],
    normalize="index"
) * 100

for col in ["Negative", "Positive"]:
    if col not in age_outcome.columns:
        age_outcome[col] = 0
age_outcome = age_outcome[["Negative", "Positive"]]

fig, ax = plt.subplots(figsize=(8, 5))
age_outcome.plot(
    kind="bar",
    stacked=True,
    color={"Negative": "#E74C3C", "Positive": "#2ECC71"},
    ax=ax
)

ax.set_title("Outcome Percentage by Age Group")
ax.set_xlabel("Age Group")
ax.set_ylabel("Percentage (%)")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc='upper left')

for c in ax.containers:
    labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("../figures/06_outcome_by_age_group.png", dpi=300, bbox_inches="tight")
plt.show()


