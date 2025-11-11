import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("mental-heath-in-tech-2016_20161114.csv")

df_original = df.copy()

drop_cols = [
    "comments",
    "Timestamp",
    "state",
    "Country",
    "What US state or territory do you live in?",
    "What country do you work in?",
    "What US state or territory do you work in?",
    "If yes, what condition(s) have you been diagnosed with?",
    "If maybe, what condition(s) do you believe you have?",
    "If so, what condition(s) were you diagnosed with?",
    "Why or why not?",
    "Why or why not?.1",
]

df_encoded = df.drop(columns=[c for c in drop_cols if c in df.columns])
df_encoded = df_encoded.fillna(
    df_encoded.mode().iloc[0] if not df_encoded.mode().empty else 0
)


original_columns = {}
for col in df_encoded.select_dtypes(include=["object"]).columns:
    original_columns[col] = df_encoded[col].copy()
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

fig = plt.figure(figsize=(14, 10))


ax1 = plt.subplot(2, 2, 1)
gender_data = df_original["What is your gender?"].fillna("Unknown")
gender_clean = gender_data.str.lower().str.strip()
gender_map = {
    "male": "Male",
    "m": "Male",
    "man": "Male",
    "female": "Female",
    "f": "Female",
    "woman": "Female",
    "w": "Female",
}
gender_clean = gender_clean.replace(gender_map)
gender_clean = gender_clean.apply(
    lambda x: x.title() if x in ["Male", "Female"] else "Other"
)
gender_counts = gender_clean.value_counts()

bars = ax1.bar(
    gender_counts.index, gender_counts.values, color=["#3498db", "#e74c3c", "#95a5a6"]
)
ax1.set_title("Gender Distribution", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Count", fontsize=11)
ax1.set_xlabel("Gender", fontsize=11)
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax2 = plt.subplot(2, 2, 2)
treatment_col = "Have you ever sought treatment for a mental health issue from a mental health professional?"
treatment_raw = df_original[treatment_col]


treatment_converted = treatment_raw.map({1: "Yes", 0: "No"}).fillna("Unknown")
treatment_counts = treatment_converted.value_counts()


categories = ["Yes", "No"]
values = [treatment_counts.get(cat, 0) for cat in categories]
bars = ax2.bar(categories, values, color=["#2ecc71", "#e74c3c"])
ax2.set_title(
    "Ever Sought Mental Health Treatment?", fontsize=13, fontweight="bold", pad=10
)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_xlabel("Response", fontsize=11)
for i, (cat, val) in enumerate(zip(categories, values)):
    ax2.text(
        i,
        val,
        f"{val}\n({val/len(df_original)*100:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax3 = plt.subplot(2, 2, 3)
current_mh_col = "Do you currently have a mental health disorder?"
mh_raw = df_original[current_mh_col].fillna("Unknown")
mh_counts = mh_raw.value_counts()

bars = ax3.bar(
    mh_counts.index,
    mh_counts.values,
    color=["#3498db", "#e74c3c", "#f39c12", "#95a5a6"],
)
ax3.set_title("Current Mental Health Disorder?", fontsize=13, fontweight="bold", pad=10)
ax3.set_ylabel("Count", fontsize=11)
ax3.set_xlabel("Response", fontsize=11)
ax3.tick_params(axis="x", rotation=0)

for bar in bars:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


ax4 = plt.subplot(2, 2, 4)
ages = pd.to_numeric(df_original["What is your age?"], errors="coerce")
ages_clean = ages[(ages >= 18) & (ages <= 80)].dropna()
ax4.hist(ages_clean, bins=25, color="#9b59b6", alpha=0.7, edgecolor="black")
ax4.set_title("Age Distribution", fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("Age", fontsize=11)
ax4.set_ylabel("Count", fontsize=11)
ax4.axvline(
    ages_clean.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {ages_clean.mean():.1f}",
)
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig("demographic_analysis.png", dpi=300, bbox_inches="tight")
plt.close()


scaler = StandardScaler()
scaled = scaler.fit_transform(df_encoded)

pca = PCA(n_components=2)
reduced = pca.fit_transform(scaled)
df_encoded["PC1"], df_encoded["PC2"] = reduced[:, 0], reduced[:, 1]

print(f"PCA explained variance: {pca.explained_variance_ratio_}")


plt.figure(figsize=(10, 8))
plt.scatter(df_encoded["PC1"], df_encoded["PC2"], alpha=0.5, s=40, color="#3498db")
plt.title("PCA Projection of Survey Data", fontsize=14, fontweight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_projection.png", dpi=300, bbox_inches="tight")
plt.close()


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_encoded["Cluster"] = kmeans.fit_predict(scaled)

plt.figure(figsize=(10, 8))
colors = ["#e74c3c", "#3498db", "#2ecc71"]
for i in range(3):
    cluster_data = df_encoded[df_encoded["Cluster"] == i]
    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        c=colors[i],
        label=f"Cluster {i}",
        alpha=0.7,
        s=50,
    )
plt.title("Clusters of Survey Participants", fontsize=14, fontweight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("clusters.png", dpi=300, bbox_inches="tight")
plt.close()


df_original["Cluster"] = df_encoded["Cluster"]


questions = {
    "Sought Treatment": "Have you ever sought treatment for a mental health issue from a mental health professional?",
    "Current Disorder": "Do you currently have a mental health disorder?",
    "Past Disorder": "Have you had a mental health disorder in the past?",
    "Productivity Affected": "Do you believe your productivity is ever affected by a mental health issue?",
}


cluster_stats = []
for cluster_id in range(3):
    cluster_mask = df_original["Cluster"] == cluster_id
    stats = {"Cluster": f"Cluster {cluster_id}", "Size": cluster_mask.sum()}

    for short_name, full_question in questions.items():
        if full_question in df_original.columns:
            responses = df_original.loc[cluster_mask, full_question]

            if short_name == "Sought Treatment":

                yes_count = (responses == 1).sum()
            else:

                yes_count = (responses == "Yes").sum()
            total = len(responses)
            stats[short_name] = (yes_count / total * 100) if total > 0 else 0

    cluster_stats.append(stats)


fig, ax = plt.subplots(figsize=(14, 8))

question_names = list(questions.keys())
x = np.arange(len(question_names))
width = 0.25

colors = ["#e74c3c", "#3498db", "#2ecc71"]
for i in range(3):
    values = [cluster_stats[i][q] for q in question_names]
    bars = ax.bar(
        x + i * width,
        values,
        width,
        label=f'Cluster {i} (n={cluster_stats[i]["Size"]})',
        color=colors[i],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar in bars:
        height = bar.get_height()
        if height > 5:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2,
                f"{height:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

ax.set_ylabel('Percentage Responding "Yes" (%)', fontsize=12, fontweight="bold")
ax.set_xlabel("Mental Health Questions", fontsize=12, fontweight="bold")
ax.set_title(
    "Cluster Comparison: Mental Health Indicators",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(x + width)
ax.set_xticklabels(question_names, fontsize=10)
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("cluster_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("CLUSTER INTERPRETATION")
print("=" * 70)

for i in range(3):
    cluster_mask = df_original["Cluster"] == i
    cluster_data = df_original[cluster_mask]

    print(f"\n{'='*70}")
    print(
        f"CLUSTER {i} - {len(cluster_data)} people ({len(cluster_data)/len(df_original)*100:.1f}%)"
    )
    print(f"{'='*70}")

    for short_name, full_question in questions.items():
        if full_question in df_original.columns:
            responses = cluster_data[full_question]

            if short_name == "Sought Treatment":

                yes_pct = (responses == 1).sum() / len(responses) * 100
                no_pct = (responses == 0).sum() / len(responses) * 100
                maybe_pct = 0
            else:

                yes_pct = (responses == "Yes").sum() / len(responses) * 100
                no_pct = (responses == "No").sum() / len(responses) * 100
                maybe_pct = (responses == "Maybe").sum() / len(responses) * 100

            print(f"\n{short_name}:")
            print(
                f"  Yes: {yes_pct:.1f}%  |  No: {no_pct:.1f}%  |  Maybe: {maybe_pct:.1f}%"
            )

            if yes_pct > 60:
                print(f"  → HIGH: Most people in this cluster answered YES")
            elif yes_pct < 30:
                print(f"  → LOW: Most people in this cluster answered NO")
            else:
                print(f"  → MODERATE: Mixed responses")

    ages = pd.to_numeric(cluster_data["What is your age?"], errors="coerce").dropna()
    print(f"\nAge: Mean = {ages.mean():.1f}, Median = {ages.median():.1f}")


df_encoded.to_csv("mental_health_clusters.csv", index=False)
