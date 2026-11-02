from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
    "Age",
    "AnnualIncome",
    "SpendingScore",
    "PurchaseFrequency",
    "AverageOrderValue",
    "WebsiteVisits",
]


# --------------------------------------------------
# 2. Generate a realistic synthetic customer dataset
# --------------------------------------------------

def generate_customer_data(n_per_segment=250):
    rng = np.random.default_rng(RANDOM_STATE)

    segments = [
        # Age, income, spending, frequency, order value, visits
        ([25, 5], [85000, 12000], [85, 8], [15, 3], [180, 30], [25, 5]),
        ([45, 8], [90000, 15000], [25, 8], [4, 2], [100, 20], [7, 2]),
        ([30, 7], [35000, 8000], [80, 10], [12, 3], [65, 15], [20, 4]),
        ([50, 10], [30000, 7000], [25, 9], [3, 2], [35, 10], [5, 2]),
    ]

    frames = []

    for segment in segments:
        columns = []

        for mean, std in segment:
            values = rng.normal(mean, std, n_per_segment)
            columns.append(values)

        segment_df = pd.DataFrame(
            np.column_stack(columns),
            columns=FEATURES
        )

        frames.append(segment_df)

    customers = pd.concat(frames, ignore_index=True)

    customers["Age"] = customers["Age"].clip(18, 75).round().astype(int)
    customers["AnnualIncome"] = (
        customers["AnnualIncome"].clip(10000, 200000).round(2)
    )
    customers["SpendingScore"] = (
        customers["SpendingScore"].clip(1, 100).round(2)
    )
    customers["PurchaseFrequency"] = (
        customers["PurchaseFrequency"].clip(1, 30).round().astype(int)
    )
    customers["AverageOrderValue"] = (
        customers["AverageOrderValue"].clip(10, 500).round(2)
    )
    customers["WebsiteVisits"] = (
        customers["WebsiteVisits"].clip(1, 50).round().astype(int)
    )

    customers = customers.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    customers.insert(
        0,
        "CustomerID",
        [f"CUST-{i:04d}" for i in range(1, len(customers) + 1)]
    )

    return customers


# --------------------------------------------------
# 3. Create and save dataset
# --------------------------------------------------

print("\nGENERATING CUSTOMER DATASET")
print("=" * 50)

customers = generate_customer_data()

data_path = DATA_DIR / "customers.csv"
customers.to_csv(data_path, index=False)

print(f"Total customers: {len(customers)}")
print(f"Features: {len(FEATURES)}")
print(f"Dataset saved: {data_path}")


# --------------------------------------------------
# 4. Prepare data and standardize features
# --------------------------------------------------

X = customers[FEATURES]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFEATURE SCALING COMPLETED")


# --------------------------------------------------
# 5. Find the best number of clusters
# --------------------------------------------------

print("\nTESTING DIFFERENT CLUSTER COUNTS")
print("=" * 50)

inertias = []
silhouette_scores = []
cluster_counts = range(2, 9)

for k in cluster_counts:
    candidate = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = candidate.fit_predict(X_scaled)

    inertias.append(candidate.inertia_)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"K={k} | Silhouette Score={score:.4f}")

best_k = list(cluster_counts)[int(np.argmax(silhouette_scores))]

print(f"\nSelected number of clusters: {best_k}")


# --------------------------------------------------
# 6. Elbow and silhouette visualizations
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(
    list(cluster_counts),
    inertias,
    marker="o"
)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")
axes[0].grid(True)

axes[1].plot(
    list(cluster_counts),
    silhouette_scores,
    marker="o"
)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(REPORT_DIR / "elbow_method.png", dpi=300)
plt.close()


# --------------------------------------------------
# 7. Train final K-Means model
# --------------------------------------------------

model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

customers["Cluster"] = model.fit_predict(X_scaled)

final_silhouette = silhouette_score(
    X_scaled,
    customers["Cluster"]
)

print("\nFINAL MODEL RESULTS")
print("=" * 50)
print(f"Number of clusters: {best_k}")
print(f"Silhouette score: {final_silhouette:.4f}")


# --------------------------------------------------
# 8. Cluster visualization
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=customers,
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="tab10",
    s=65
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "customer_clusters.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# 9. PCA visualization
# --------------------------------------------------

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    pca_data,
    columns=["PC1", "PC2"]
)

pca_df["Cluster"] = customers["Cluster"].values

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=65
)

plt.title("Customer Clusters Using PCA")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "pca_clusters.png",
    dpi=300
)
plt.close()

explained_variance = pca.explained_variance_ratio_.sum()


# --------------------------------------------------
# 10. Analyze customer segments
# --------------------------------------------------

summary = customers.groupby("Cluster")[FEATURES].mean().round(2)

summary["CustomerCount"] = customers.groupby("Cluster").size()
summary["CustomerPercentage"] = (
    summary["CustomerCount"] / len(customers) * 100
).round(2)

summary_path = REPORT_DIR / "cluster_summary.csv"
summary.to_csv(summary_path)

print("\nCUSTOMER SEGMENT SUMMARY")
print("=" * 50)
print(summary.to_string())


# --------------------------------------------------
# 11. Save clustered customer dataset
# --------------------------------------------------

customers.to_csv(
    DATA_DIR / "segmented_customers.csv",
    index=False
)


# --------------------------------------------------
# 12. Save model and preprocessing package
# --------------------------------------------------

package = {
    "model": model,
    "scaler": scaler,
    "features": FEATURES,
    "best_k": best_k,
    "random_state": RANDOM_STATE,
}

model_path = MODEL_DIR / "customer_segmentation.joblib"

joblib.dump(package, model_path)


# --------------------------------------------------
# 13. Generate evaluation report
# --------------------------------------------------

report = f"""
CUSTOMER SEGMENTATION - MODEL REPORT
====================================

Project: Day 31 Customer Segmentation
Algorithm: K-Means Clustering

DATASET
-------
Total customers: {len(customers)}
Number of features: {len(FEATURES)}
Features: {", ".join(FEATURES)}

MODEL
-----
Algorithm: K-Means
Selected clusters: {best_k}
Random state: {RANDOM_STATE}
Silhouette score: {final_silhouette:.4f}

PCA
---
Explained variance by two components: {explained_variance:.4f}

CLUSTER COUNTS
--------------
{customers["Cluster"].value_counts().sort_index().to_string()}

SEGMENT SUMMARY
---------------
{summary.to_string()}

IMPORTANT
---------
This project uses synthetic customer data.
The results demonstrate a machine learning workflow.
They should not be treated as findings about real customers.
"""

report_path = REPORT_DIR / "evaluation_report.txt"
report_path.write_text(report, encoding="utf-8")


# --------------------------------------------------
# 14. Completion
# --------------------------------------------------

print("\nPROJECT COMPLETED")
print("=" * 50)
print(f"Dataset: {data_path}")
print(f"Segmented data: {DATA_DIR / 'segmented_customers.csv'}")
print(f"Model: {model_path}")
print(f"Report: {report_path}")
print("Charts saved in reports/")
print("\nAll outputs generated successfully!")