from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR/"model"
REPORT_DIR = BASE_DIR/"reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Creating a synthetic imbalanced dataset....")

X,y = make_classification(n_samples=3000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    flip_y=0.01,
    random_state=42,)

feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]

print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Class 0 samples:", int(np.sum(y == 0)))
print("Class 1 samples:", int(np.sum(y == 1)))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42),),])

print("\nTraining Logistic Regression...")
model.fit(X_train,y_train)

y_prob = model.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score(y_test,y_prob)

thresholds_to_test = [0.30, 0.50, 0.70]

report_lines = [
    "DAY 26: ROC-AUC AND THRESHOLD TUNING",
    "=" * 45,
    "",
    "Dataset: synthetic imbalanced binary classification",
    f"Training samples: {len(X_train)}",
    f"Testing samples: {len(X_test)}",
    f"ROC-AUC: {roc_auc:.4f}",
    "",
    "THRESHOLD COMPARISON",
    "-" * 45,
]

results = []

for threshold in thresholds_to_test:
    y_pred = (y_prob>=threshold).astype(int)
    matrix = confusion_matrix(y_test,y_pred,labels=[0,1])
    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    report_lines.extend(
        [
            "",
            f"Threshold: {threshold:.2f}",
            f"Accuracy: {accuracy:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1-score: {f1:.4f}",
            f"True negatives: {tn}",
            f"False positives: {fp}",
            f"False negatives: {fn}",
            f"True positives: {tp}",
            "Confusion matrix:",
            str(matrix),
        ]
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix:")
    print(matrix)


fpr,tpr,_ = roc_curve(y_test,y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "roc_curve.png", dpi=150)
plt.close()

# --------------------------------------------------
# Chart 2: Precision-recall curve
# --------------------------------------------------

precision_values, recall_values, _ = precision_recall_curve(
    y_test, y_prob
)

plt.figure(figsize=(7, 5))
plt.plot(recall_values, precision_values)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(REPORT_DIR / "precision_recall_curve.png", dpi=150)
plt.close()

# --------------------------------------------------
# Chart 3: Compare metrics across thresholds
# --------------------------------------------------

threshold_values = [item["threshold"] for item in results]

plt.figure(figsize=(8, 5))

for metric in ["precision", "recall", "f1"]:
    metric_values = [item[metric] for item in results]
    plt.plot(
        threshold_values,
        metric_values,
        marker="o",
        label=metric.capitalize(),
    )

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Metric Comparison by Threshold")
plt.xticks(threshold_values)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "threshold_comparison.png", dpi=150)
plt.close()

report_path = REPORT_DIR / "evaluation_report.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")


package = {
    "model": model,
    "feature_names": feature_names,
    "thresholds": thresholds_to_test,
    "demo_features": X_test[0].tolist(),
    "demo_true_label": int(y_test[0]),
    "roc_auc": float(roc_auc),
}

joblib.dump(package, MODEL_DIR / "threshold_model.joblib")

print("\nROC-AUC:", round(roc_auc, 4))
print("Model saved to:", MODEL_DIR / "threshold_model.joblib")
print("Report saved to:", report_path)
print("Charts saved to:", REPORT_DIR)
print("\nTraining and evaluation completed!")