
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Create dataset
print("Creating classification dataset...")

X, y = make_classification(
    n_samples=3000,
    n_features=15,
    n_informative=8,
    n_redundant=3,
    n_repeated=0,
    n_classes=2,
    weights=[0.80, 0.20],
    flip_y=0.02,
    random_state=42,
)

feature_names = [
    f"feature_{i + 1}" for i in range(X.shape[1])
]

print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Class 0:", int(np.sum(y == 0)))
print("Class 1:", int(np.sum(y == 1)))


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Train Random Forest
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)


# 5. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

matrix = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

print("\nMODEL EVALUATION")
print("=" * 40)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("\nConfusion matrix:")
print(matrix)


# 6. Built-in feature importance
print("\nCalculating built-in feature importance...")

built_in_importance = model.feature_importances_

built_in_order = np.argsort(
    built_in_importance
)[::-1]

print("\nBUILT-IN FEATURE IMPORTANCE")

for index in built_in_order:
    print(
        f"{feature_names[index]}: "
        f"{built_in_importance[index]:.4f}"
    )


# 7. Permutation importance
print("\nCalculating permutation importance...")

permutation_result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="f1",
    n_jobs=1,
)

permutation_means = permutation_result.importances_mean
permutation_stds = permutation_result.importances_std

permutation_order = np.argsort(
    permutation_means
)[::-1]

print("\nPERMUTATION IMPORTANCE")

for index in permutation_order:
    print(
        f"{feature_names[index]}: "
        f"{permutation_means[index]:.4f}"
    )


# 8. Built-in importance chart
top_count = 10
top_builtin_indices = built_in_order[:top_count]

plt.figure(figsize=(9, 6))

plt.barh(
    [
        feature_names[index]
        for index in top_builtin_indices
    ][::-1],
    built_in_importance[top_builtin_indices][::-1],
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Built-in Feature Importance")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "built_in_feature_importance.png",
    dpi=150,
)
plt.close()


# 9. Permutation importance chart
top_permutation_indices = permutation_order[:top_count]

plt.figure(figsize=(9, 6))

plt.barh(
    [
        feature_names[index]
        for index in top_permutation_indices
    ][::-1],
    permutation_means[top_permutation_indices][::-1],
    xerr=permutation_stds[top_permutation_indices][::-1],
)

plt.xlabel("Decrease in F1-score after shuffling")
plt.ylabel("Features")
plt.title("Permutation Feature Importance")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "permutation_feature_importance.png",
    dpi=150,
)
plt.close()


# 10. Save evaluation report
report_lines = [
    "DAY 29: MODEL EXPLAINABILITY",
    "=" * 45,
    "",
    f"Training samples: {len(X_train)}",
    f"Testing samples: {len(X_test)}",
    "",
    "MODEL EVALUATION",
    "-" * 40,
    f"Accuracy: {accuracy:.4f}",
    f"Precision: {precision:.4f}",
    f"Recall: {recall:.4f}",
    f"F1-score: {f1:.4f}",
    "",
    "CONFUSION MATRIX",
    str(matrix),
    "",
    "BUILT-IN FEATURE IMPORTANCE",
    "-" * 40,
]

for index in built_in_order:
    report_lines.append(
        f"{feature_names[index]}: "
        f"{built_in_importance[index]:.4f}"
    )

report_lines.extend([
    "",
    "PERMUTATION IMPORTANCE",
    "-" * 40,
])

for index in permutation_order:
    report_lines.append(
        f"{feature_names[index]}: "
        f"{permutation_means[index]:.4f} "
        f"+/- {permutation_stds[index]:.4f}"
    )

report_lines.extend([
    "",
    "Note: Feature importance measures model reliance,",
    "not causation.",
])

report_path = REPORT_DIR / "evaluation_report.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# 11. Save trained model
package = {
    "model": model,
    "feature_names": feature_names,
    "demo_features": X_test[0].tolist(),
    "demo_true_label": int(y_test[0]),
    "accuracy": float(accuracy),
    "f1_score": float(f1),
}

model_path = MODEL_DIR / "explainable_model.joblib"

joblib.dump(package, model_path)


# 12. Finish
print("\nModel saved to:", model_path)
print("Report saved to:", report_path)
print("Charts saved to:", REPORT_DIR)

print("\nDay 29 training completed!")
