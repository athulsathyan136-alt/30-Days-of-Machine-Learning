from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Create dataset
# --------------------------------------------------

print("Creating classification dataset...")

X, y = make_classification(
    n_samples=3000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
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
print("Class 0 samples:", int(np.sum(y == 0)))
print("Class 1 samples:", int(np.sum(y == 1)))


# --------------------------------------------------
# 3. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Define base models
# --------------------------------------------------

logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

forest_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=1,
)


# --------------------------------------------------
# 5. Define all four models
# --------------------------------------------------

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
    ),

    "Random Forest": forest_model,

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),

    "Voting Classifier": VotingClassifier(
        estimators=[
            ("logistic", logistic_model),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=8,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ],
        voting="soft",
        n_jobs=1,
    ),
}


# --------------------------------------------------
# 6. Train and evaluate models
# --------------------------------------------------

results = {}
report_lines = [
    "DAY 28: ENSEMBLE LEARNING",
    "=" * 45,
    "",
    "Dataset: synthetic binary classification",
    f"Training samples: {len(X_train)}",
    f"Testing samples: {len(X_test)}",
    "",
]

print("\nTraining and comparing models...")

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    results[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "probabilities": probabilities,
    }

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Confusion matrix:")
    print(matrix)

    report_lines.extend(
        [
            "",
            f"MODEL: {model_name}",
            "-" * 40,
            f"Accuracy: {accuracy:.4f}",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1-score: {f1:.4f}",
            f"ROC-AUC: {roc_auc:.4f}",
            f"True negatives: {tn}",
            f"False positives: {fp}",
            f"False negatives: {fn}",
            f"True positives: {tp}",
            "Confusion matrix:",
            str(matrix),
        ]
    )


# --------------------------------------------------
# 7. Select the best model using F1-score
# --------------------------------------------------

best_name = max(
    results,
    key=lambda name: results[name]["f1"],
)

best_model = results[best_name]["model"]

print("\nBest model:", best_name)

report_lines.extend(
    [
        "",
        "MODEL SELECTION",
        "-" * 40,
        f"Best model: {best_name}",
        f"Best F1-score: {results[best_name]['f1']:.4f}",
        "Selection metric: F1-score",
    ]
)


# --------------------------------------------------
# 8. Create F1-score comparison chart
# --------------------------------------------------

model_names = list(results.keys())

f1_scores = [
    results[name]["f1"] for name in model_names
]

plt.figure(figsize=(9, 5))

plt.bar(model_names, f1_scores)

plt.xlabel("Models")
plt.ylabel("F1-score")
plt.title("Ensemble Model Comparison")

plt.ylim(0, 1.05)
plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_comparison.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 9. Create ROC curve comparison
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for model_name, result in results.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        result["probabilities"],
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={result['roc_auc']:.3f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random ranking",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "roc_comparison.png",
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 10. Save model and report
# --------------------------------------------------

package = {
    "model": best_model,
    "model_name": best_name,
    "feature_names": feature_names,
    "demo_features": X_test[0].tolist(),
    "demo_true_label": int(y_test[0]),
    "best_f1": float(results[best_name]["f1"]),
}

model_path = MODEL_DIR / "best_model.joblib"

joblib.dump(package, model_path)

report_lines.extend(
    [
        "",
        "FINAL RESULT",
        "-" * 40,
        f"Selected model: {best_name}",
        "",
        "All metrics were calculated on the held-out test set.",
    ]
)

report_path = REPORT_DIR / "evaluation_report.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\nModel saved to:", model_path)
print("Report saved to:", report_path)
print("Charts saved to:", REPORT_DIR)

print("\nDay 28 completed!")