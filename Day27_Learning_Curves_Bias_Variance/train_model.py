from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
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
# 2. Create a synthetic classification dataset
# --------------------------------------------------

print("Creating the dataset...")

X, y = make_classification(
    n_samples=2000,
    n_features=12,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    class_sep=1.0,
    random_state=42,
)

feature_names = [
    f"feature_{i + 1}" for i in range(X.shape[1])
]

print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])

# Keep a held-out test set separate.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. Define the models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
    ),
}


# --------------------------------------------------
# 4. Define cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_sizes_fraction = np.linspace(0.2, 1.0, 5)

results = {}
report_lines = [
    "DAY 27: LEARNING CURVES AND BIAS-VARIANCE",
    "=" * 50,
    "",
    "Dataset: synthetic binary classification",
    f"Total samples: {len(X)}",
    f"Training samples: {len(X_train)}",
    f"Testing samples: {len(X_test)}",
    "",
]


# --------------------------------------------------
# 5. Generate learning curves for each model
# --------------------------------------------------

for model_name, estimator in models.items():

    print(f"\nAnalyzing: {model_name}")

    train_sizes, train_scores, validation_scores = (
        learning_curve(
            estimator,
            X_train,
            y_train,
            train_sizes=train_sizes_fraction,
            cv=cv,
            scoring="accuracy",
            shuffle=True,
            random_state=42,
            n_jobs=1,
        )
    )

    # Average score across the five CV folds.
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    # A simple gap between training and validation scores
    # at the largest training size.
    final_gap = float(
        train_mean[-1] - validation_mean[-1]
    )

    # Store results for later comparison and plotting.
    results[model_name] = {
        "estimator": estimator,
        "train_sizes": train_sizes,
        "train_mean": train_mean,
        "train_std": train_std,
        "validation_mean": validation_mean,
        "validation_std": validation_std,
        "final_cv_score": float(validation_mean[-1]),
        "final_gap": final_gap,
    }

    report_lines.extend(
        [
            "",
            f"MODEL: {model_name}",
            "-" * 40,
        ]
    )

    for i, size in enumerate(train_sizes):
        line = (
            f"Samples={size}, "
            f"Training accuracy={train_mean[i]:.4f}, "
            f"Validation accuracy={validation_mean[i]:.4f}"
        )

        report_lines.append(line)
        print(line)

    report_lines.extend(
        [
            f"Final training score: {train_mean[-1]:.4f}",
            f"Final validation score: {validation_mean[-1]:.4f}",
            f"Final train-validation gap: {final_gap:.4f}",
        ]
    )

    # ----------------------------------------------
    # 6. Save this model's learning-curve chart
    # ----------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training accuracy",
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Cross-validation accuracy",
    )

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )

    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve: {model_name}")
    plt.legend()
    plt.tight_layout()

    chart_name = (
        model_name.lower().replace(" ", "_")
        + "_learning_curve.png"
    )

    plt.savefig(REPORT_DIR / chart_name, dpi=150)
    plt.close()


# --------------------------------------------------
# 7. Select a model using cross-validation
# --------------------------------------------------

best_name = max(
    results,
    key=lambda name: results[name]["final_cv_score"],
)

best_model = results[best_name]["estimator"]

report_lines.extend(
    [
        "",
        "MODEL COMPARISON",
        "-" * 40,
    ]
)

for model_name, result in results.items():
    report_lines.append(
        f"{model_name}: "
        f"final CV accuracy={result['final_cv_score']:.4f}"
    )

report_lines.extend(
    [
        "",
        f"Selected model: {best_name}",
        "",
        "Selection method: highest mean validation accuracy "
        "at the largest learning-curve training size.",
    ]
)

print("\nSelected model:", best_name)


# --------------------------------------------------
# 8. Check the selection with cross-validation
# --------------------------------------------------

# Cross-validation on training data only.
# The test set remains untouched during model selection.
cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy",
    n_jobs=1,
)

print("Selected model CV scores:", np.round(cv_scores, 4))
print("Mean CV accuracy:", round(cv_scores.mean(), 4))

report_lines.extend(
    [
        "",
        "SELECTED MODEL CROSS-VALIDATION",
        "-" * 40,
        f"Fold scores: {np.round(cv_scores, 4).tolist()}",
        f"Mean CV accuracy: {cv_scores.mean():.4f}",
        f"CV standard deviation: {cv_scores.std():.4f}",
    ]
)


# --------------------------------------------------
# 9. Train the selected model and evaluate the test set
# --------------------------------------------------

best_model.fit(X_train, y_train)

test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Held-out test accuracy:", round(test_accuracy, 4))

report_lines.extend(
    [
        "",
        "HELD-OUT TEST RESULT",
        "-" * 40,
        f"Test accuracy: {test_accuracy:.4f}",
        "",
        "Note: the test set is held out from model selection.",
    ]
)


# --------------------------------------------------
# 10. Save the model package
# --------------------------------------------------

package = {
    "model": best_model,
    "model_name": best_name,
    "feature_names": feature_names,
    "demo_features": X_test[0].tolist(),
    "demo_true_label": int(y_test[0]),
    "test_accuracy": float(test_accuracy),
}

model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(package, model_path)


# --------------------------------------------------
# 11. Save the report
# --------------------------------------------------

report_path = REPORT_DIR / "learning_curves_report.txt"
report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\nModel saved to:", model_path)
print("Report saved to:", report_path)
print("Charts saved to:", REPORT_DIR)
print("\nDay 27 training completed!")