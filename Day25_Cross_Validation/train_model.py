
from pathlib import Path

import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# STEP 1: Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
REPORT_PATH = REPORT_DIR / "cross_validation_report.txt"


# STEP 2: Load the dataset
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("====== DATASET INFORMATION ======")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", dataset.target_names)


# STEP 3: Keep a separate test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# STEP 4: Define cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# STEP 5: Define two models
models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    ),
}


# STEP 6: Evaluate each model using training data only
results = {}

print("\n====== 5-FOLD CROSS-VALIDATION ======")

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
    )

    results[name] = {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }

    print(f"\nModel: {name}")
    print("Fold scores:", np.round(scores, 4))
    print(f"Mean accuracy: {np.mean(scores):.2%}")
    print(f"Standard deviation: {np.std(scores):.4f}")


# STEP 7: Select the model with the highest mean CV accuracy
best_name = max(
    results,
    key=lambda name: results[name]["mean"],
)

best_model = models[best_name]

# STEP 8: Fit the chosen model on all training data
best_model.fit(X_train, y_train)

# STEP 9: Evaluate once on the held-out test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)


# STEP 10: Create output folders
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# STEP 11: Save the model and metadata
model_package = {
    "model": best_model,
    "model_name": best_name,
    "target_names": dataset.target_names.tolist(),
    "feature_names": dataset.feature_names.tolist(),
    "demo_features": X_test[0].tolist(),
    "demo_true_class": int(y_test[0]),
}

joblib.dump(model_package, MODEL_PATH)


# STEP 12: Save the report
report_lines = [
    "CROSS-VALIDATION MODEL COMPARISON",
    "================================",
    f"Training samples: {len(X_train)}",
    f"Test samples: {len(X_test)}",
    "Cross-validation folds: 5",
    "",
]

for name, result in results.items():
    report_lines.extend(
        [
            f"Model: {name}",
            f"Fold scores: {np.round(result['scores'], 4)}",
            f"Mean CV accuracy: {result['mean']:.4f}",
            f"Standard deviation: {result['std']:.4f}",
            "",
        ]
    )

report_lines.extend(
    [
        f"Selected model: {best_name}",
        f"Held-out test accuracy: {test_accuracy:.4f}",
    ]
)

REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# STEP 13: Display final results
print("\n====== FINAL MODEL ======")
print("Selected model:", best_name)
print(f"Held-out test accuracy: {test_accuracy:.2%}")
print("Saved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
