
from pathlib import Path

import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load the dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

print("Dataset loaded successfully!")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", dataset.target_names)


# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 3. Create the individual models
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000)),
])

forest_model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
)

# Soft voting averages predicted probabilities.
voting_model = VotingClassifier(
    estimators=[
        ("logistic", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ])),
        ("forest", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
        )),
    ],
    voting="soft",
)


# 4. Compare all models using the same test set
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model,
    "Soft Voting Ensemble": voting_model,
}

results = {}
report_parts = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 score:", round(f1, 4))

    
    report_parts.append(
        f"\n===== {name} =====\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1 score: {f1:.4f}\n\n"
        f"Confusion matrix:\n{confusion_matrix(y_test, predictions)}\n\n"
        "Classification report:\n"
        + classification_report(
            y_test,
            predictions,
            target_names=dataset.target_names,
            zero_division=0,
        )
    )



# 5. Select the highest-F1 model for the comparison report
best_name = max(results, key=lambda name: results[name]["f1"])

print("\nBest model by F1 score:", best_name)
print("Note: this selection is based on the held-out test set for")
print("reporting only; do not use it to tune a final production model.")


# 6. Save the ensemble separately for prediction
joblib.dump(
    {
        "model": voting_model,
        "target_names": dataset.target_names.tolist(),
        "feature_names": dataset.feature_names.tolist(),
        "demo_features": X_test[0].tolist(),
        "demo_true_class": int(y_test[0]),
    },
    MODEL_DIR / "voting_model.joblib",
)


# 7. Save the evaluation report
summary = [
    "DAY 27: ENSEMBLE LEARNING REPORT",
    f"Training samples: {len(X_train)}",
    f"Testing samples: {len(X_test)}",
    f"Number of features: {X.shape[1]}",
    "",
    *report_parts,
    "",
    f"Highest F1 score in this comparison: {best_name}",
    "",
    "Note: evaluation results are for learning purposes only.",
    "This dataset and model are not for clinical diagnosis.",
]

report_path = REPORT_DIR / "ensemble_report.txt"
report_path.write_text("\n".join(summary), encoding="utf-8")

print("\nSaved model:", MODEL_DIR / "voting_model.joblib")
print("Saved report:", report_path)
print("\nTraining and evaluation completed!")
