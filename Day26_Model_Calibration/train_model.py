
from pathlib import Path

import joblib
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# 1. Load the dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

print("Dataset loaded successfully")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Classes:", dataset.target_names)


# 2. Keep a separate test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# 3. Build the original model
base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000)),
])

base_model.fit(X_train, y_train)


# 4. Calibrate using cross-validation on training data
calibrated_model = CalibratedClassifierCV(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]),
    method="sigmoid",
    cv=5,
)

calibrated_model.fit(X_train, y_train)


# 5. Compare probabilities on the untouched test set
base_probabilities = base_model.predict_proba(X_test)[:, 1]
calibrated_probabilities = calibrated_model.predict_proba(X_test)[:, 1]

base_predictions = base_model.predict(X_test)
calibrated_predictions = calibrated_model.predict(X_test)

base_brier = brier_score_loss(y_test, base_probabilities)
calibrated_brier = brier_score_loss(y_test, calibrated_probabilities)

base_log_loss = log_loss(y_test, base_model.predict_proba(X_test))
calibrated_log_loss = log_loss(
    y_test,
    calibrated_model.predict_proba(X_test),
)

base_accuracy = accuracy_score(y_test, base_predictions)
calibrated_accuracy = accuracy_score(y_test, calibrated_predictions)


# 6. Display results
print("\n--- Original Model ---")
print("Accuracy:", round(base_accuracy, 4))
print("Brier score:", round(base_brier, 4))
print("Log loss:", round(base_log_loss, 4))

print("\n--- Calibrated Model ---")
print("Accuracy:", round(calibrated_accuracy, 4))
print("Brier score:", round(calibrated_brier, 4))
print("Log loss:", round(calibrated_log_loss, 4))


# 7. Save the calibrated model and a sample for prediction
package = {
    "model": calibrated_model,
    "target_names": dataset.target_names.tolist(),
    "feature_names": dataset.feature_names.tolist(),
    "demo_features": X_test[0].tolist(),
    "demo_true_class": int(y_test[0]),
}

joblib.dump(package, MODEL_DIR / "calibrated_model.joblib")


# 8. Save a report
report = f"""
DAY 26: MODEL CALIBRATION REPORT

Dataset: Breast Cancer Wisconsin (built into scikit-learn)
Training samples: {len(X_train)}
Testing samples: {len(X_test)}
Number of features: {X.shape[1]}

Original Model
Accuracy: {base_accuracy:.4f}
Brier score: {base_brier:.4f}
Log loss: {base_log_loss:.4f}

Calibrated Model
Accuracy: {calibrated_accuracy:.4f}
Brier score: {calibrated_brier:.4f}
Log loss: {calibrated_log_loss:.4f}

Interpretation:
Lower Brier score and lower log loss generally indicate
better probability predictions on this test set.
Calibration does not necessarily improve classification accuracy.
"""

report_path = REPORT_DIR / "calibration_report.txt"
report_path.write_text(report.strip(), encoding="utf-8")

print("\nSaved model:", MODEL_DIR / "calibrated_model.joblib")
print("Saved report:", report_path)
print("\nTraining and evaluation completed!")
