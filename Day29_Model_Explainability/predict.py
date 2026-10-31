
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "explainable_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run train_model.py first."
    )

package = joblib.load(MODEL_PATH)

model = package["model"]

features = np.array(
    package["demo_features"]
).reshape(1, -1)

actual_label = package["demo_true_label"]

prediction = int(model.predict(features)[0])

probabilities = model.predict_proba(features)[0]

print("DAY 29: MODEL PREDICTION")
print("=" * 40)

print("Actual class:", actual_label)
print("Predicted class:", prediction)

print(
    "Prediction matches actual:",
    prediction == actual_label,
)

print("\nClass probabilities:")

for class_label, probability in enumerate(probabilities):
    print(f"Class {class_label}: {probability:.4f}")
