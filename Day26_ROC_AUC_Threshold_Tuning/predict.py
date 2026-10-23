from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "threshold_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

package = joblib.load(MODEL_PATH)

model = package["model"]
features = np.array(package["demo_features"]).reshape(1, -1)

true_label = package["demo_true_label"]

# Obtain the probability of class 1.
probability = float(model.predict_proba(features)[0, 1])

print("DAY 26: THRESHOLD TUNING PREDICTION")
print("=" * 40)
print("Actual class:", true_label)
print(f"Probability of class 1: {probability:.4f}")

for threshold in package["thresholds"]:
    prediction = int(probability >= threshold)

    print(
        f"Threshold {threshold:.2f} "
        f"-> Predicted class: {prediction}"
    )