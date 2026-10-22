
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"model"/"voting_model.joblib"

package = joblib.load(MODEL_PATH)

model = package["model"]
target_names = package["target_names"]
feature_names = package["feature_names"]

features = np.array(package["demo_features"]).reshape(1, -1)

predicted_class = int(model.predict(features)[0])
probabilities = model.predict_proba(features)[0]

print("Voting Ensemble loaded successfully!")
print("Number of input features:", len(feature_names))
print("Actual class:", target_names[package["demo_true_class"]])
print("Predicted class:", target_names[predicted_class])

print("\nPredicted class probabilities:")
for index, name in enumerate(target_names):
    print(f"{name}: {probabilities[index] * 100:.2f}%")