from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"model"/"calibrated_model.joblib"

package = joblib.load(MODEL_PATH)

model = package["model"]
target_names = package["target_names"]
feature_names = package["feature_names"]

features = np.array(package["demo_features"]).reshape(1, -1)

predicted_class = int(model.predict(features)[0])
probabilities = model.predict_proba(features)[0]

print("Model loaded successfully!")
print("Number of input features:", len(feature_names))
print("Predicted class:", target_names[predicted_class])
print(
    "Estimated probability for predicted class:",
    f"{probabilities[predicted_class] * 100:.2f}%",
)

print("\nProbabilities for each class:")
for index, name in enumerate(target_names):
    print(f"{name}: {probabilities[index] * 100:.2f}%")
