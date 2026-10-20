
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.joblib"

if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run train_model.py first.")

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
model_name = model_package["model_name"]
target_names = model_package["target_names"]

sample = [model_package["demo_features"]]
actual_class = model_package["demo_true_class"]

predicted_class = int(model.predict(sample)[0])
predicted_name = target_names[predicted_class]
actual_name = target_names[actual_class]

print("====== CROSS-VALIDATION PREDICTION ======")
print("Selected model:", model_name)
print("Actual class:", actual_name)
print("Predicted class:", predicted_name)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sample)[0]

    print("\nEstimated class probabilities:")

    for name, probability in zip(target_names, probabilities):
        print(f"{name}: {probability:.2%}")

if predicted_class == actual_class:
    print("\nPrediction matches the known test label.")
else:
    print("\nPrediction differs from the known test label.")
