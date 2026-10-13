from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "iris_model.joblib"

if not MODEL_PATH.exists():
    raise SystemExit("Saved model not found" "Run train_model.py first")

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
target_nameas = model_package["target_names"]
feature_names = model_package["feature_names"]

flower = [[5.1, 3.5, 1.4, 0.2]]

predicted_class = model.predict(flower)[0]
predicted_species = target_nameas[predicted_class]

probabilitess = model.predict_proba(flower)[0]
confidence = probabilitess[predicted_class]

print("\n====== FLOWER PREDICTION ======")
for name , value in zip(feature_names,flower[0]):
    print(f"{name} : {value} cm")

print("\nPredicted species:", predicted_species)
print(f"Model probability for this class: {confidence:.2%}")