import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"model"/"wine_pipeline.joblib"


if not MODEL_PATH.exists():
    raise SystemExit("Model not Found!!")

model_package = joblib.load(MODEL_PATH)

pipeline = model_package["model"]
target_names = model_package["target_names"]
feature_names = model_package["feature_names"]

sample = [model_package["demo_features"]]
true_class = int(model_package["demo_true_class"])

predicted_class = int(pipeline.predict(sample)[0])
predicted_name = target_names[predicted_class]
actual_name = target_names[true_class]

probabilities = pipeline.predict_proba(sample)[0]

probability_results = {
    name:round(float(probability),4)
    for name ,probability in zip(target_names,probabilities)

}

print("====== WINE PREDICTION ======")

print("Number of input features:", len(feature_names))
print("Actual class:", actual_name)
print("Predicted class:", predicted_name)


print("\nEstimated probabilities:")

for name,probability in probability_results.items():
    print(f"{name} : {probability}")

if predicted_class == true_class:
    print("\nPrediction matches the known test label.")

else:
    print("\nPrediction differs from the known test label.")

