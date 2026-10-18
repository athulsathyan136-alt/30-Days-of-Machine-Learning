from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR/"model"/"feature_selection_model.joblib"

if not MODEL_PATH.exists():
    raise SyntaxError("Model not found. Run train_model.py first")


model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
target_names = model_package["target_names"]
feature_names = model_package["feature_names"]
selected_feature_names = model_package["selected_feature_names"]

sample =[model_package["demo_features"]]
true_class = int(model_package["demo_true_class"])

predicted_class = int(model.predict(sample)[0])

predicted_name = target_names[predicted_class]
actual_name = target_names[true_class]

probabilites= model.predict_proba(sample)[0]

probabilites_results = {
    name:float(probability) for name,probability in zip(target_names,probabilites)


}

print("====== FEATURE SELECTION PREDICTION ======")

print("Input features available:", len(feature_names))
print("Features selected by pipeline:", len(selected_feature_names))

print("\nActual class:", actual_name)
print("Predicted class:", predicted_name)

print("\nEstimated class probabilities:")

for name, probability in probabilites_results.items():
    print(f"{name}: {probability:.2%}")

if predicted_class == true_class:
    print("\nPrediction matches the known test label.")
else:
    print("\nPrediction differs from the known test label.")

print("\nSelected features:")

for name in selected_feature_names:
    print("-", name)

print(
    "\nEducational demonstration only. "
    "Not for medical diagnosis."
)