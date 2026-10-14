from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=42,
    stratify=y,
    test_size=0.2,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test,prediction)

MODEL_DIR.mkdir(parents=True,exist_ok=True)

model_package = {
    "model": model,
    "target_names": iris.target_names.tolist(),
    "feature_names": iris.feature_names
}

joblib.dump(model_package,MODEL_PATH)

print("====== MODEL TRAINING COMPLETE ======")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Test accuracy: {accuracy:.2%}")
print("Model saved to:", MODEL_PATH)