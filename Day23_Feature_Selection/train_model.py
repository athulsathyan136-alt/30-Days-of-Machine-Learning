from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR/"model"
REPORT_DIR = BASE_DIR/"reports"

MODEL_PATH = MODEL_DIR/"feature_selection_model.joblib"
REPORT_PATH = REPORT_DIR /"evaluation_report.txt"

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

target_names = cancer.target_names.tolist()
feature_names = cancer.feature_names.tolist()

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

baseline_model = Pipeline(steps=[("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=5000)),])

selected_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", LogisticRegression(max_iter=5000)),
    ]
)

print("Training baseline model....")
baseline_model.fit(X_train,y_train)
print("Training feature-selection model...")
selected_model.fit(X_train,y_train)

baseline_prediction = baseline_model.predict(X_test)

selected_prediction = selected_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test,baseline_prediction)

selected_accuracy = accuracy_score(y_test,selected_prediction)

selector = selected_model.named_steps["feature_selection"]

selected_feature_names = [name for name , is_selected in zip(feature_names,selector.get_support())if is_selected]

report = classification_report(y_test,selected_prediction,target_names=target_names,labels=[0,1],zero_division=0,)

matrix = confusion_matrix(y_test,selected_prediction,labels=[0,1])


MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)


model_package = {"model": selected_model,
    "target_names": target_names,
    "feature_names": feature_names,
    "selected_feature_names": selected_feature_names,
    "demo_features": X_test[0].tolist(),
    "demo_true_class": int(y_test[0]),}


joblib.dump(model_package,MODEL_PATH)

report_text = ("FEATURE SELECTION - CLASSIFICATION REPORT\n"
    "=========================================\n"
    f"Dataset samples: {len(X)}\n"
    f"Total features: {X.shape[1]}\n"
    f"Selected features: {len(selected_feature_names)}\n"
    f"Training samples: {len(X_train)}\n"
    f"Testing samples: {len(X_test)}\n\n"
    f"Baseline accuracy (all features): {baseline_accuracy:.2%}\n"
    f"Selected-model accuracy: {selected_accuracy:.2%}\n\n"
    "SELECTED FEATURES\n"
    "-----------------\n"
    + "\n".join(selected_feature_names)
    + "\n\n"
    "CLASSIFICATION REPORT\n"
    "---------------------\n"
    f"{report}\n"
    "CONFUSION MATRIX\n"
    "----------------\n"
    f"{matrix}\n"
)

REPORT_PATH.write_text(report_text,encoding="utf-8")

print("\n====== FEATURE SELECTION RESULTS ======")

print("Total samples:", len(X))
print("Total features:", X.shape[1])
print("Selected features:", len(selected_feature_names))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(f"\nBaseline accuracy: {baseline_accuracy:.2%}")
print(f"Selected-model accuracy: {selected_accuracy:.2%}")

print("\nSelected features:")

for name in selected_feature_names:
    print("-", name)

print("\nClassification report:")
print(report)

print("Confusion matrix:")
print(matrix)

print("\nSaved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)

