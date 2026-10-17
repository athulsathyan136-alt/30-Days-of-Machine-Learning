import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "wine_pipeline.joblib"
REPORT_PATH = REPORT_DIR/"evaluation_report.txt"


wine = load_wine()

X = wine.data
y = wine.target

target_names = wine.target_names.tolist()
feature_names = wine.feature_names

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=42,
    stratify=y,
    test_size=0.2
)

pipeline = Pipeline(steps=[("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2000))])


pipeline.fit(X_train,y_train)

prediction = pipeline.predict(X_test)

accuracy = accuracy_score(y_test,prediction)

report = classification_report(
    y_test,prediction,
    target_names=target_names,
    zero_division=0
)

matrix = confusion_matrix(y_test,prediction)

MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

model_package = {
    "model": pipeline,
    "target_names": target_names,
    "feature_names": feature_names,
    "demo_features": X_test[0].tolist(),
    "demo_true_class": int(y_test[0])
}

joblib.dump(model_package,MODEL_PATH)


report_text = ('WINE CLASSIFIACTION EVALUATION REPORT\n''==========================\n'
               f"Total samples: {len(X)}\n"
               f"Total features: {X.shape[1]}\n"
               f"Trainig sample: {len(X_train)}\n"
               f"Testing sample: {len(X_test)}\n"
               f"Accuracy : {accuracy:.2f}\n\n"
               "CLASSIFICATION REPORT\n"
               "--------------------------\n"
               f"{report}\n"
               "----------------\n"
               f"{matrix}\n")


REPORT_PATH.write_text(report_text,encoding="utf-8")


print("====== WINE CLASSIFICATION PIPELINE ======")
print("Total samples:", len(X))
print("Features:", X.shape[1])
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(report)

print("Confusion matrix:")
print(matrix)

print("\nSaved pipeline:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
