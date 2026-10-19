from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR/"model"
REPORT_DIR = BASE_DIR/"reports"

MODEL_PATH = MODEL_DIR /"balanced_model.joblib"

REPORT_PATH = REPORT_DIR/"evaluation_report.txt"

X,y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    flip_y=0.01,
    random_state=42,
)

class_0_count = int((y==0).sum())
class_1_count = int((y==1).sum())

print("=======DATASET INFORMATION============")
print("Total sample: ",len(X))
print("Total features: ",X.shape[1])
print("class 0 sample: ",class_0_count)
print("class 1 sample: ",class_1_count)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

basline_model = Pipeline(steps=[("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=2000)),])

balanced_model = Pipeline(steps=[("scaler",StandardScaler()),("classifier",LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42,),),])

print("\nTraining baseline model....")
basline_model.fit(X_train,y_train)

print("\nTraining balanced model...")
balanced_model.fit(X_train,y_train)

basline_prediction = basline_model.predict(X_test)
balanced_prediction = balanced_model.predict(X_test)


def evaluate_model(y_true,predictions):
    return{
        "accuracy":accuracy_score(y_true,predictions),
        "precision":precision_score(y_true,predictions,pos_label=1,zero_division=0),
        "recall":recall_score(y_true,predictions,pos_label=1,zero_division=0),
        "f1_score":f1_score(y_true,predictions,pos_label=1,zero_division=0)


    }

basline_metrics = evaluate_model(y_test,basline_prediction)
balanced_metrics = evaluate_model(y_test,balanced_prediction)


target_names = ["Majority class (0)", "Minority class (1)"]

balanced_report = classification_report(y_test,balanced_prediction,labels=[0,1],target_names=target_names,zero_division=0,)

basline_matrix = confusion_matrix(y_test,basline_prediction,labels=[0,1])

balanced_matrix = confusion_matrix(y_test,balanced_prediction,labels=[0,1])

MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

minority_indices = [i for i,label in enumerate(y_test) if label == 1]

demo_index = minority_indices[0]

model_package = {
    "model": balanced_model,
    "target_names": target_names,
    "feature_names": [
        f"feature_{i}" for i in range(X.shape[1])
    ],
    "demo_features": X_test[demo_index].tolist(),
    "demo_true_class": int(y_test[demo_index]),
}

joblib.dump(model_package,MODEL_PATH)

report_text = f"""
IMBALANCED DATASET CLASSIFICATION REPORT
========================================

Total samples: {len(X)}
Number of features: {X.shape[1]}
Class 0 samples: {class_0_count}
Class 1 samples: {class_1_count}
Training samples: {len(X_train)}
Testing samples: {len(X_test)}

METRICS FOR MINORITY CLASS (CLASS 1)
====================================

Accuracy               {basline_metrics['accuracy']:.4f}         {balanced_metrics['accuracy']:.4f}
Precision              {basline_metrics['precision']:.4f}       {balanced_metrics['precision']:.4f}
Recall                 {basline_metrics['recall']:.4f}          {balanced_metrics['recall']:.4f}
F1-score               {basline_metrics['f1_score']:.4f}        {balanced_metrics['f1_score']:.4f}

BASELINE CONFUSION MATRIX
=========================
Rows = actual classes; columns = predicted classes
{basline_matrix}

BALANCED MODEL CONFUSION MATRIX
================================
Rows = actual classes; columns = predicted classes
{balanced_matrix}

BALANCED MODEL CLASSIFICATION REPORT
====================================
{balanced_report}
"""


REPORT_PATH.write_text(report_text,encoding="utf-8")


print("\n====== MODEL COMPARISON ======")
print(f"{'Metric':<15} {'Baseline':>12} {'Balanced':>12}")

for metric in ["accuracy", "precision", "recall", "f1_score"]:
    print(
        f"{metric:<15} "
        f"{basline_metrics[metric]:>12.4f} "
        f"{balanced_metrics[metric]:>12.4f}"
    )

print("\nBaseline confusion matrix:")
print(basline_matrix)

print("\nBalanced-model confusion matrix:")
print(balanced_matrix)

print("\nBalanced-model classification report:")
print(balanced_report)

print("\nSaved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)