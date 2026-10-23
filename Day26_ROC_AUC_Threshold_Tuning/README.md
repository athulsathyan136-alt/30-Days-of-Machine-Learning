# Day 26 - ROC-AUC and Threshold Tuning

## Objective
Learn how probability thresholds affect binary classification
and evaluate a model using ROC-AUC, precision, recall, and F1-score.

## Dataset
A synthetic imbalanced binary classification dataset created
with scikit-learn's make_classification function.

## Concepts
- Logistic Regression
- Feature scaling with StandardScaler
- Class imbalance
- Probability predictions
- Decision thresholds
- ROC-AUC
- Precision and recall
- F1-score
- Confusion matrices

## Project Structure
- train_model.py: trains and evaluates the model
- predict.py: loads the saved model and compares thresholds
- model/: saved model package
- reports/: evaluation report and charts

## Run the Project

Install dependencies:

    ..\.venv\Scripts\python.exe -m pip install -r requirements.txt

Train and evaluate:

    ..\.venv\Scripts\python.exe train_model.py

Run the saved model:

    ..\.venv\Scripts\python.exe predict.py

## Generated Reports
- evaluation_report.txt
- roc_curve.png
- precision_recall_curve.png
- threshold_comparison.png

## Important Note
This project uses synthetic data for educational purposes.
The selected threshold is not automatically the best threshold
for every real-world problem.