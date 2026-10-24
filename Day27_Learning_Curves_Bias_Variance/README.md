# Day 27 - Learning Curves and Bias-Variance Analysis

## Objective
Compare Logistic Regression and Random Forest using learning
curves and cross-validation.

## Concepts
- Training and validation accuracy
- Learning curves
- High bias and high variance
- Cross-validation
- Model comparison
- Held-out test evaluation
- Model persistence with joblib

## Dataset
Synthetic binary classification dataset generated with
scikit-learn's make_classification.

## Project Structure
- train_model.py: trains and compares the models
- predict.py: loads the saved model and predicts a sample
- model/: saved model package
- reports/: evaluation report and learning-curve charts

## Run the Project

Install dependencies from the repository root:

    .\.venv\Scripts\python.exe -m pip install -r .\Day27_Learning_Curves_Bias_Variance\requirements.txt

From this project directory, train the models:

    ..\.venv\Scripts\python.exe train_model.py

Run the prediction script:

    ..\.venv\Scripts\python.exe predict.py

## Generated Files
- model/best_model.joblib
- reports/learning_curves_report.txt
- reports/logistic_regression_learning_curve.png
- reports/random_forest_learning_curve.png

## Important Note
This is an educational project using synthetic data.
The highest cross-validation accuracy does not guarantee the
best choice for every real-world task.