# Day 28 - Ensemble Learning

## Objective

Compare individual and ensemble machine learning models
to understand how combining models can improve prediction.

## Models

1. Decision Tree
2. Random Forest
3. Gradient Boosting
4. Voting Classifier

## Concepts

- Decision Trees
- Bagging
- Boosting
- Soft Voting
- Ensemble Learning
- Classification Metrics
- F1-score
- ROC-AUC

## Dataset

Synthetic binary classification dataset generated using
scikit-learn's make_classification.

## Project Structure

- train_model.py: trains and compares all four models
- predict.py: loads the best model and predicts a sample
- model/: stores the trained model
- reports/: contains model evaluation and charts

## Run the Project

Install dependencies from the repository root:

    .\.venv\Scripts\python.exe -m pip install -r .\Day28_Ensemble_Learning\requirements.txt

Navigate to the project folder and train:

    ..\.venv\Scripts\python.exe train_model.py

Run prediction:

    ..\.venv\Scripts\python.exe predict.py

## Generated Files

- model/best_model.joblib
- reports/evaluation_report.txt
- reports/model_comparison.png
- reports/roc_comparison.png

## Model Selection

The model with the highest F1-score on the held-out
test set is selected for this educational demonstration.

## Note

This project uses synthetic data and is intended
for learning purposes.