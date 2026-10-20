# Day 25: Cross-Validation Model Comparison

## Overview

Compare Logistic Regression and Random Forest using
five-fold stratified cross-validation.

## Features

- Loads Scikit-learn's breast cancer dataset
- Splits data into training and test sets
- Uses five-fold stratified cross-validation
- Compares mean accuracy and score variation
- Selects a model based on mean CV accuracy
- Evaluates the selected model on held-out test data
- Saves the fitted model with Joblib
- Generates a text evaluation report

## Technologies

- Python
- NumPy
- Scikit-learn
- Joblib

## How to Run

Train and compare models:

    ..\.venv\Scripts\python.exe .\train_model.py

Make a demonstration prediction:

    ..\.venv\Scripts\python.exe .\predict.py

## Concepts Learned

- Cross-validation
- StratifiedKFold
- cross_val_score
- Mean and standard deviation
- Model comparison
- Held-out test evaluation
- Model persistence