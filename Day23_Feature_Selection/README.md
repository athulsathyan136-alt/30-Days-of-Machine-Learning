# Day 22: Hyperparameter Tuning - Digit Classification

## Overview

This project uses GridSearchCV to tune a Random Forest
classifier that recognises handwritten digits from 0 to 9.

## Features

- Loads the built-in handwritten digits dataset
- Splits data into training and testing sets
- Tests multiple Random Forest parameter combinations
- Uses three-fold cross-validation
- Selects the best-performing model settings
- Reports accuracy, precision, recall and F1-score
- Generates a confusion matrix
- Saves the tuned model using Joblib
- Saves a report of tuning and test results
- Reuses the best model for a prediction

## Technologies

- Python
- Scikit-learn
- Joblib

## Project Structure

    Day22_Hyperparameter_Tuning_Digits/
    ├── model/
    │   └── digit_model.joblib
    ├── reports/
    │   └── tuning_report.txt
    ├── train_tune.py
    ├── predict.py
    ├── requirements.txt
    └── README.md

## How to Run

Install dependencies:

    ..\.venv\Scripts\python.exe -m pip install -r requirements.txt

Train and tune the model:

    ..\.venv\Scripts\python.exe .\train_tune.py

Load the tuned model and predict:

    ..\.venv\Scripts\python.exe .\predict.py

## Learning Outcomes

- Understanding hyperparameters
- Using GridSearchCV
- Applying cross-validation
- Evaluating a multiclass classifier
- Saving and reusing a tuned model