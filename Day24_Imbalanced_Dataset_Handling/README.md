
# Day 24: Imbalanced Dataset Handling

## Overview

This project compares a baseline Logistic Regression model
with a class-weighted model on a synthetic imbalanced dataset.

## Features

- Generates an imbalanced binary classification dataset
- Trains baseline and class-balanced models
- Calculates accuracy, precision, recall, and F1-score
- Creates confusion matrices
- Saves a comparison report
- Saves the balanced model with Joblib
- Loads the saved model for a prediction

## Technologies

- Python
- Scikit-learn
- Joblib

## Project Structure

    Day24_Imbalanced_Dataset_Handling/
    ├── model/
    │   └── balanced_model.joblib
    ├── reports/
    │   └── evaluation_report.txt
    ├── train_model.py
    ├── predict.py
    ├── requirements.txt
    └── README.md

## How to Run

Install dependencies:

    ..\.venv\Scripts\python.exe -m pip install -r requirements.txt

Train and compare models:

    ..\.venv\Scripts\python.exe .\train_model.py

Load the saved model and predict:

    ..\.venv\Scripts\python.exe .\predict.py

## Learning Outcomes

- Understanding class imbalance
- Comparing baseline and class-weighted models
- Measuring minority-class precision and recall
- Reading confusion matrices
- Saving and loading a fitted model
